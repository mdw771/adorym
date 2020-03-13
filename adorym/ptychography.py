import numpy as np
from mpi4py import MPI
import dxchange
import time
import datetime
import os
import h5py
import gc
import warnings

from adorym.util import *
from adorym.misc import *
from adorym.propagate import *
from adorym.array_ops import *
from adorym.optimizers import *
from adorym.differentiator import *
import adorym.wrappers as w
import adorym.global_settings
from adorym.forward_model import *

PI = 3.1415927


def reconstruct_ptychography(
        # ______________________________________
        # |Raw data and experimental parameters|________________________________
        fname, probe_pos, probe_size, obj_size, theta_st=0, theta_end=PI, n_theta=None, theta_downsample=None,
        energy_ev=5000, psize_cm=1e-7, free_prop_cm=None, raw_data_type='magnitude',
        # ___________________________
        # |Reconstruction parameters|___________________________________________
        n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200, alpha_d=None, alpha_b=None,
        gamma=1e-6, minibatch_size=None, multiscale_level=1, n_epoch_final_pass=None,
        initial_guess=None, n_batch_per_update=1, reweighted_l1=False, interpolation='bilinear',
        # __________________________
        # |Object optimizer options|____________________________________________
        optimizer='adam', learning_rate=1e-5,
        # ___________________________
        # |Finite support constraint|___________________________________________
        finite_support_mask_path=None, shrink_cycle=None, shrink_threshold=1e-9,
        # ___________________
        # |Object contraints|___________________________________________________
        object_type='normal',
        # _______________
        # |Forward model|_______________________________________________________
        forward_algorithm='fresnel', binning=1, fresnel_approx=True, pure_projection=False, two_d_mode=False,
        probe_type='gaussian', probe_initial=None, loss_function_type='lsq',
        # _____
        # |I/O|_________________________________________________________________
        save_path='.', output_folder=None, save_intermediate=False, save_history=False, use_checkpoint=True,
        save_stdout=False,
        # _____________
        # |Performance|_________________________________________________________
        cpu_only=False, core_parallelization=True, shared_file_object=True, n_dp_batch=20,
        # _________________________
        # |Other optimizer options|_____________________________________________
        probe_learning_rate=1e-3,
        optimize_probe_defocusing=False, probe_defocusing_learning_rate=1e-5,
        optimize_probe_pos_offset=False, probe_pos_offset_learning_rate=1,
        optimize_all_probe_pos=False, all_probe_pos_learning_rate=1e-2,
        # ________________
        # |Other settings|______________________________________________________
        dynamic_rate=True, pupil_function=None, probe_circ_mask=0.9, dynamic_dropping=False, dropping_threshold=8e-5,
        backend='autograd', **kwargs,):
        # ______________________________________________________________________

    """
    Notes:
        1. Input data are assumed to be contained in an HDF5 under 'exchange/data', as a 4D dataset of
           shape [n_theta, n_spots, detector_size_y, detector_size_x].
        2. Full-field reconstruction is treated as ptychography. If the image is not divided, the programs
           runs as if it is dealing with ptychography with only 1 spot per angle.
        3. Full-field reconstruction with minibatch_size > 1 but without image dividing is not supported.
           In this case, minibatch_size will be forced to be 1, so that each rank process only one
           rotation angle's image at a time. To perform large fullfield reconstruction efficiently,
           divide the data into sub-chunks.
        4. Full-field reconstruction using shared_file_mode but without image dividing is not recommended
           even if minibatch_size is 1. In shared_file_mode, all ranks process data from the same rotation
           angle in each synchronized batch. Doing this will cause all ranks to process the same data.
           To perform large fullfield reconstruction efficiently, divide the data into sub-chunks.
    """

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()
    global_settings.backend = backend
    device_obj = None if cpu_only else 0
    device_obj = w.get_device(device_obj)

    timestr = str(datetime.datetime.today())
    timestr = timestr[:timestr.find('.')]
    for i in [':', '-', ' ']:
        if i == ' ':
            timestr = timestr.replace(i, '_')
        else:
            timestr = timestr.replace(i, '')

    # ================================================================================
    # Create pointer for raw data.
    # ================================================================================
    t0 = time.time()
    print_flush('Reading data...', 0, rank)
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj = f['exchange/data']
    if n_theta is None:
        n_theta = prj.shape[0]
    if two_d_mode:
        n_theta = 1
    prj_theta_ind = np.arange(n_theta, dtype=int)
    theta = -np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    if theta_downsample is not None:
        theta = theta[::theta_downsample]
        prj_theta_ind = prj_theta_ind[::theta_downsample]
        n_theta = len(theta)
    original_shape = [n_theta, *prj.shape[1:]]

    print_flush('Data reading: {} s'.format(time.time() - t0), 0, rank)
    print_flush('Data shape: {}'.format(original_shape), 0, rank)
    comm.Barrier()

    not_first_level = False
    stdout_options = {'save_stdout': save_stdout, 'output_folder': output_folder, 
                      'timestamp': timestr}

    n_pos = len(probe_pos)
    probe_pos = np.array(probe_pos)

    # ================================================================================
    # Batching check.
    # ================================================================================
    if minibatch_size > 1 and n_pos == 1:
        warnings.warn('It seems that you are processing undivided fullfield data with'
                      'minibatch > 1. A rank can only process data from the same rotation'
                      'angle at a time. I am setting minibatch_size to 1.')
        minibatch_size = 1
    if shared_file_object and n_pos == 1:
        warnings.warn('It seems that you are processing undivided fullfield data with'
                      'shared_file_object=True. In shared-file mode, all ranks must'
                      'process data from the same rotation angle in each synchronized'
                      'batch.')

    # ================================================================================
    # Set output folder name if not specified.
    # ================================================================================
    if output_folder is None:
        output_folder = 'recon_{}'.format(timestr)
        if abs(PI - theta_end) < 1e-3:
            output_folder += '_180'
    print_flush('Output folder is {}'.format(output_folder), 0, rank)

    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

    for ds_level in range(multiscale_level - 1, -1, -1):

        # ================================================================================
        # Set metadata.
        # ================================================================================
        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), 0, rank, **stdout_options)
        comm.Barrier()

        prj_shape = original_shape

        if ds_level > 1:
            this_obj_size = [int(x / ds_level) for x in obj_size]
        else:
            this_obj_size = obj_size

        dim_y, dim_x = prj_shape[-2:]
        if minibatch_size is None:
            minibatch_size = n_pos
        comm.Barrier()

        # ================================================================================
        # Create output directory.
        # ================================================================================
        if rank == 0:
            try:
                os.makedirs(os.path.join(output_folder))
            except:
                print_flush('Target folder {} exists.'.format(output_folder), 0, rank, **stdout_options)
        comm.Barrier()

        # ================================================================================
        # generate Fresnel kernel.
        # ================================================================================
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)

        # ================================================================================
        # Read rotation data.
        # ================================================================================
        try:
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)
        except:
            if rank == 0:
                print_flush('Saving rotation coordinates...', 0, rank, **stdout_options)
                save_rotation_lookup(this_obj_size, n_theta)
            comm.Barrier()
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)

        # ================================================================================
        # Unify random seed for all threads.
        # ================================================================================
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        # ================================================================================
        # Get checkpointed parameters.
        # ================================================================================
        starting_epoch, starting_batch = (0, 0)
        needs_initialize = False if use_checkpoint else True
        if use_checkpoint and shared_file_object:
            try:
                starting_epoch, starting_batch = restore_checkpoint(output_folder, shared_file_object)
            except:
                needs_initialize = True

        elif use_checkpoint and (not shared_file_object):
            try:
                starting_epoch, starting_batch, obj_delta, obj_beta = restore_checkpoint(output_folder, shared_file_object, opt)
            except:
                needs_initialize = True

        # ================================================================================
        # Create object class.
        # ================================================================================
        obj = ObjectFunction([*this_obj_size, 2], shared_file_object=shared_file_object,
                             output_folder=output_folder, ds_level=ds_level, object_type=object_type)
        if shared_file_object:
            obj.create_file_object(use_checkpoint)
            obj.create_temporary_file_object()
            if needs_initialize:
                obj.initialize_file_object(save_stdout=save_stdout, timestr=timestr,
                                           not_first_level=not_first_level, initial_guess=initial_guess)
        else:
            if needs_initialize:
                obj.initialize_array(save_stdout=save_stdout, timestr=timestr,
                                     not_first_level=not_first_level, initial_guess=initial_guess, device=device_obj)
            else:
                obj.delta = obj_delta
                obj.beta = obj_beta

        # ================================================================================
        # Create object function optimizer.
        # ================================================================================
        if optimizer == 'adam':
            opt = AdamOptimizer([*this_obj_size, 2], output_folder=output_folder)
            optimizer_options_obj = {'step_size': learning_rate, 'verbose': False}
        elif optimizer == 'gd':
            opt = GDOptimizer([*this_obj_size, 2], output_folder=output_folder)
            optimizer_options_obj = {'step_size': learning_rate,
                                     'dynamic_rate': True,
                                     'first_downrate_iteration': 20}
        if shared_file_object:
            opt.create_file_objects(use_checkpoint=use_checkpoint)
        else:
            if use_checkpoint:
                try:
                    opt.restore_param_arrays_from_checkpoint(device=device_obj)
                except:
                    opt.create_param_arrays(device=device_obj)
            else:
                opt.create_param_arrays(device=device_obj)

        # ================================================================================
        # Create forward model class.
        # ================================================================================
        forward_model = PtychographyModel(loss_function_type=loss_function_type,
                                          shared_file_object=shared_file_object,
                                          device=device_obj, common_vars_dict=locals(),
                                          raw_data_type=raw_data_type)
        if reweighted_l1:
            forward_model.add_reweighted_l1_norm(alpha_d, alpha_b, None)
        else:
            if alpha_d not in [0, None]: forward_model.add_l1_norm(alpha_d, alpha_b)
        if gamma not in [0, None]: forward_model.add_tv(gamma)

        # ================================================================================
        # Create gradient class.
        # ================================================================================
        gradient = Gradient(obj)
        if shared_file_object:
            gradient.create_file_object()
            gradient.initialize_gradient_file()
        else:
            gradient.initialize_array_with_values(np.zeros(this_obj_size), np.zeros(this_obj_size), device=device_obj)

        # ================================================================================
        # If a finite support mask path is specified (common for full-field imaging),
        # create an instance of monochannel mask class. While finite_support_mask_path
        # has to point to a 3D tiff file, the mask will be written as an HDF5 if
        # share_file_mode is True.
        # ================================================================================
        mask = None
        if finite_support_mask_path is not None:
            mask = Mask(this_obj_size, finite_support_mask_path, shared_file_object=shared_file_object,
                        output_folder=output_folder, ds_level=ds_level)
            if shared_file_object:
                mask.create_file_object(use_checkpoint=use_checkpoint)
                mask.initialize_file_object()
            else:
                mask_arr = dxchange.read_tiff(finite_support_mask_path)
                mask.initialize_array_with_values(mask_arr, device=device_obj)

        # ================================================================================
        # Initialize probe functions.
        # ================================================================================
        print_flush('Initialzing probe...', 0, rank, **stdout_options)
        probe_real, probe_imag = initialize_probe(probe_size, probe_type, pupil_function=pupil_function, probe_initial=probe_initial,
                             save_stdout=save_stdout, output_folder=output_folder, timestr=timestr,
                             save_path=save_path, fname=fname, raw_data_type=raw_data_type, **kwargs)
        probe_real = w.create_variable(probe_real, device=device_obj)
        probe_imag = w.create_variable(probe_imag, device=device_obj)

        # ================================================================================
        # Create other optimizers (probe, probe defocus, probe positions, etc.).
        # ================================================================================
        opt_args_ls = [0, 1]
        if probe_type == 'optimizable':
            opt_probe = AdamOptimizer([*probe_size, 2], output_folder=output_folder)
            opt_probe.create_param_arrays(device=device_obj)
            optimizer_options_probe = {'step_size': probe_learning_rate}
            opt_probe.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls = opt_args_ls + [2, 3]

        probe_defocus_mm = w.create_variable(0.0)
        if optimize_probe_defocusing:
            opt_probe_defocus = GDOptimizer([1], output_folder=output_folder)
            opt_probe_pos.create_param_arrays(device=device_obj)
            optimizer_options_probe_defocus = {'step_size': probe_defocusing_learning_rate,
                                               'dynamic_rate': True,
                                               'first_downrate_iteration': 4 * max([ceil(n_pos / (minibatch_size * n_ranks)), 1])}
            opt_probe_defocus.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls.append(4)

        probe_pos_offset = w.zeros([n_theta, 2], requires_grad=True, device=device_obj)
        if optimize_probe_pos_offset:
            assert optimize_all_probe_pos == False
            opt_probe_pos_offset = GDOptimizer(probe_pos_offset.shape, output_folder=output_folder)
            opt_probe_pos.create_param_arrays(device=device_obj)
            optimizer_options_probe_pos_offset = {'step_size': probe_pos_offset_learning_rate,
                                                  'dynamic_rate': False}
            opt_probe_pos_offset.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls.append(5)

        probe_pos_int = np.round(probe_pos).astype(int)
        probe_pos_correction = w.create_variable(probe_pos - probe_pos_int, requires_grad=optimize_all_probe_pos, device=device_obj)
        if optimize_all_probe_pos:
            assert optimize_probe_pos_offset == False
            assert shared_file_object == False
            # probe_pos_correction = np.full([n_theta, n_pos, 2], 5).astype(float)
            opt_probe_pos = AdamOptimizer(probe_pos_correction.shape, output_folder=output_folder)
            opt_probe_pos.create_param_arrays(device=device_obj)
            optimizer_options_probe_pos = {'step_size': all_probe_pos_learning_rate}
            opt_probe_pos.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls.append(9)

        # ================================================================================
        # Get gradient of loss function w.r.t. optimizable variables.
        # ================================================================================
        diff = Differentiator()
        calculate_loss = forward_model.get_loss_function()
        diff.create_loss_node(calculate_loss, opt_args_ls)

        # ================================================================================
        # Save convergence data.
        # ================================================================================
        if rank == 0:
            try:
                os.makedirs(os.path.join(output_folder, 'convergence'))
            except:
                pass
        comm.Barrier()
        f_conv = open(os.path.join(output_folder, 'convergence', 'loss_rank_{}.txt'.format(rank)), 'w')
        f_conv.write('i_epoch,i_batch,loss,time\n')

        # ================================================================================
        # Create parameter summary file.
        # ================================================================================
        print_flush('Optimizer started.', 0, rank, **stdout_options)
        if rank == 0:
            create_summary(output_folder, locals(), preset='ptycho')

        # ================================================================================
        # Start outer (epoch) loop.
        # ================================================================================
        cont = True
        i_epoch = starting_epoch
        i_full_angle = 0
        while cont:
            n_pos = len(probe_pos)
            n_spots = n_theta * n_pos
            n_tot_per_batch = minibatch_size * n_ranks
            n_batch = int(np.ceil(float(n_spots) / n_tot_per_batch))

            t0 = time.time()
            spots_ls = range(n_spots)
            ind_list_rand = []

            t00 = time.time()
            print_flush('Allocating jobs over threads...', 0, rank, **stdout_options)
            # Make a list of all thetas and spot positions'
            np.random.seed(i_epoch)
            comm.Barrier()
            if not two_d_mode:
                theta_ls = np.arange(n_theta)
                np.random.shuffle(theta_ls)
            else:
                theta_ls = np.linspace(0, 2 * PI, prj.shape[0])
                theta_ls = abs(theta_ls - theta_st) < 1e-5
                i_theta = np.nonzero(theta_ls)[0][0]
                theta_ls = np.array([i_theta])

            # ================================================================================
            # Put diffraction spots from all angles together, and divide into minibatches.
            # ================================================================================
            for i, i_theta in enumerate(theta_ls):
                spots_ls = range(n_pos)
                # ================================================================================
                # Append randomly selected diffraction spots if necessary, so that a rank won't be given
                # spots from different angles in one batch.
                # When using shared file object, we must also ensure that all ranks deal with data at the
                # same angle at a time.
                # ================================================================================
                if not shared_file_object and n_pos % minibatch_size != 0:
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls,
                                                                    minibatch_size - (n_pos % minibatch_size),
                                                                    replace=False))
                elif shared_file_object and n_pos % n_tot_per_batch != 0:
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls,
                                                                    n_tot_per_batch - (n_pos % n_tot_per_batch),
                                                                    replace=False))
                # ================================================================================
                # Create task list for the current angle.
                # ind_list_rand is in the format of [((5, 0), (5, 1), ...), ((17, 0), (17, 1), ..., (...))]
                #                                    |___________________|   |_____|
                #                       a batch for all ranks  _|               |_ (i_theta, i_spot)
                #                    (minibatch_size * n_ranks)
                # ================================================================================
                if i == 0:
                    ind_list_rand = np.vstack([np.array([i_theta] * len(spots_ls)), spots_ls]).transpose()
                else:
                    ind_list_rand = np.concatenate(
                        [ind_list_rand, np.vstack([np.array([i_theta] * len(spots_ls)), spots_ls]).transpose()], axis=0)
            ind_list_rand = split_tasks(ind_list_rand, n_tot_per_batch)

            print_flush('Allocation done in {} s.'.format(time.time() - t00), 0, rank, **stdout_options)

            current_i_theta = 0
            for i_batch in range(starting_batch, n_batch):

                # ================================================================================
                # Initialize.
                # ================================================================================
                print_flush('Epoch {}, batch {} of {} started.'.format(i_epoch, i_batch, n_batch), 0, rank, **stdout_options)
                opt.i_batch = 0

                # ================================================================================
                # Save checkpoint.
                # ================================================================================
                if shared_file_object:
                    save_checkpoint(i_epoch, i_batch, output_folder, shared_file_object=True,
                                    obj_array=None, optimizer=opt)
                    obj.f.flush()
                else:
                    save_checkpoint(i_epoch, i_batch, output_folder, shared_file_object=False,
                                    obj_array=w.to_numpy(w.stack([obj.delta, obj.beta], axis=-1)),
                                    optimizer=opt)

                # ================================================================================
                # Get scan position, rotation angle indices, and raw data for current batch.
                # ================================================================================
                t00 = time.time()
                if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                    n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                    ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                this_ind_batch_allranks = ind_list_rand[i_batch]
                this_i_theta = this_ind_batch_allranks[rank * minibatch_size, 0]
                this_ind_batch = np.sort(this_ind_batch_allranks[rank * minibatch_size:(rank + 1) * minibatch_size, 1])
                print_flush('Current rank is processing angle ID {}.'.format(this_i_theta), 0, rank, **stdout_options)

                # Apply offset correction
                this_pos_batch = probe_pos_int[this_ind_batch]

                t_prj_0 = time.time()
                this_prj_batch = prj[this_i_theta, this_ind_batch]
                print_flush('  Raw data reading done in {} s.'.format(time.time() - t_prj_0), 0, rank, **stdout_options)

                # ================================================================================
                # In shared file mode, if moving to a new angle, rotate the HDF5 object and saved
                # the rotated object into the temporary file object.
                # ================================================================================
                if shared_file_object and this_i_theta != current_i_theta:
                    current_i_theta = this_i_theta
                    print_flush('  Rotating dataset...', 0, rank, **stdout_options)
                    t_rot_0 = time.time()
                    obj.rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation, dset_2=obj.dset_rot)
                    # opt.rotate_files(coord_ls[this_i_theta], interpolation=interpolation)
                    # if mask is not None: mask.rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation)
                    comm.Barrier()
                    print_flush('  Dataset rotation done in {} s.'.format(time.time() - t_rot_0), 0, rank, **stdout_options)

                if ds_level > 1:
                    this_prj_batch = this_prj_batch[:, :, ::ds_level, ::ds_level]
                comm.Barrier()

                if shared_file_object:
                    # ================================================================================
                    # Get values for local chunks of object_delta and beta; interpolate and read directly from HDF5
                    # ================================================================================
                    t_read_0 = time.time()
                    obj_rot = obj.read_chunks_from_file(this_pos_batch, probe_size, dset_2=obj.dset_rot, device=device_obj)
                    print_flush('  Chunk reading done in {} s.'.format(time.time() - t_read_0), 0, rank, **stdout_options)
                    obj_delta = obj_rot[:, :, :, :, 0]
                    obj_beta = obj_rot[:, :, :, :, 1]
                    opt.get_params_from_file(this_pos_batch, probe_size)
                else:
                    obj_delta = obj.delta
                    obj_beta = obj.beta

                # ================================================================================
                # Update weight for reweighted l1 if necessary
                # ================================================================================
                if reweighted_l1:
                    if shared_file_object:
                        weight_l1 = w.max(obj_delta) / (w.abs(obj_delta) + 1e-8)
                    else:
                        if i_batch % 10 == 0: weight_l1 = w.max(obj_delta) / (w.abs(obj_delta) + 1e-8)
                    forward_model.update_l1_weight(weight_l1)

                # ================================================================================
                # Calculate object gradients.
                # ================================================================================
                t_grad_0 = time.time()
                grad_func_args = {}
                for arg in forward_model.argument_ls:
                    grad_func_args[arg] = locals()[arg]
                grads = diff.get_gradients(**grad_func_args)
                print_flush('  Gradient calculation done in {} s.'.format(time.time() - t_grad_0), 0, rank, **stdout_options)
                grads = list(grads)

                # ================================================================================
                # Reshape object gradient to [y, x, z, c] or [n, y, x, z, c] and average over
                # ranks.
                # ================================================================================
                if shared_file_object:
                    obj_grads = w.stack(grads[:2], axis=-1)
                else:
                    this_obj_grads = w.stack(grads[:2], axis=-1)
                    obj_grads = w.zeros_like(this_obj_grads, requires_grad=False)
                    comm.Barrier()
                    obj_grads = comm.allreduce(this_obj_grads)
                obj_grads = obj_grads / n_ranks

                # ================================================================================
                # Update object function with optimizer if not shared_file_object; otherwise,
                # just save the gradient chunk into the gradient file.
                # ================================================================================
                if not shared_file_object:
                    obj_temp = opt.apply_gradient(w.stack([obj_delta, obj_beta], axis=-1), obj_grads, i_full_angle,
                                                            **optimizer_options_obj)
                    obj_delta, obj_beta = w.split_channel(obj_temp)
                else:
                    t_grad_write_0 = time.time()
                    gradient.write_chunks_to_file(this_pos_batch, *w.split_channel(obj_grads), probe_size,
                                                  write_difference=False)
                    print_flush('  Gradient writing done in {} s.'.format(time.time() - t_grad_write_0), 0, rank, **stdout_options)
                # ================================================================================
                # Nonnegativity and phase/absorption-only constraints for non-shared-file-mode,
                # and update arrays in instance.
                # ================================================================================
                if not shared_file_object:
                    obj_delta = w.clip(obj_delta, 0, None)
                    obj_beta = w.clip(obj_beta, 0, None)
                    if object_type == 'absorption_only': obj_delta *= 0
                    if object_type == 'phase_only': obj_beta *= 0
                    obj.delta = obj_delta
                    obj.beta = obj_beta

                # ================================================================================
                # Optimize probe and other parameters if necessary.
                # ================================================================================
                if probe_type == 'optimizable':
                    this_probe_grads = w.stack(grads[2:4], axis=-1)
                    probe_grads = w.zeros_like(this_probe_grads, requires_grad=False)
                    probe_grads = comm.allreduce(this_probe_grads)
                    probe_grads = probe_grads / n_ranks
                    probe_temp = opt_probe.apply_gradient(w.stack([probe_real, probe_imag], axis=-1), probe_grads, i_full_angle, **optimizer_options_probe)
                    probe_real, probe_imag = w.split_channel(probe_temp)

                if optimize_probe_defocusing:
                    this_pd_grad = grads[opt_probe_defocus.index_in_grad_returns]
                    pd_grads = w.create_variable(0.0)
                    pd_grads = comm.Allreduce(this_pd_grad)
                    pd_grads = pd_grads / n_ranks
                    probe_defocus_mm = opt_probe_defocus.apply_gradient(probe_defocus_mm, pd_grads, i_full_angle,
                                                                        **optimizer_options_probe_defocus)
                    print_flush('  Probe defocus is {} mm.'.format(probe_defocus_mm), 0, rank,
                                **stdout_options)

                if optimize_probe_pos_offset:
                    this_pos_offset_grad = grads[opt_probe_pos_offset.index_in_grad_returns]
                    pos_offset_grads = w.zeros_like(probe_pos_offset, requires_grad=False)
                    pos_offset_grads = comm.allreduce(this_pos_offset_grad)
                    pos_offset_grads = pos_offset_grads / n_ranks
                    probe_pos_offset = opt_probe_pos_offset.apply_gradient(probe_pos_offset, pos_offset_grads, i_full_angle,
                                                                        **optimizer_options_probe_pos_offset)

                if optimize_all_probe_pos:
                    this_all_pos_grad = grads[opt_probe_pos.index_in_grad_returns]
                    all_pos_grads = w.zeros_like(probe_pos_correction, requires_grad=False)
                    all_pos_grads = comm.allreduce(this_all_pos_grad)
                    all_pos_grads = all_pos_grads / n_ranks
                    probe_pos_correction = opt_probe_pos.apply_gradient(probe_pos_correction, all_pos_grads, i_full_angle,
                                                                        **optimizer_options_probe_pos)
                # ================================================================================
                # For shared-file-mode, if finishing or above to move to a different angle,
                # rotate the gradient back, and use it to update the object at 0 deg. Then
                # update the object using gradient at 0 deg.
                # ================================================================================
                if shared_file_object and (i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != current_i_theta):
                    coord_new = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                                   this_i_theta, reverse=True)
                    print_flush('  Rotating gradient dataset back...', 0, rank, **stdout_options)
                    t_rot_0 = time.time()
                    # dxchange.write_tiff(gradient.dset[:, :, :, 0], 'adhesin/test_shared_file/grad_prerot', dtype='float32')
                    # gradient.reverse_rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation)
                    gradient.rotate_data_in_file(coord_new, interpolation=interpolation)
                    # dxchange.write_tiff(gradient.dset[:, :, :, 0], 'adhesin/test_shared_file/grad_postrot', dtype='float32')
                    # comm.Barrier()
                    print_flush('  Gradient rotation done in {} s.'.format(time.time() - t_rot_0), 0, rank, **stdout_options)

                    t_apply_grad_0 = time.time()
                    opt.apply_gradient_to_file(obj, gradient, **optimizer_options_obj)
                    print_flush('  Object update done in {} s.'.format(time.time() - t_apply_grad_0), 0, rank, **stdout_options)
                    gradient.initialize_gradient_file()

                # ================================================================================
                # Apply finite support mask if specified.
                # ================================================================================
                if mask is not None:
                    if not shared_file_object:
                        obj.apply_finite_support_mask_to_array(mask)
                    else:
                        obj.apply_finite_support_mask_to_file(mask)
                    print_flush('  Mask applied.', 0, rank, **stdout_options)

                # ================================================================================
                # Update finite support mask if necessary.
                # ================================================================================
                if mask is not None and shrink_cycle is not None:
                    if i_batch % shrink_cycle == 0 and i_batch > 0:
                        if shared_file_object:
                            mask.update_mask_file(obj, shrink_threshold)
                        else:
                            mask.update_mask_array(obj, shrink_threshold)
                        print_flush('  Mask updated.', 0, rank, **stdout_options)

                # ================================================================================
                # Save intermediate object.
                # ================================================================================
                if rank == 0 and save_intermediate:
                    intermediate_fname = 'delta_{}_{}'.format(i_epoch, i_batch) if save_history else 'delta'
                    if shared_file_object:
                        dxchange.write_tiff(obj.dset[:, :, :, 0],
                                            fname=os.path.join(output_folder, 'intermediate', intermediate_fname),
                                            dtype='float32', overwrite=True)
                    else:
                        dxchange.write_tiff(w.to_numpy(obj.delta),
                                            fname=os.path.join(output_folder, 'intermediate', intermediate_fname),
                                            dtype='float32', overwrite=True)
                    if optimize_probe_pos_offset:
                        f_offset = open(os.path.join(output_folder, 'probe_pos_offset.txt'), 'a' if i_batch > 0 or i_epoch > 0 else 'w')
                        f_offset.write('{:4d}, {:4d}, {}\n'.format(i_epoch, i_batch, list(w.to_numpy(probe_pos_offset).flatten())))
                        f_offset.close()
                    elif optimize_all_probe_pos:
                        if not os.path.exists(os.path.join(output_folder, 'intermediate')):
                            os.makedirs(os.path.join(output_folder, 'intermediate'))
                        for i_theta_pos in range(n_theta):
                            np.savetxt(os.path.join(output_folder, 'intermediate',
                                                    'probe_pos_correction_{}_{}_{}.txt'.format(i_epoch, i_batch, i_theta_pos)),
                                                    w.to_numpy(probe_pos_correction[i_theta_pos]))

                comm.Barrier()
                current_loss = forward_model.current_loss
                print_flush('Minibatch done in {} s; loss (rank 0) is {}.'.format(time.time() - t00, current_loss), 0, rank, **stdout_options)
                f_conv.write('{},{},{},{}\n'.format(i_epoch, i_batch, current_loss, time.time() - t_zero))
                f_conv.flush()

                # ================================================================================
                # Update full-angle count.
                # ================================================================================
                if i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != current_i_theta: i_full_angle += 1

            # ================================================================================
            # Stopping criterion.
            # ================================================================================
            if n_epochs == 'auto':
                    pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            average_loss = 0
            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    time.time() - t0, time.time() - t_zero),
                0, rank, **stdout_options)
            i_epoch = i_epoch + 1

            # ================================================================================
            # Save reconstruction after an epoch.
            # ================================================================================
            if rank == 0:
                if shared_file_object:
                    dxchange.write_tiff(obj.dset[:, :, :, 0],
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(obj.dset[:, :, :, 1],
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.sqrt(probe_real ** 2 + probe_imag ** 2),
                                        fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.arctan2(probe_imag, probe_real),
                                        fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                else:
                    dxchange.write_tiff(w.to_numpy(obj.delta),
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(w.to_numpy(obj.beta),
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(w.to_numpy(w.sqrt(probe_real ** 2 + probe_imag ** 2)),
                                        fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(w.to_numpy(w.arctan2(probe_imag, probe_real)),
                                        fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
            print_flush('Current iteration finished.', 0, rank, **stdout_options)
        comm.Barrier()
