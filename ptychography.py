import autograd.numpy as np
from autograd import grad
from mpi4py import MPI
import dxchange
import time
import datetime
import os
import h5py
import warnings
from scipy.ndimage import rotate as sp_rotate
from util import *
from misc import *
from propagate import *
from array_ops import *
from optimizers import *

PI = 3.1415927


def reconstruct_ptychography(fname, probe_pos, probe_size, obj_size, theta_st=0, theta_end=PI, n_theta=None, theta_downsample=None,
                             n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                             alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                             output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                             energy_ev=5000, psize_cm=1e-7, cpu_only=False, save_path='.',
                             core_parallelization=True, free_prop_cm=None, optimize_probe_defocusing=False,
                             probe_defocusing_learning_rate=1e-5,
                             multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=1,
                             dynamic_rate=True, probe_type='gaussian', probe_initial=None, probe_learning_rate=1e-3,
                             pupil_function=None, probe_circ_mask=0.9, finite_support_mask_path=None,
                             forward_algorithm='fresnel', dynamic_dropping=False, dropping_threshold=8e-5, shrink_cycle=None, shrink_threshold=1e-9,
                             n_dp_batch=20, object_type='normal', fresnel_approx=False, pure_projection=False, two_d_mode=False,
                             shared_file_object=True, reweighted_l1=False, optimizer='adam', interpolation='bilinear', save_stdout=False, use_checkpoint=True, binning=1, **kwargs):

    def calculate_loss(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm, this_i_theta, this_pos_batch, this_prj_batch):

        if optimize_probe_defocusing:
            h_probe = get_kernel(probe_defocus_mm * 1e6, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)
            probe_complex = probe_real + 1j * probe_imag
            probe_complex = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(probe_complex)) * h_probe))
            probe_real = np.real(probe_complex)
            probe_imag = np.imag(probe_complex)

        if not shared_file_object:
            obj_stack = np.stack([obj_delta, obj_beta], axis=3)
            if not two_d_mode:
                obj_rot = apply_rotation(obj_stack, coord_ls[this_i_theta],
                                         'arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta))
                # obj_rot = sp_rotate(obj_stack, theta, axes=(1, 2), reshape=False)
            else:
                obj_rot = obj_stack
            probe_pos_batch_ls = []
            exiting_ls = []
            i_dp = 0
            while i_dp < minibatch_size:
                probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
                i_dp += n_dp_batch

            # Pad if needed
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, probe_pos, probe_size_half)

            for k, pos_batch in enumerate(probe_pos_batch_ls):
                subobj_ls = []
                for j in range(len(pos_batch)):
                    pos = pos_batch[j]
                    pos = [int(x) for x in pos]
                    pos[0] = pos[0] + pad_arr[0, 0]
                    pos[1] = pos[1] + pad_arr[1, 0]
                    subobj = obj_rot[pos[0] - probe_size_half[0]:pos[0] - probe_size_half[0] + probe_size[0],
                             pos[1] - probe_size_half[1]:pos[1] - probe_size_half[1] + probe_size[1],
                             :, :]
                    subobj_ls.append(subobj)

                subobj_ls = np.stack(subobj_ls)
                exiting = multislice_propagate_batch_numpy(subobj_ls[:, :, :, :, 0], subobj_ls[:, :, :, :, 1], probe_real,
                                                           probe_imag, energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm,
                                                           obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                                           fresnel_approx=fresnel_approx, pure_projection=pure_projection)
                exiting_ls.append(exiting)
            exiting_ls = np.concatenate(exiting_ls, 0)
            loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)

        else:
            probe_pos_batch_ls = []
            exiting_ls = []
            i_dp = 0
            while i_dp < minibatch_size:
                probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
                i_dp += n_dp_batch

            pos_ind = 0
            for k, pos_batch in enumerate(probe_pos_batch_ls):
                subobj_ls_delta = obj_delta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                subobj_ls_beta = obj_beta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                exiting = multislice_propagate_batch_numpy(subobj_ls_delta, subobj_ls_beta, probe_real,
                                                           probe_imag, energy_ev, psize_cm * ds_level, kernel=h,
                                                           free_prop_cm=free_prop_cm,
                                                           obj_batch_shape=[len(pos_batch), *probe_size,
                                                                            this_obj_size[-1]],
                                                           fresnel_approx=fresnel_approx,
                                                           pure_projection=pure_projection)
                exiting_ls.append(exiting)
                pos_ind += len(pos_batch)
            exiting_ls = np.concatenate(exiting_ls, 0)
            loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)
            # dxchange.write_tiff(abs(exiting_ls._value[0]), output_folder + '/det/det', dtype='float32', overwrite=True)
            # raise

        # Regularization
        if reweighted_l1:
            if alpha_d not in [None, 0]:
                loss = loss + alpha_d * np.mean(weight_l1 * np.abs(obj_delta))
            if alpha_b not in [None, 0]:
                loss = loss + alpha_b * np.mean(weight_l1 * np.abs(obj_beta))
        else:
            if alpha_d not in [None, 0]:
                loss = loss + alpha_d * np.mean(np.abs(obj_delta))
            if alpha_b not in [None, 0]:
                loss = loss + alpha_b * np.mean(np.abs(obj_beta))
        if gamma not in [None, 0]:
            if shared_file_object:
                loss = loss + gamma * total_variation_3d(obj_delta, axis_offset=1)
            else:
                loss = loss + gamma * total_variation_3d(obj_delta, axis_offset=0)

        # Write convergence data
        global current_loss
        current_loss = loss._value
        f_conv.write('{},{},{},'.format(i_epoch, i_batch, current_loss))
        f_conv.flush()

        return loss

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()

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

        n_pos = len(probe_pos)
        probe_pos = np.array(probe_pos)
        probe_size_half = (np.array(probe_size) / 2).astype('int')
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
                print('Target folder {} exists.'.format(output_folder))
        comm.Barrier()

        # ================================================================================
        # Create optimizers.
        # ================================================================================
        if optimizer == 'adam':
            opt = AdamOptimizer([*this_obj_size, 2], output_folder=output_folder)
            optimizer_options_obj = {'step_size': learning_rate,
                                     'shared_file_object': shared_file_object}
        elif optimizer == 'gd':
            opt = GDOptimizer([*this_obj_size, 2], output_folder=output_folder)
            optimizer_options_obj = {'step_size': learning_rate,
                                     'dynamic_rate': True,
                                     'first_downrate_iteration': 20 * max([ceil(n_pos / (minibatch_size * n_ranks)), 1])}
        if shared_file_object:
            opt.create_file_objects(use_checkpoint=use_checkpoint)
        else:
            if use_checkpoint:
                try:
                    opt.restore_param_arrays_from_checkpoint()
                except:
                    opt.create_param_arrays()
            else:
                opt.create_param_arrays()

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
                                     not_first_level=not_first_level, initial_guess=initial_guess)
            else:
                obj.delta = obj_delta
                obj.beta = obj_beta

        # ================================================================================
        # Create gradient class.
        # ================================================================================
        gradient = Gradient(obj)
        if shared_file_object:
            gradient.create_file_object()
            gradient.initialize_gradient_file()
        else:
            gradient.initialize_array_with_values(np.zeros(this_obj_size), np.zeros(this_obj_size))

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
                mask.initialize_array_with_values(mask_arr)

        # ================================================================================
        # Initialize probe functions.
        # ================================================================================
        print_flush('Initialzing probe...', 0, rank, **stdout_options)
        probe_real, probe_imag = initialize_probe(probe_size, probe_type, pupil_function=pupil_function, probe_initial=probe_initial,
                             save_stdout=save_stdout, output_folder=output_folder, timestr=timestr,
                             save_path=save_path, fname=fname, **kwargs)

        # ================================================================================
        # generate Fresnel kernel.
        # ================================================================================
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)

        # ================================================================================
        # Set optimizer parameters.
        # ================================================================================
        opt_arg_ls = [0, 1]
        if probe_type == 'optimizable':
            opt_arg_ls = opt_arg_ls + [2, 3]
            opt_probe = GDOptimizer([*probe_size, 2], output_folder=output_folder)
            optimizer_options_probe = {'step_size': probe_learning_rate,
                                      'dynamic_rate': True,
                                      'first_downrate_iteration': 4 * max([ceil(n_pos / (minibatch_size * n_ranks)), 1])}
        if optimize_probe_defocusing:
            opt_arg_ls.append(4)
            opt_probe_defocus = GDOptimizer([1], output_folder=output_folder)
            optimizer_options_probe_defocus = {'step_size': probe_defocusing_learning_rate,
                                               'dynamic_rate': True,
                                               'first_downrate_iteration': 4 * max([ceil(n_pos / (minibatch_size * n_ranks)), 1])}
        probe_defocus_mm = np.array(0.0)

        # ================================================================================
        # Get gradient of loss function w.r.t. optimizable variables.
        # ================================================================================
        loss_grad = grad(calculate_loss, opt_arg_ls)

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
        m_p, v_p, m_pd, v_pd = (None, None, None, None)
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
                # Append randomly selected diffraction spots if necessary, so that a rank won't be given
                # spots from different angles in one batch.
                # When using shared file object, we must also ensure that all ranks deal with data at the
                # same angle at a time.
                if not shared_file_object and n_pos % minibatch_size != 0:
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-(n_pos % minibatch_size)],
                                                                    minibatch_size - (n_pos % minibatch_size),
                                                                    replace=False))
                elif shared_file_object and n_pos % n_tot_per_batch != 0:
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-(n_pos % n_tot_per_batch)],
                                                                    n_tot_per_batch - (n_pos % n_tot_per_batch),
                                                                    replace=False))
                if i == 0:
                    ind_list_rand = np.vstack([np.array([i_theta] * len(spots_ls)), spots_ls]).transpose()
                else:
                    ind_list_rand = np.concatenate(
                        [ind_list_rand, np.vstack([np.array([i_theta] * len(spots_ls)), spots_ls]).transpose()], axis=0)
            ind_list_rand = split_tasks(ind_list_rand, n_tot_per_batch)
            # ind_list_rand is in the format of [((5, 0), (5, 1), ...), ((17, 0), (17, 1), ..., (...))]
            #                                    |___________________|   |_____|
            #                       a batch for all ranks  _|               |_ (i_theta, i_spot)
            #                    (minibatch_size * n_ranks)
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
                                    obj_array=np.stack([obj.delta, obj.beta], axis=-1), optimizer=opt)

                # ================================================================================
                # Get scan position, rotation angle indices, and raw data for current batch.
                # ================================================================================
                t00 = time.time()
                if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                    n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                    ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                this_ind_batch = ind_list_rand[i_batch]
                this_i_theta = this_ind_batch[rank * minibatch_size, 0]
                this_ind_rank = np.sort(this_ind_batch[rank * minibatch_size:(rank + 1) * minibatch_size, 1])
                this_pos_batch = probe_pos[this_ind_rank]
                print_flush('Current rank is processing angle ID {}.'.format(this_i_theta), 0, rank, **stdout_options)

                t_prj_0 = time.time()
                this_prj_batch = prj[this_i_theta, this_ind_rank]
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
                    obj_rot = obj.read_chunks_from_file(this_pos_batch, probe_size_half, dset_2=obj.dset_rot)
                    print_flush('  Chunk reading done in {} s.'.format(time.time() - t_read_0), 0, rank, **stdout_options)
                    obj_delta = np.array(obj_rot[:, :, :, :, 0])
                    obj_beta = np.array(obj_rot[:, :, :, :, 1])
                    opt.get_params_from_file(this_pos_batch, probe_size_half)
                else:
                    obj_delta = obj.delta
                    obj_beta = obj.beta

                # Update weight for reweighted L1
                if shared_file_object:
                    weight_l1 = np.max(obj_delta) / (abs(obj_delta) + 1e-8)
                else:
                    if i_batch % 10 == 0: weight_l1 = np.max(obj_delta) / (abs(obj_delta) + 1e-8)

                # ================================================================================
                # Calculate object gradients.
                # ================================================================================
                t_grad_0 = time.time()
                grads = loss_grad(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm, this_i_theta, this_pos_batch, this_prj_batch)
                print_flush('  Gradient calculation done in {} s.'.format(time.time() - t_grad_0), 0, rank, **stdout_options)
                grads = list(grads)

                # ================================================================================
                # Reshape object gradient to [y, x, z, c] or [n, y, x, z, c] and average over
                # ranks.
                # ================================================================================
                if shared_file_object:
                    obj_grads = np.stack(grads[:2], axis=-1)
                else:
                    this_obj_grads = np.stack(grads[:2], axis=-1)
                    obj_grads = np.zeros_like(this_obj_grads)
                    comm.Barrier()
                    comm.Allreduce(this_obj_grads, obj_grads)
                obj_grads = obj_grads / n_ranks

                # ================================================================================
                # Update object function with optimizer if not shared_file_object; otherwise,
                # just save the gradient chunk into the gradient file.
                # ================================================================================
                if not shared_file_object:
                    effective_iter = i_batch // max([ceil(n_pos / (minibatch_size * n_ranks)), 1])
                    obj_temp = opt.apply_gradient(np.stack([obj_delta, obj_beta], axis=-1), obj_grads, effective_iter,
                                                            **optimizer_options_obj)
                    obj_delta = np.take(obj_temp, 0, axis=-1)
                    obj_beta = np.take(obj_temp, 1, axis=-1)
                else:
                    t_grad_write_0 = time.time()
                    gradient.write_chunks_to_file(this_pos_batch, np.take(obj_grads, 0, axis=-1),
                                                  np.take(obj_grads, 1, axis=-1), probe_size_half,
                                                  write_difference=False)
                    print_flush('  Gradient writing done in {} s.'.format(time.time() - t_grad_write_0), 0, rank, **stdout_options)
                # ================================================================================
                # Nonnegativity and phase/absorption-only constraints for non-shared-file-mode,
                # and update arrays in instance.
                # ================================================================================
                if not shared_file_object:
                    obj_delta = np.clip(obj_delta, 0, None)
                    obj_beta = np.clip(obj_beta, 0, None)
                    if object_type == 'absorption_only': obj_delta[...] = 0
                    if object_type == 'phase_only': obj_beta[...] = 0
                    obj.delta = obj_delta
                    obj.beta = obj_beta

                # ================================================================================
                # Optimize probe and other parameters if necessary.
                # ================================================================================
                if probe_type == 'optimizable':
                    this_probe_grads = np.stack(grads[2:4], axis=-1)
                    probe_grads = np.zeros_like(this_probe_grads)
                    comm.Allreduce(this_probe_grads, probe_grads)
                    probe_grads = probe_grads / n_ranks
                    probe_temp = opt_probe.apply_gradient(np.stack([probe_real, probe_imag], axis=-1), probe_grads, **optimizer_options_probe)
                    probe_real = np.take(probe_temp, 0, axis=-1)
                    probe_imag = np.take(probe_temp, 1, axis=-1)

                if optimize_probe_defocusing:
                    this_pd_grad = np.array(grads[len(opt_arg_ls) - 1])
                    pd_grads = np.array(0.0)
                    comm.Allreduce(this_pd_grad, pd_grads)
                    pd_grads = pd_grads / n_ranks
                    probe_defocus_mm = opt_probe_defocus.apply_gradient(probe_defocus_mm, pd_grads,
                                                                        **optimizer_options_probe_defocus)
                    print_flush('  Probe defocus is {} mm.'.format(probe_defocus_mm), 0, rank,
                                **stdout_options)

                # ================================================================================
                # For shared-file-mode, if finishing or above to move to a different angle,
                # rotate the gradient back, and use it to update the object at 0 deg. Then
                # update the object using gradient at 0 deg.
                # ================================================================================
                if shared_file_object and (i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != current_i_theta):
                    # coord_new = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                    #                                this_i_theta, reverse=True)
                    print_flush('  Rotating gradient dataset back...', 0, rank, **stdout_options)
                    t_rot_0 = time.time()
                    gradient.reverse_rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation)
                    comm.Barrier()
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
                    if shared_file_object:
                        dxchange.write_tiff(obj.dset[:, :, :, 0],
                                            fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                            dtype='float32', overwrite=True)
                    else:
                        dxchange.write_tiff(obj.delta,
                                            fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                            dtype='float32', overwrite=True)
                comm.Barrier()
                print_flush('Minibatch done in {} s; loss (rank 0) is {}.'.format(time.time() - t00, current_loss), 0, rank, **stdout_options)
                f_conv.write('{}\n'.format(time.time() - t_zero))
                f_conv.flush()

            if n_epochs == 'auto':
                    pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            i_epoch = i_epoch + 1

            average_loss = 0
            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    time.time() - t0, time.time() - t_zero), **stdout_options)
            if rank == 0 and save_intermediate:
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
                    dxchange.write_tiff(obj.delta,
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(obj.beta,
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.sqrt(probe_real ** 2 + probe_imag ** 2),
                                        fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.arctan2(probe_imag, probe_real),
                                        fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
            print_flush('Current iteration finished.', 0, rank, **stdout_options)
        comm.Barrier()
