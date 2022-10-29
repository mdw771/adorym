import numpy as np
import dxchange
import time
import datetime
import os
import h5py
import gc
import warnings
import pickle

from adorym.util import *
from adorym.misc import *
from adorym.propagate import *
from adorym.array_ops import *
from adorym.optimizers import *
from adorym.differentiator import *
import adorym.wrappers as w
import adorym.global_settings as global_settings
from adorym.forward_model import *
from adorym.conventional import *

project_config = check_config_indept_mpi()
try:
    independent_mpi = project_config['independent_mpi']
except:
    independent_mpi = False

try:
    if independent_mpi:
        raise Exception
    from mpi4py import MPI
except:
    from adorym.pseudo import MPI

PI = 3.1415927


def simulate_ptychography(
        # ______________________________________
        # |Raw data and experimental parameters|________________________________
        fname, obj_size, probe_pos=None, probe_pos_ls=None, probe_size=(256, 256), theta_st=0, theta_end=PI, n_theta=None, theta_downsample=None,
        energy_ev=None, psize_cm=None, free_prop_cm=None,
        raw_data_type='magnitude',  # Choose from 'magnitude' or 'intensity'
        is_minus_logged=False,  # Select True if raw data (usually conventional tomography) is minus-logged
        slice_pos_cm_ls=None,
        # ___________________________
        # |Reconstruction parameters|___________________________________________
        n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200, alpha_d=None, alpha_b=None,
        gamma=1e-6, minibatch_size=None, multiscale_level=1, n_epoch_final_pass=None,
        initial_guess=None,
        random_guess_means_sigmas=(8.7e-7, 5.1e-8, 1e-7, 1e-8),
        # Give as (mean_delta, mean_beta, sigma_delta, sigma_beta) or (mean_mag, mean_phase, sigma_mag, sigma_phase)
        n_batch_per_update=1, reweighted_l1=False, interpolation='bilinear',
        update_scheme='immediate',  # Choose from 'immediate' or 'per_angle'
        unknown_type='delta_beta',  # Choose from 'delta_beta' or 'real_imag'
        randomize_probe_pos=False,
        common_probe_pos=True,  # Set to False if the values/number of probe positions vary with projection angle
        fix_object=False,  # Do not update the object, just update other parameters
        # __________________________
        # |Object optimizer options|____________________________________________
        optimize_object=True,
        # Keep True in most cases. Setting to False forbids the object from being updated using gradients, which
        # might be desirable when you just want to refine parameters for other reconstruction algorithms.
        optimizer='adam',  # Choose from 'gd' or 'adam' or 'curveball'
        learning_rate=1e-5,
        update_using_external_algorithm=None,
        # ___________________________
        # |Finite support constraint|___________________________________________
        finite_support_mask_path=None, shrink_cycle=None, shrink_threshold=1e-9,
        # ___________________
        # |Object contraints|___________________________________________________
        object_type='normal',  # Choose from 'normal', 'phase_only', or 'absorption_only
        non_negativity=False,
        # _______________
        # |Forward model|_______________________________________________________
        forward_model='auto',
        forward_algorithm='fresnel',  # Choose from 'fresnel' or 'ctf'
        # ---- CTF parameters ----
        ctf_lg_kappa=1.7,  # This is the common log of kappa, i.e. kappa = 10 ** ctf_lg_kappa
        # ------------------------
        binning=1, fresnel_approx=True, pure_projection=False, two_d_mode=False,
        probe_type='gaussian',  # Choose from 'gaussian', 'plane', 'ifft', 'aperture_defocus', 'supplied'
        probe_initial=None,  # Give as [probe_mag, probe_phase]
        probe_extra_defocus_cm=None,
        n_probe_modes=1,
        rescale_probe_intensity=False,
        loss_function_type='lsq',  # Choose from 'lsq' or 'poisson'
        poisson_multiplier=1.,
        # Intensity scaling factor in Poisson loss function. If intensity data is normalized, this should be the
        # average number of incident photons per pixel.
        beamstop=None,
        normalize_fft=False,
        # Use False for simulated data generated without normalization. Normalize for Fraunhofer FFT only
        safe_zone_width=0,
        scale_ri_by_k=True,
        sign_convention=1,
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        fourier_disparity=False,
        # _____
        # |I/O|_________________________________________________________________
        save_path='.', output_folder=None, phantom_path='phantom', save_intermediate=False, save_intermediate_level='batch', save_history=False,
        store_checkpoint=True, use_checkpoint=True, force_to_use_checkpoint=False, n_batch_per_checkpoint=10,
        save_stdout=False,
        # _____________
        # |Performance|_________________________________________________________
        cpu_only=False, core_parallelization=True, gpu_index=0,
        n_dp_batch=20,
        distribution_mode=None,  # Choose from None (for data parallelism), 'shared_file', 'distributed_object'
        dist_mode_n_batch_per_update=None,  # If None, object is updated only after all DPs on an angle are processed.
        precalculate_rotation_coords=True,
        cache_dtype='float32',
        rotate_out_of_loop=False,
        # Applies to simple data parallelism mode only. If True, DP will do rotation outside the loss function
        # and the rotated object function is sent for differentiation. May reduce the number
        # of rotation operations if minibatch_size < n_tiles_per_angle, but object can be updated once only after
        # all tiles on an angle are processed. Also this will save the object-sized gradient array in GPU memory
        # or RAM depending on current device setting.
        # _________________________
        # |Other optimizer options|_____________________________________________
        optimize_probe=False, probe_learning_rate=1e-5, optimizer_probe=None,
        probe_update_delay=0, probe_update_limit=None,
        optimize_probe_defocusing=False, probe_defocusing_learning_rate=1e-5, optimizer_probe_defocusing=None,
        optimize_probe_pos_offset=False, probe_pos_offset_learning_rate=1e-2, optimizer_probe_pos_offset=None,
        optimize_prj_pos_offset=False, probe_prj_offset_learning_rate=1e-2, optimizer_prj_pos_offset=None,
        optimize_all_probe_pos=False, all_probe_pos_learning_rate=1e-2, optimizer_all_probe_pos=None,
        optimize_slice_pos=False, slice_pos_learning_rate=1e-4, optimizer_slice_pos=None,
        optimize_free_prop=False, free_prop_learning_rate=1e-2, optimizer_free_prop=None,
        optimize_prj_affine=False, prj_affine_learning_rate=1e-3, optimizer_prj_affine=None,
        optimize_tilt=False, tilt_learning_rate=1e-3, optimizer_tilt=None, initial_tilt=None,
        optimize_ctf_lg_kappa=False, ctf_lg_kappa_learning_rate=1e-3, optimizer_ctf_lg_kappa=None,
        other_params_update_delay=0,
        # _________________________
        # |Alternative algorithms |_____________________________________________
        use_epie=False, epie_alpha=0.8,
        # ________________
        # |Other settings|______________________________________________________
        dynamic_rate=True, pupil_function=None, probe_circ_mask=0.9, dynamic_dropping=False, dropping_threshold=8e-5,
        backend='autograd',  # Choose from 'autograd' or 'pytorch
        debug=False,
        t_max_min=None,
        # At the end of a batch, terminate the program with s tatus 0 if total time exceeds the set value.
        # Useful for working with supercomputers' job dependency system, where the dependent may start only
        # if the parent job exits with status 0.
        **kwargs, ):
    # ______________________________________________________________________

    """
    Notes:
        1. This simulation function uses the predict method of the selected forward model. Make sure to
           check the return of the predict method for the type of data returned (i.e., magnitude or intensity).
    """

    t_zero = time.time()

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()
    global_settings.backend = backend
    device_obj = None if cpu_only else gpu_index
    device_obj = w.get_device(device_obj)
    print(device_obj)
    n_pos = len(probe_pos)

    if rank == 0:
        timestr = str(datetime.datetime.today())
        timestr = timestr[:timestr.find('.')]
        for i in [':', '-', ' ']:
            if i == ' ':
                timestr = timestr.replace(i, '_')
            else:
                timestr = timestr.replace(i, '')
    else:
        timestr = None
    timestr = comm.bcast(timestr, root=0)

    # ================================================================================
    # Set output folder name if not specified.
    # ================================================================================
    if output_folder is None:
        output_folder = 'recon_{}'.format(timestr)
    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

    stdout_options = {'save_stdout': save_stdout, 'output_folder': output_folder,
                      'timestamp': timestr}
    sto_rank = 0 if not debug else rank
    print_flush('Output folder is {}'.format(output_folder), sto_rank, rank, **stdout_options)

    # ================================================================================
    # Create pointer for raw data.
    # ================================================================================

    try:
        f = h5py.File(os.path.join(save_path, fname), 'a', driver='mpio', comm=comm)
    except:
        f = h5py.File(os.path.join(save_path, fname), 'a')
    try:
        prj = f.create_group('exchange').create_dataset('data', shape=[n_theta, n_pos, *probe_size], dtype=np.complex64)
    except:
        prj = f['exchange/data']

    # ================================================================================
    # Get metadata.
    # ================================================================================
    if obj_size[-1] == 1:
        two_d_mode = True
    if n_theta is None:
        n_theta = prj.shape[0]
    if two_d_mode:
        n_theta = 1
    prj_theta_ind = np.arange(n_theta, dtype=int)
    theta_ls = np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    original_shape = [n_theta, *prj.shape[1:]]
    not_first_level = False
    this_obj_size = obj_size
    ds_level = 1
    is_multi_dist = True if free_prop_cm not in [None, 'inf'] and len(free_prop_cm) > 1 else False
    is_sparse_multislice = True if slice_pos_cm_ls is not None else False

    if is_multi_dist:
        subdiv_probe = True
    else:
        subdiv_probe = False

    if subdiv_probe:
        probe_size = obj_size[:2]
        subprobe_size = prj.shape[-2:]
    else:
        probe_size = prj.shape[-2:]
        subprobe_size = probe_size

    if not common_probe_pos:
        n_pos_ls = []
        for i in range(n_theta):
            n_pos_ls.append(len(probe_pos_ls[i]))

    comm.Barrier()

    # ================================================================================
    # Remove kwargs that may cause issue (removing args that were required in
    # previous versions).
    # ================================================================================
    for kw in ['probe_size']:
        if kw in kwargs.keys():
            del kwargs[kw]

    # ================================================================================
    # Set metadata.
    # ================================================================================
    prj_shape = original_shape

    dim_y, dim_x = prj_shape[-2:]
    if minibatch_size is None:
        minibatch_size = n_pos
    comm.Barrier()

    # ================================================================================
    # generate Fresnel kernel.
    # ================================================================================
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    lmbda_nm = 1240. / energy_ev
    delta_nm = voxel_nm[-1]
    h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx,
                   sign_convention=sign_convention)

    # ================================================================================
    # Read or write rotation transformation coordinates.
    # ================================================================================
    if precalculate_rotation_coords:
        if not os.path.exists('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta)):
            comm.Barrier()
            if rank == 0:
                os.makedirs('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta))
            comm.Barrier()
            print_flush('Saving rotation coordinates...', sto_rank, rank, **stdout_options)
            save_rotation_lookup(this_obj_size, theta_ls)
    comm.Barrier()

    # ================================================================================
    # Create object class.
    # ================================================================================
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'), mmap_mode='r')
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'), mmap_mode='r')
    initial_guess = [grid_delta, grid_beta]

    obj = ObjectFunction([*this_obj_size, 2], distribution_mode=distribution_mode,
                         output_folder=output_folder, ds_level=ds_level, object_type=object_type)
    if distribution_mode == 'shared_file':
        obj.create_file_object(use_checkpoint)
        obj.create_temporary_file_object()
        print_flush('Initializing object function in file...', sto_rank, rank, **stdout_options)
        obj.initialize_file_object(save_stdout=save_stdout, timestr=timestr,
                                   not_first_level=not_first_level, initial_guess=initial_guess,
                                   random_guess_means_sigmas=random_guess_means_sigmas,
                                   unknown_type=unknown_type,
                                   dtype=cache_dtype, non_negativity=non_negativity)
    elif distribution_mode == 'distributed_object':
        print_flush('Initializing object array...', sto_rank, rank, **stdout_options)
        obj.initialize_distributed_array(save_stdout=save_stdout, timestr=timestr,
                                         not_first_level=not_first_level, initial_guess=initial_guess,
                                         random_guess_means_sigmas=random_guess_means_sigmas,
                                         unknown_type=unknown_type,
                                         dtype=cache_dtype, non_negativity=non_negativity)
    elif distribution_mode is None:
        print_flush('Initializing object array...', sto_rank, rank, **stdout_options)
        obj.initialize_array(save_stdout=save_stdout, timestr=timestr,
                             not_first_level=not_first_level, initial_guess=initial_guess, device=device_obj,
                             random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type,
                             non_negativity=non_negativity)

    # ================================================================================
    # Create forward model class.
    # ================================================================================

    forwardmodel_args = {'loss_function_type': 'lsq',
                         'distribution_mode': distribution_mode,
                         'device': device_obj,
                         'common_vars_dict': locals(),
                         'raw_data_type': 'intensity',
                         'simulation_mode': True}
    if forward_model == 'auto':
        if is_multi_dist:
            forward_model = MultiDistModel(**forwardmodel_args)
        elif slice_pos_cm_ls is not None:
            forward_model = SparseMultisliceModel(**forwardmodel_args)
        elif common_probe_pos and minibatch_size == 1 and len(probe_pos) == 1 and np.allclose(probe_pos[0], 0):
            forward_model = SingleBatchFullfieldModel(**forwardmodel_args)
        elif common_probe_pos and minibatch_size == 1 and len(probe_pos) > 1 and n_probe_modes == 1:
            forward_model = SingleBatchPtychographyModel(**forwardmodel_args)
        else:
            forward_model = PtychographyModel(**forwardmodel_args)
        print_flush('Auto-selected forward model: {}.'.format(type(forward_model).__name__), sto_rank, rank,
                    **stdout_options)
    else:
        forward_model = forward_model(**forwardmodel_args)
        print_flush('Specified forward model: {}.'.format(type(forward_model).__name__), sto_rank, rank,
                    **stdout_options)

    # ================================================================================
    # Initialize probe functions.
    # ================================================================================
    print_flush('Initialzing probe...', sto_rank, rank, **stdout_options)
    if rank == 0:
        probe_init_kwargs = kwargs
        probe_init_kwargs['lmbda_nm'] = lmbda_nm
        probe_init_kwargs['psize_cm'] = psize_cm
        probe_init_kwargs['normalize_fft'] = normalize_fft
        probe_init_kwargs['n_probe_modes'] = n_probe_modes
        probe_real_init, probe_imag_init = initialize_probe(probe_size, probe_type, pupil_function=pupil_function,
                                                            probe_initial=probe_initial,
                                                            rescale_intensity=False,
                                                            save_path=save_path, fname=fname,
                                                            extra_defocus_cm=probe_extra_defocus_cm,
                                                            raw_data_type=raw_data_type,
                                                            stdout_options=stdout_options,
                                                            sign_convention=sign_convention, **probe_init_kwargs)
        if n_probe_modes == 1:
            probe_real = np.stack([np.squeeze(probe_real_init)])
            probe_imag = np.stack([np.squeeze(probe_imag_init)])
        else:
            if len(probe_real_init.shape) == 3 and len(probe_real_init) == n_probe_modes:
                probe_real = probe_real_init
                probe_imag = probe_imag_init
            elif len(probe_real_init.shape) == 2 or len(probe_real_init) == 1:
                probe_real = []
                probe_imag = []
                probe_real_init = np.squeeze(probe_real_init)
                probe_imag_init = np.squeeze(probe_imag_init)
                i_cum_factor = 0
                for i_mode in range(n_probe_modes):
                    probe_real.append(np.random.normal(probe_real_init, abs(probe_real_init) * 0.2))
                    probe_imag.append(np.random.normal(probe_imag_init, abs(probe_imag_init) * 0.2))
                probe_real = np.stack(probe_real)
                probe_imag = np.stack(probe_imag)
            else:
                raise RuntimeError('Length of supplied supplied probe does not match number of probe modes.')
    else:
        probe_real = None
        probe_imag = None
    probe_real = comm.bcast(probe_real, root=0)
    probe_imag = comm.bcast(probe_imag, root=0)
    probe_real = w.create_variable(probe_real, device=device_obj)
    probe_imag = w.create_variable(probe_imag, device=device_obj)

    # ================================================================================
    # Create variables and optimizers for other parameters (probe, probe defocus,
    # probe positions, etc.).
    # ================================================================================
    opt_args_ls = [0]

    # Common variables to be created regardless if they are optimizable or not.
    if common_probe_pos:
        probe_pos_int = np.round(probe_pos).astype(int)
    else:
        probe_pos_int_ls = [np.round(probe_pos).astype(int) for probe_pos in probe_pos_ls]
    tilt_ls = np.zeros([3, n_theta])
    tilt_ls[0] = theta_ls
    if is_multi_dist:
        n_dists = len(free_prop_cm)
    else:
        if not common_probe_pos:
            n_pos_max = np.max([len(poses) for poses in probe_pos_ls])
            probe_pos_correction = np.zeros([n_theta, n_pos_max, 2])
            for j, (probe_pos, probe_pos_int) in enumerate(zip(probe_pos_ls, probe_pos_int_ls)):
                probe_pos_correction[j, :len(probe_pos)] = probe_pos - probe_pos_int
        n_dists = 1
    prj_affine_ls = np.array([[1., 0, 0], [0, 1., 0]]).reshape([1, 2, 3])
    prj_affine_ls = np.tile(prj_affine_ls, [n_dists, 1, 1])

    # If optimizable parameters are not checkpointed, create them.
    optimizable_params = {}

    optimizable_params['probe_real'] = probe_real
    optimizable_params['probe_imag'] = probe_imag

    optimizable_params['probe_defocus_mm'] = w.create_variable(0.0)
    optimizable_params['probe_pos_offset'] = w.zeros([n_theta, 2], requires_grad=True, device=device_obj)

    if is_multi_dist:
        optimizable_params['probe_pos_correction'] = w.create_variable(np.zeros([n_dists, 2]),
                                                                       requires_grad=optimize_all_probe_pos,
                                                                       device=device_obj)
    else:
        if common_probe_pos:
            optimizable_params['probe_pos_correction'] = w.create_variable(
                np.tile(probe_pos - probe_pos_int, [n_theta, 1, 1]),
                requires_grad=optimize_all_probe_pos, device=device_obj)
        else:
            optimizable_params['probe_pos_correction'] = w.create_variable(probe_pos_correction,
                                                                           requires_grad=optimize_all_probe_pos,
                                                                           device=device_obj)

    if is_sparse_multislice:
        optimizable_params['slice_pos_cm_ls'] = w.create_variable(slice_pos_cm_ls,
                                                                  requires_grad=optimize_slice_pos,
                                                                  device=device_obj)

    if is_multi_dist:
        if optimize_free_prop:
            optimizable_params['free_prop_cm'] = w.create_variable(free_prop_cm,
                                                                   requires_grad=optimize_free_prop,
                                                                   device=device_obj)

    if optimize_tilt:
        optimizable_params['tilt_ls'] = w.create_variable(tilt_ls, device=device_obj, requires_grad=True)

    optimizable_params['prj_affine_ls'] = w.create_variable(prj_affine_ls, device=device_obj,
                                                            requires_grad=optimize_prj_affine)

    if optimize_ctf_lg_kappa:
        optimizable_params['ctf_lg_kappa'] = w.create_variable([ctf_lg_kappa], requires_grad=True,
                                                               device=device_obj, dtype='float64')

    # ================================================================================
    # Start outer (epoch) loop.
    # ================================================================================
    t0 = time.time()

    n_tot_per_batch = minibatch_size * n_ranks

    t00 = time.time()
    print_flush('Allocating jobs over threads...', sto_rank, rank, **stdout_options)
    # Make a list of all thetas and spot positions'
    comm.Barrier()
    if not two_d_mode:
        theta_ind_ls = np.arange(n_theta)
    else:
        temp = abs(theta_ls - theta_st) < 1e-5
        i_theta = np.nonzero(temp)[0][0]
        theta_ind_ls = np.array([i_theta])
    starting_i_theta = 0
    if use_checkpoint:
        try:
            starting_i_theta = np.loadtxt(os.path.join(save_path, 'sim_checkpoint_i_theta.txt'))[0]
            print_flush('Starting from i_theta {}.'.format(starting_i_theta), sto_rank, rank, **stdout_options)
        except:
            pass

    # ================================================================================
    # Put diffraction spots from all angles together, and divide into minibatches.
    # ================================================================================
    for i, i_theta in enumerate(theta_ind_ls[starting_i_theta:]):

        np.savetxt(os.path.join(save_path, 'sim_checkpoint_i_theta.txt'), [i_theta], fmt='%d')

        n_pos = len(probe_pos) if common_probe_pos else n_pos_ls[i_theta]
        spots_ls = range(n_pos)
        # ================================================================================
        # Append randomly selected diffraction spots if necessary, so that a rank won't be given
        # spots from different angles in one batch.
        # When using shared file object, we must also ensure that all ranks deal with data at the
        # same angle at a time.
        # ================================================================================
        if (distribution_mode is None and update_scheme == 'immediate') and n_pos % minibatch_size != 0:
            spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-n_pos % minibatch_size],
                                                            minibatch_size - (n_pos % minibatch_size),
                                                            replace=False))
        elif (distribution_mode is not None or update_scheme == 'per_angle') and n_pos % n_tot_per_batch != 0:
            spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-n_pos % n_tot_per_batch],
                                                            n_tot_per_batch - (n_pos % n_tot_per_batch),
                                                            replace=False))
        # ================================================================================
        # Create task list for the current angle.
        # ind_list_rand is in the format of [((5, 0), (5, 1), ...), ((17, 0), (17, 1), ..., (...))]
        #                                    |___________________|   |_____|
        #                       a batch for all ranks  _|               |_ (i_theta, i_spot)
        #                    (minibatch_size * n_ranks)
        # ================================================================================
        if common_probe_pos:
            # Optimized task distribution for common_probe_pos with lower peak memory.
            if i == 0:
                ind_list_rand = np.zeros([len(theta_ind_ls) * len(spots_ls), 2], dtype='int32')
                temp = np.stack([np.array([i_theta] * len(spots_ls)), spots_ls], axis=1)
                ind_list_rand[:len(spots_ls), :] = temp
            else:
                temp = np.stack([np.array([i_theta] * len(spots_ls)), spots_ls], axis=1)
                ind_list_rand[i * len(spots_ls):(i + 1) * len(spots_ls), :] = temp
        else:
            if i == 0:
                ind_list_rand = np.stack([np.array([i_theta] * len(spots_ls)), spots_ls], axis=1)
            else:
                temp = np.stack([np.array([i_theta] * len(spots_ls)), spots_ls], axis=1)
                ind_list_rand = np.concatenate([ind_list_rand, temp], axis=0)
    ind_list_rand = split_tasks(ind_list_rand, n_tot_per_batch)
    n_batch = len(ind_list_rand)

    print_flush('Allocation done in {} s.'.format(time.time() - t00), sto_rank, rank, **stdout_options)

    # ================================================================================
    # Initialize runtime indices and flags.
    # ================================================================================
    current_i_theta = -1
    initialize_gradients = True
    shared_file_update_flag = False

    for i_batch in range(n_batch):

        # ================================================================================
        # Initialize batch.
        # ================================================================================
        print_flush('Batch {} of {} started.'.format(i_batch, n_batch), sto_rank, rank, **stdout_options)
        starting_batch = 0

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
        probe_pos_int = probe_pos_int if common_probe_pos else probe_pos_int_ls[this_i_theta]
        this_pos_batch = probe_pos_int[this_ind_batch]
        is_last_batch_of_this_theta = i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != this_i_theta
        comm.Barrier()
        print_flush('  Current rank is processing angle ID {}.'.format(this_i_theta), sto_rank, rank,
                    **stdout_options)

        # ================================================================================
        # If moving to a new angle, rotate the HDF5 object and saved
        # the rotated object into the temporary file object.
        # ================================================================================
        if (not (distribution_mode is None and not rotate_out_of_loop)) and \
                (this_i_theta != current_i_theta or shared_file_update_flag):
            current_i_theta = this_i_theta
            print_flush('  Rotating dataset...', sto_rank, rank, **stdout_options)
            t_rot_0 = time.time()
            if precalculate_rotation_coords:
                coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              theta_ls[this_i_theta], reverse=False)
            else:
                coord_ls = theta_ls[this_i_theta]
            if distribution_mode == 'shared_file':
                obj.rotate_data_in_file(coord_ls, interpolation=interpolation, dset_2=obj.dset_rot,
                                        precalculate_rotation_coords=precalculate_rotation_coords)
            elif distribution_mode == 'distributed_object':
                obj.rotate_array(coord_ls, interpolation=interpolation,
                                 precalculate_rotation_coords=precalculate_rotation_coords,
                                 apply_to_arr_rot=False, override_backend='autograd', dtype=cache_dtype,
                                 override_device='cpu')
            elif distribution_mode is None and rotate_out_of_loop:
                obj.rotate_array(coord_ls, interpolation=interpolation,
                                 precalculate_rotation_coords=precalculate_rotation_coords,
                                 apply_to_arr_rot=False, override_device=device_obj)
            # if mask is not None: mask.rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation)
            comm.Barrier()
            print_flush('  Dataset rotation done in {} s.'.format(time.time() - t_rot_0), sto_rank, rank,
                        **stdout_options)

        if distribution_mode:
            # ================================================================================
            # Get values for local chunks of object_delta and beta; interpolate and read directly from HDF5
            # ================================================================================
            t_read_0 = time.time()
            # If probe for each image is a part of the full probe, pad the object with safe_zone_width.
            if distribution_mode == 'shared_file':
                if subdiv_probe:
                    obj_rot = obj.read_chunks_from_file(this_pos_batch - np.array([safe_zone_width] * 2),
                                                        subprobe_size + np.array([safe_zone_width] * 2) * 2,
                                                        dset_2=obj.dset_rot, device=device_obj,
                                                        unknown_type=unknown_type)
                else:
                    obj_rot = obj.read_chunks_from_file(this_pos_batch, probe_size, dset_2=obj.dset_rot,
                                                        device=device_obj, unknown_type=unknown_type)
            elif distribution_mode == 'distributed_object':
                if subdiv_probe:
                    obj_rot = obj.read_chunks_from_distributed_object(
                        probe_pos_int - np.array([safe_zone_width] * 2),
                        this_ind_batch_allranks,
                        minibatch_size, subprobe_size + np.array([safe_zone_width] * 2) * 2,
                        device=device_obj, unknown_type=unknown_type, apply_to_arr_rot=True,
                        dtype=cache_dtype)
                else:
                    obj_rot = obj.read_chunks_from_distributed_object(probe_pos_int, this_ind_batch_allranks,
                                                                      minibatch_size, probe_size,
                                                                      device=device_obj,
                                                                      unknown_type=unknown_type,
                                                                      apply_to_arr_rot=True,
                                                                      dtype=cache_dtype)
            comm.Barrier()
            print_flush('  Chunk reading done in {} s.'.format(time.time() - t_read_0), sto_rank, rank,
                        **stdout_options)
            obj.chunks = obj_rot

        # ================================================================================
        # Calculate object gradients.
        # ================================================================================
        # After gradient is calculated, any modification to optimizable arrays must be
        # inside a no_grad() block!
        # ================================================================================
        t_grad_0 = time.time()
        grad_func_args = {}
        if distribution_mode is None:
            if rotate_out_of_loop:
                obj_arr = obj.arr_rot
            else:
                obj_arr = obj.arr
        else:
            obj_arr = obj.chunks
        for arg in forward_model.argument_ls:
            if arg == 'obj':
                grad_func_args[arg] = obj_arr
            else:
                try:
                    grad_func_args[arg] = optimizable_params[arg]
                except:
                    try:
                        grad_func_args[arg] = locals()[arg]
                    except:
                        grad_func_args[arg] = None
        comm.Barrier()
        print_flush('  Entering simulation loop...', sto_rank, rank, **stdout_options)

        this_pred_batch = forward_model.predict(**grad_func_args)
        complex_output = True if isinstance(this_pred_batch, tuple) else False
        comm.Barrier()
        print_flush('  Batch simulation calculation done in {} s.'.format(time.time() - t_grad_0), sto_rank, rank,
                    **stdout_options)

        # ================================================================================
        # Write data.
        # ================================================================================
        if complex_output:
            prj[this_i_theta, this_ind_batch] = np.stack(w.to_numpy(this_pred_batch[0])) + 1j * w.to_numpy(np.stack(this_pred_batch[1]))
        else:
            prj[this_i_theta, this_ind_batch] = w.to_numpy(this_pred_batch) + 1j * 0
        f.flush()

        # ================================================================================
        # Finishing a batch.
        # ================================================================================
        print_flush(
            'Minibatch/angle done in {} s.'.format(time.time() - t00), sto_rank, rank, **stdout_options)

        gc.collect()
        if not cpu_only:
            print_flush('GPU memory usage (current/peak): {:.2f}/{:.2f} MB; cache space: {:.2f} MB.'.format(
                w.get_gpu_memory_usage_mb(), w.get_peak_gpu_memory_usage_mb(), w.get_gpu_memory_cache_mb()),
                sto_rank, rank, **stdout_options)

        t_elapsed = (time.time() - t_zero) / 60
        t_elapsed = comm.bcast(t_elapsed, root=0)
        if t_max_min is not None and t_elapsed >= t_max_min:
            print_flush('Terminating program because maximum time limit is reached.', sto_rank, rank,
                        **stdout_options)
            sys.exit()


