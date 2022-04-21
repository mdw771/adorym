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
from adorym.regularizers import *
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

def reconstruct_ptychography(
        # ______________________________________
        # |Raw data and experimental parameters|________________________________
        fname, obj_size, probe_pos=None, theta_st=0, theta_end=PI, n_theta=None, theta_downsample=None,
        energy_ev=None, psize_cm=None, free_prop_cm=None,
        raw_data_type='magnitude', # Choose from 'magnitude' or 'intensity'
        is_minus_logged=False, # Select True if raw data (usually conventional tomography) is minus-logged
        slice_pos_cm_ls=None,
        # ___________________________
        # |Reconstruction parameters|___________________________________________
        n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
        # Either pre-declare all regularizers and pass them as a list, or specify values of alpha and gamma
        regularizers=None,
        alpha_d=None, alpha_b=None, gamma=1e-6,
        minibatch_size=None, multiscale_level=1, n_epoch_final_pass=None,
        initial_guess=None,
        random_guess_means_sigmas=(8.7e-7, 5.1e-8, 1e-7, 1e-8),
        # Give as (mean_delta, mean_beta, sigma_delta, sigma_beta) or (mean_mag, mean_phase, sigma_mag, sigma_phase)
        n_batch_per_update=1, reweighted_l1=False, interpolation='bilinear',
        update_scheme='immediate', # Choose from 'immediate' or 'per angle'
        unknown_type='delta_beta', # Choose from 'delta_beta' or 'real_imag'
        randomize_probe_pos=False,
        common_probe_pos=True, # Set to False if the values/number of probe positions vary with projection angle
        fix_object=False, # Do not update the object, just update other parameters
        # __________________________
        # |Object optimizer options|____________________________________________
        optimize_object=True,
        # Keep True in most cases. Setting to False forbids the object from being updated using gradients, which
        # might be desirable when you just want to refine parameters for other reconstruction algorithms.
        optimizer='adam', # Provide adorym.Optimizer type, or choose from 'gd' or 'adam' or 'curveball' or 'momentum' or 'cg'
        learning_rate=1e-5, # Ignored when optimizer is an adorym.Optimizer type
        update_using_external_algorithm=None,
        # Applies to optimizers that use the current batch number for calculation, such as Adam. If 'angle', batch
        # number passed to optimizer increments after each angle. If 'batch', it increases after each batch.
        optimizer_batch_number_increment='angle',
        # ___________________________
        # |Finite support constraint|___________________________________________
        finite_support_mask_path=None, shrink_cycle=None, shrink_threshold=1e-9,
        # ___________________
        # |Object contraints|___________________________________________________
        object_type='normal', # Choose from 'normal', 'phase_only', or 'absorption_only
        non_negativity=False,
        # _______________
        # |Forward model|_______________________________________________________
        forward_model='auto',
        forward_algorithm='fresnel', # Choose from 'fresnel' or 'ctf'
        # ---- CTF parameters ----
        ctf_lg_kappa=1.7, # This is the common log of kappa, i.e. kappa = 10 ** ctf_lg_kappa
        # ------------------------
        binning=1, fresnel_approx=True, pure_projection=False, two_d_mode=False,
        probe_type='gaussian', # Choose from 'gaussian', 'plane', 'ifft', 'aperture_defocus', 'supplied'
        probe_initial=None, # Give as [probe_mag, probe_phase]
        probe_extra_defocus_cm=None,
        n_probe_modes=1,
        shared_probe_among_angles=True,
        rescale_probe_intensity=False,
        loss_function_type='lsq', # Choose from 'lsq' or 'poisson'
        poisson_multiplier = 1.,
        # Intensity scaling factor in Poisson loss function. If intensity data is normalized, this should be the
        # average number of incident photons per pixel.
        beamstop=None,
        normalize_fft=False, # Use False for simulated data generated without normalization. Normalize for Fraunhofer FFT only
        safe_zone_width=0,
        scale_ri_by_k=True,
        sign_convention=1,
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        fourier_disparity=False,
        # _____
        # |I/O|_________________________________________________________________
        save_path='.', output_folder=None, save_intermediate=False, save_intermediate_level='batch', save_history=False,
        store_checkpoint=True, use_checkpoint=True, force_to_use_checkpoint=False, n_batch_per_checkpoint=10,
        save_stdout=False,
        # _____________
        # |Performance|_________________________________________________________
        cpu_only=False, core_parallelization=True, gpu_index=0,
        n_dp_batch=20,
        distribution_mode=None, # Choose from None (for data parallelism), 'shared_file', 'distributed_object'
        dist_mode_n_batch_per_update=None, # If None, object is updated only after all DPs on an angle are processed.
        precalculate_rotation_coords=True,
        cache_dtype='float32',
        rotate_out_of_loop=False,
        n_split_mpi_ata='auto', # Number of segments that the arrays should be split into for MPI AlltoAll
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
        optimize_prj_pos_offset=False, prj_pos_offset_learning_rate=1e-2, optimizer_prj_pos_offset=None,
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
        backend='autograd', # Choose from 'autograd' or 'pytorch
        debug=False,
        t_max_min=None,
        # At the end of a batch, terminate the program with s tatus 0 if total time exceeds the set value.
        # Useful for working with supercomputers' job dependency system, where the dependent may start only
        # if the parent job exits with status 0.
        **kwargs,):
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

    t_zero = time.time()

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()
    global_settings.backend = backend
    device_obj = None if cpu_only else gpu_index
    device_obj = w.get_device(device_obj)
    w.set_device(device_obj)

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
    t0 = time.time()
    print_flush('Reading data...', sto_rank, rank, **stdout_options)
    f = h5py.File(os.path.join(save_path, fname), 'r')
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

    try:
        theta_ls = f['metadata/theta'][...]
        print_flush('Theta list read from HDF5.', sto_rank, rank, **stdout_options)
    except:
        theta_ls = np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    if theta_downsample is not None:
        theta_ls = theta_ls[::theta_downsample]
        prj_theta_ind = prj_theta_ind[::theta_downsample]
        n_theta = len(theta_ls)

    original_shape = [n_theta, *prj.shape[1:]]

    # Probe position.
    if probe_pos is None:
        if common_probe_pos:
            probe_pos = f['metadata/probe_pos_px']
            probe_pos = np.array(probe_pos).astype(float)
        else:
            probe_pos_ls = []
            n_pos_ls = []
            for i in range(n_theta):
                probe_pos_ls.append(f['metadata/probe_pos_px_{}'.format(i)])
                n_pos_ls.append(len(f['metadata/probe_pos_px_{}'.format(i)]))
    else:
        probe_pos = np.array(probe_pos).astype(float)

    # Energy.
    if energy_ev is None:
        energy_ev = float(f['metadata/energy_ev'][...])

    # Pixel size on sample plane.
    if psize_cm is None:
        psize_cm = float(f['metadata/psize_cm'][...])

    # Slice positions (sparse).
    if slice_pos_cm_ls is None or len(slice_pos_cm_ls) == 1:
        is_sparse_multislice = False
    else:
        is_sparse_multislice = True
        u, v = gen_freq_mesh(np.array([psize_cm * 1e7] * 3), prj.shape[2:4])
        u = w.create_variable(u, requires_grad=False, device=device_obj)
        v = w.create_variable(v, requires_grad=False, device=device_obj)

    # Sample to detector distance.
    if free_prop_cm is None:
        free_prop_cm = f['metadata/free_prop_cm'][...]
    if np.array(free_prop_cm).size == 1:
        is_multi_dist = False
        if isinstance(free_prop_cm, np.ndarray):
            try:
                free_prop_cm = free_prop_cm[0]
            except:
                free_prop_cm = float(free_prop_cm)
    else:
        is_multi_dist = True

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

    if is_multi_dist:
        u_free, v_free = gen_freq_mesh(np.array([psize_cm * 1e7] * 3),
                                    [subprobe_size[i] + 2 * safe_zone_width for i in range(2)])
        u_free = w.create_variable(u_free, requires_grad=False, device=device_obj)
        v_free = w.create_variable(v_free, requires_grad=False, device=device_obj)

    print_flush('Data reading: {} s'.format(time.time() - t0), sto_rank, rank, **stdout_options)
    print_flush('Data shape: {}'.format(original_shape), sto_rank, rank, **stdout_options)
    comm.Barrier()

    not_first_level = False

    # ================================================================================
    # Remove kwargs that may cause issue (removing args that were required in
    # previous versions).
    # ================================================================================
    for kw in ['probe_size']:
        if kw in kwargs.keys():
            del kwargs[kw]

    # ================================================================================
    # Batching check.
    # ================================================================================
    if minibatch_size > 1 and common_probe_pos and len(probe_pos) == 1:
        warnings.warn('It seems that you are processing undivided fullfield data with'
                      'minibatch > 1. A rank can only process data from the same rotation'
                      'angle at a time. I am setting minibatch_size to 1.')
        minibatch_size = 1
    if distribution_mode is not None and common_probe_pos and len(probe_pos) == 1:
        warnings.warn('It seems that you are processing undivided fullfield data with'
                      'distribution_mode not None. In shared-file mode and distributed '
                      'object mode, all ranks must'
                      'process data from the same rotation angle in each synchronized'
                      'batch.')

    for ds_level in range(multiscale_level - 1, -1, -1):
        # TODO TOM I think this does reconstructions at different resolutions and increases the resolution over time
        # ================================================================================
        # Set metadata.
        # ================================================================================
        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), sto_rank, rank, **stdout_options)
        comm.Barrier()

        prj_shape = original_shape

        if ds_level > 1:
            this_obj_size = [int(x / ds_level) for x in obj_size]
        else:
            this_obj_size = obj_size

        dim_y, dim_x = prj_shape[-2:]
        if minibatch_size is None:
            minibatch_size = len(probe_pos)
        comm.Barrier()

        # ================================================================================
        # Create output directory.
        # ================================================================================
        if rank == 0:
            try:
                os.makedirs(os.path.join(output_folder))
            except:
                print_flush('Target folder {} exists.'.format(output_folder), sto_rank, rank, **stdout_options)
        comm.Barrier()

        # ================================================================================
        # generate Fresnel kernel.
        # ================================================================================
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        # lmbda_nm = 12.398 / np.sqrt((2*511 + energy_ev)*energy_ev) / 10 # angstrom to nm for electrons
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx, sign_convention=sign_convention)

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
        # Unify random seed for all threads.
        # ================================================================================
        comm.Barrier()
        seed = int(time.time() / 60)
        seed = comm.bcast(seed, root=0)
        np.random.seed(seed)

        # ================================================================================
        # Create object function optimizer.
        # ================================================================================
        if isinstance(optimizer, Optimizer):
            opt = optimizer
            opt.name = 'obj'
        else:
            if optimizer == 'adam':
                optimizer_options_obj = {'step_size': learning_rate}
                opt = AdamOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                    options_dict=optimizer_options_obj)
            elif optimizer == 'gd':
                optimizer_options_obj = {'step_size': learning_rate,
                                         'dynamic_rate': True,
                                         'first_downrate_iteration': 20}
                opt = GDOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                  options_dict=optimizer_options_obj)
            elif optimizer == 'curveball':
                optimizer_options_obj = {}
                opt = CurveballOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                  options_dict=optimizer_options_obj)
            elif optimizer == 'cg':
                optimizer_options_obj = {'step_size': learning_rate}
                opt = CGOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                  options_dict=optimizer_options_obj)
            elif optimizer == 'momentum':
                optimizer_options_obj = {'step_size': learning_rate}
                opt = MomentumOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                  options_dict=optimizer_options_obj)
            elif optimizer == 'scipy':
                if distribution_mode is not None or backend != 'autograd':
                    raise NotImplementedError('ScipyOptimizer supports only data parallelism and Autograd backend.')
                optimizer_options_obj = {'method': 'CG', 'options': {'maxiter': 20}}
                opt = ScipyOptimizer('obj', output_folder=output_folder,
                                     distribution_mode=distribution_mode, options_dict=optimizer_options_obj)
            else:
                raise ValueError('Invalid optimizer type. Must be "gd" or "adam" or "cg" or "scipy".')
        opt.create_container([*this_obj_size, 2], use_checkpoint, device_obj, use_numpy=True)
        opt.set_index_in_grad_return(0)
        opt_ls = [opt]

        # ================================================================================
        # Get checkpointed parameters.
        # ================================================================================
        starting_epoch, starting_batch = (0, 0)
        needs_initialize = False if use_checkpoint else True
        if use_checkpoint:
            try:
                optimizable_params = load_params_checkpoint(os.path.join(output_folder, 'checkpoint', 'params_{}'.format(rank)))
            except:
                optimizable_params = None
            if distribution_mode == 'shared_file':
                try:
                    starting_epoch, starting_batch = restore_checkpoint(output_folder, distribution_mode)
                except:
                    if force_to_use_checkpoint:
                        raise sys.exc_info()
                    needs_initialize = True

            elif distribution_mode != 'shared_file':
                try:
                    starting_epoch, starting_batch, obj_arr = restore_checkpoint(output_folder, distribution_mode, opt, dtype=cache_dtype)
                except:
                    if distribution_mode == 'distributed_object':
                        if rank < obj_size[0] and force_to_use_checkpoint:
                            raise sys.exc_info()
                    obj_arr = None
                    needs_initialize = True
        else:
            optimizable_params = None
        
        needs_initialize = comm.bcast(needs_initialize, root=0)
        starting_epoch = comm.bcast(starting_epoch, root=0)
        starting_batch = comm.bcast(starting_batch, root=0)

        # ================================================================================
        # Create object class.
        # ================================================================================
        obj = ObjectFunction([*this_obj_size, 2], distribution_mode=distribution_mode,
                             output_folder=output_folder, ds_level=ds_level, object_type=object_type)
        if distribution_mode == 'shared_file':
            obj.create_file_object(use_checkpoint)
            obj.create_temporary_file_object()
            if needs_initialize:
                print_flush('Initializing object function in file...', sto_rank, rank, **stdout_options)
                obj.initialize_file_object(save_stdout=save_stdout, timestr=timestr,
                                           not_first_level=not_first_level, initial_guess=initial_guess,
                                           random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type,
                                           dtype=cache_dtype, non_negativity=non_negativity)
        elif distribution_mode == 'distributed_object':
            if needs_initialize:
                print_flush('Initializing object array...', sto_rank, rank, **stdout_options)
                obj.initialize_distributed_array(save_stdout=save_stdout, timestr=timestr,
                                     not_first_level=not_first_level, initial_guess=initial_guess,
                                     random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type,
                                     dtype=cache_dtype, non_negativity=non_negativity)
            else:
                obj.arr = obj_arr

        elif distribution_mode is None:
            if needs_initialize:
                print_flush('Initializing object array...', sto_rank, rank, **stdout_options)
                obj.initialize_array(save_stdout=save_stdout, timestr=timestr,
                                     not_first_level=not_first_level, initial_guess=initial_guess, device=device_obj,
                                     random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type,
                                     non_negativity=non_negativity)
            else:
                obj.arr = w.create_variable(obj_arr, device=device_obj)

        # ================================================================================
        # Create forward model class.
        # ================================================================================
        forwardmodel_args = {'loss_function_type': loss_function_type,
                             'distribution_mode': distribution_mode,
                             'device': device_obj,
                             'common_vars_dict': locals(),
                             'raw_data_type': raw_data_type}
        if forward_model == 'auto':
            if is_multi_dist:
                forward_model = MultiDistModel(**forwardmodel_args)
            elif is_sparse_multislice:
                forward_model = SparseMultisliceModel(**forwardmodel_args)
            elif common_probe_pos and minibatch_size == 1 and len(probe_pos) == 1 and np.allclose(probe_pos[0], 0):
                forward_model = SingleBatchFullfieldModel(**forwardmodel_args)
            elif common_probe_pos and minibatch_size == 1 and len(probe_pos) > 1 and n_probe_modes == 1:
                forward_model = SingleBatchPtychographyModel(**forwardmodel_args)
            else:
                forward_model = PtychographyModel(**forwardmodel_args)
            print_flush('Auto-selected forward model: {}.'.format(type(forward_model).__name__), sto_rank, rank, **stdout_options)
        else:
            forward_model = forward_model(**forwardmodel_args)
            print_flush('Specified forward model: {}.'.format(type(forward_model).__name__), sto_rank, rank, **stdout_options)

        if regularizers is None:
            regularizers = []
            if alpha_d not in [0, None]:
                if reweighted_l1:
                    regularizers.append(ReweightedL1Regularizer(alpha_d, alpha_b, unknown_type=unknown_type))
                else:
                    regularizers.append(L1Regularizer(alpha_d, alpha_b, unknown_type=unknown_type))
            if gamma not in [0, None]:
                regularizers.append(TVRegularizer(gamma, unknown_type=unknown_type))
        forward_model.add_regularizers(regularizers)
        reg_rwl1 = None
        reweighted_l1 = False
        for r in regularizers:
            if isinstance(r, ReweightedL1Regularizer):
                reg_rwl1 = r
                reweighted_l1 = True

        # ================================================================================
        # Create gradient class.
        # ================================================================================
        gradient = Gradient(obj, forward_model=forward_model)
        if distribution_mode == 'shared_file':
            gradient.create_file_object()
            gradient.initialize_gradient_file()
        elif distribution_mode == 'distributed_object':
            gradient.initialize_distributed_array_with_zeros(dtype=cache_dtype)
        else:
            gradient.initialize_array_with_values(np.zeros(this_obj_size), np.zeros(this_obj_size), device=device_obj)

        # ================================================================================
        # If a finite support mask path is specified (common for full-field imaging),
        # create an instance of monochannel mask class. While finite_support_mask_path
        # has to point to a 3D tiff file, the mask will be written  as an HDF5 if
        # share_file_mode is True.
        # ================================================================================
        mask = None
        if finite_support_mask_path is not None:
            mask = Mask(this_obj_size, finite_support_mask_path, distribution_mode=distribution_mode,
                        output_folder=output_folder, ds_level=ds_level)
            if distribution_mode == 'shared_file':
                mask.create_file_object(use_checkpoint=use_checkpoint)
                mask.initialize_file_object(dtype=cache_dtype)
            elif distribution_mode == 'distributed_object':
                mask_arr = dxchange.read_tiff(finite_support_mask_path)
                mask.initialize_distributed_array(mask_arr, dtype=cache_dtype)
            else:
                mask_arr = dxchange.read_tiff(finite_support_mask_path)
                mask.initialize_array_with_values(mask_arr, device=device_obj)

        # ================================================================================
        # Instantize beamstop if provided.
        # ================================================================================
        if beamstop is not None:
            beamstop = w.create_variable(beamstop, device=device_obj, requires_grad=False)

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
            probe_real_init, probe_imag_init = initialize_probe(probe_size, probe_type, pupil_function=pupil_function, probe_initial=probe_initial,
                                                      rescale_intensity=rescale_probe_intensity, save_path=save_path, fname=fname,
                                                      extra_defocus_cm=probe_extra_defocus_cm,
                                                      raw_data_type=raw_data_type, stdout_options=stdout_options,
                                                      sign_convention=sign_convention, **probe_init_kwargs)
            if n_probe_modes == 1:
                if len(probe_real_init.shape) == 3:
                    if len(probe_real_init) > n_probe_modes:
                        probe_real = probe_real_init[:n_probe_modes]
                        probe_imag = probe_imag_init[:n_probe_modes]
                        print_flush('Supplied probe mode number is larger than specified. Only the first {} '
                                    'are used.'.format(n_probe_modes), 0, rank)
                else:
                    probe_real = np.stack([np.squeeze(probe_real_init)])
                    probe_imag = np.stack([np.squeeze(probe_imag_init)])
            else:
                if len(probe_real_init.shape) == 3:
                    probe_real = probe_real_init
                    probe_imag = probe_imag_init
                    if len(probe_real_init) > n_probe_modes:
                        probe_real = probe_real[:n_probe_modes]
                        probe_imag = probe_imag[:n_probe_modes]
                        print_flush('Supplied probe mode number is larger than specified. Only the first {} '
                                    'are used.'.format(n_probe_modes), 0, rank)
                elif len(probe_real_init.shape) == 2 or len(probe_real_init) == 1:
                    probe_real = []
                    probe_imag = []
                    probe_real_init = np.squeeze(probe_real_init)
                    probe_imag_init = np.squeeze(probe_imag_init)
                    i_cum_factor = 0
                    for i_mode in range(n_probe_modes):
                        probe_real.append(np.random.normal(probe_real_init, abs(probe_real_init) * 0.2))
                        probe_imag.append(np.random.normal(probe_imag_init, abs(probe_imag_init) * 0.2))
                        # if i_mode < n_probe_modes - 1:
                        #     probe_real.append(probe_real_init * np.sqrt((1 - i_cum_factor) * 0.85))
                        #     probe_imag.append(probe_imag_init * np.sqrt((1 - i_cum_factor) * 0.85))
                        #     i_cum_factor += (1 - i_cum_factor) * 0.85
                        # else:
                        #     probe_real.append(probe_real_init * np.sqrt((1 - i_cum_factor)))
                        #     probe_imag.append(probe_imag_init * np.sqrt((1 - i_cum_factor)))
                    probe_real = np.stack(probe_real)
                    probe_imag = np.stack(probe_imag)
                else:
                    raise RuntimeError('Length of supplied supplied probe does not match number of probe modes.')
            if not shared_probe_among_angles:
                probe_real = np.tile(probe_real, [n_theta] + [1] * len(probe_real.shape))
                probe_imag = np.tile(probe_imag, [n_theta] + [1] * len(probe_imag.shape))
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
        if initial_tilt is None:
            tilt_ls = np.zeros([3, n_theta])
            tilt_ls[0] = theta_ls
        else:
            tilt_ls = initial_tilt
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
        if optimizable_params is None:
            optimizable_params = {}

            optimizable_params['probe_real'] = probe_real
            optimizable_params['probe_imag'] = probe_imag

            optimizable_params['probe_defocus_mm'] = w.create_variable(0.0)
            optimizable_params['probe_pos_offset'] = w.zeros([n_theta, 2], requires_grad=True, device=device_obj)
            optimizable_params['prj_pos_offset'] = w.zeros([n_theta, 2], requires_grad=True, device=device_obj)

            if is_multi_dist:
                optimizable_params['probe_pos_correction'] = w.create_variable(np.zeros([n_dists, 2]),
                                                             requires_grad=optimize_all_probe_pos, device=device_obj)
            else:
                if common_probe_pos:
                    optimizable_params['probe_pos_correction'] = w.create_variable(np.tile(probe_pos - probe_pos_int, [n_theta, 1, 1]),
                                                             requires_grad=optimize_all_probe_pos, device=device_obj)
                else:
                    optimizable_params['probe_pos_correction'] = w.create_variable(probe_pos_correction, requires_grad=optimize_all_probe_pos, device=device_obj)

            if is_sparse_multislice:
                optimizable_params['slice_pos_cm_ls'] = w.create_variable(slice_pos_cm_ls, requires_grad=optimize_slice_pos, device=device_obj)

            if is_multi_dist:
                if optimize_free_prop:
                    optimizable_params['free_prop_cm'] = w.create_variable(free_prop_cm, requires_grad=optimize_free_prop, device=device_obj)

            if optimize_tilt:
                optimizable_params['tilt_ls'] = w.create_variable(tilt_ls, device=device_obj, requires_grad=True)
            elif initial_tilt is not None:
                tilt_ls = w.create_variable(tilt_ls, device=device_obj, requires_grad=False)

            optimizable_params['prj_affine_ls'] = w.create_variable(prj_affine_ls, device=device_obj, requires_grad=optimize_prj_affine)

            if optimize_ctf_lg_kappa:
                optimizable_params['ctf_lg_kappa'] = w.create_variable([ctf_lg_kappa], requires_grad=True, device=device_obj, dtype='float64')

        opt_ls, opt_args_ls = create_and_initialize_parameter_optimizers(optimizable_params, locals())


        # ================================================================================
        # Use ePIE?
        # This does not work as obj.delta and obj.beta are not implemented
        # ================================================================================
        if use_epie:
            print_flush('WARNING: Reconstructing using ePIE!', sto_rank, rank, **stdout_options)
            warnings.warn('use_epie is True. I will reconstruct using ePIE instead of AD!')
            time.sleep(0.5)
            alt_reconstruction_epie(obj.arr[:, :, :, 0], obj.arr[:,:,:,1], probe_real, probe_imag, probe_pos,
                                    optimizable_params['probe_pos_correction'], prj, device_obj=device_obj,
                                    minibatch_size=minibatch_size, alpha=epie_alpha, n_epochs=n_epochs, energy_ev=energy_ev,
                                    psize_cm=psize_cm, output_folder=output_folder,
                                    raw_data_type=raw_data_type)
            return

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
        print_flush('Optimizer started.', sto_rank, rank, **stdout_options)
        if rank == 0:
            create_summary(output_folder, locals(), preset='ptycho')

        # ================================================================================
        # Start outer (epoch) loop.
        # ================================================================================
        cont = True
        i_epoch = starting_epoch
        while cont:
            t0 = time.time()

            n_tot_per_batch = minibatch_size * n_ranks

            t00 = time.time()
            print_flush('Allocating jobs over threads...', sto_rank, rank, **stdout_options)
            # Make a list of all thetas and spot positions'
            np.random.seed(i_epoch)
            comm.Barrier()
            if not two_d_mode:
                theta_ind_ls = np.arange(n_theta)
                np.random.shuffle(theta_ind_ls)
                comm.Bcast(theta_ind_ls, root=0)
            else:
                temp = abs(theta_ls - theta_ls[0]) < 1e-5
                i_theta = np.nonzero(temp)[0][0]
                theta_ind_ls = np.array([i_theta])

            # ================================================================================
            # Put diffraction spots from all angles together, and divide into minibatches.
            # ================================================================================
            for i, i_theta in enumerate(theta_ind_ls):
                n_pos = len(probe_pos) if common_probe_pos else n_pos_ls[i_theta]
                spots_ls = range(n_pos)
                if randomize_probe_pos:
                    spots_ls = np.random.choice(spots_ls, len(spots_ls), replace=False)
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
                elif (distribution_mode is not None or update_scheme == 'per angle') and n_pos % n_tot_per_batch != 0:
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
            i_opt_batch = starting_epoch * n_batch + starting_batch

            print_flush('Allocation done in {} s.'.format(time.time() - t00), sto_rank, rank, **stdout_options)

            # ================================================================================
            # Initialize runtime indices and flags.
            # ================================================================================
            current_i_theta = -1
            initialize_gradients = True
            shared_file_update_flag = False

            for i_batch in range(starting_batch, n_batch):

                # ================================================================================
                # Time limit check.
                # ================================================================================
                t_elapsed = (time.time() - t_zero) / 60
                t_elapsed = comm.bcast(t_elapsed, root=0)
                if t_max_min is not None and t_elapsed >= t_max_min:
                    print_flush('Terminating program because maximum time limit is reached.', sto_rank, rank, **stdout_options)
                    sys.exit()

                # ================================================================================
                # Initialize batch.
                # ================================================================================
                print_flush('Epoch {}, batch {} of {} started.'.format(i_epoch, i_batch, n_batch), sto_rank, rank, **stdout_options)
                starting_batch = 0

                # ================================================================================
                # Save checkpoint.
                # ================================================================================
                if store_checkpoint and i_batch % n_batch_per_checkpoint == 0:
                    if distribution_mode == 'shared_file':
                        obj.f.flush()
                        obj_arr = None
                    else:
                        if obj.arr is not None:
                            obj_arr = w.to_numpy(obj.arr)
                        else:
                            obj_arr = None
                        cp_path = os.path.join(output_folder, 'checkpoint')
                        create_directory_multirank(cp_path)
                        if (distribution_mode is None and rank == 0) or (distribution_mode is not None):
                            if obj_arr is not None:
                                save_checkpoint(i_epoch, i_batch, output_folder, distribution_mode=distribution_mode,
                                                obj_array=obj_arr, optimizer=opt)
                            save_params_checkpoint(os.path.join(cp_path, 'params_{}'.format(rank)), optimizable_params)
                comm.Barrier()

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
                print_flush('  Current rank is processing angle ID {}.'.format(this_i_theta), sto_rank, rank, **stdout_options)

                # ================================================================================
                # If moving to a new angle, rotate the HDF5 object and saved
                # the rotated object into the temporary file object.
                # ================================================================================
                if (not (distribution_mode is None and not rotate_out_of_loop)) and \
                        (this_i_theta != current_i_theta or shared_file_update_flag):
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
                        if optimizer == 'curveball':
                            opt.rotate_arrays(coord_ls, overwrite_arr=False)
                    elif distribution_mode is None and rotate_out_of_loop:
                        obj.rotate_array(coord_ls, interpolation=interpolation,
                                         precalculate_rotation_coords=precalculate_rotation_coords,
                                         apply_to_arr_rot=False, override_device=device_obj)
                    # if mask is not None: mask.rotate_data_in_file(coord_ls[this_i_theta], interpolation=interpolation)
                    comm.Barrier()
                    print_flush('  Dataset rotation done in {} s.'.format(time.time() - t_rot_0), sto_rank, rank, **stdout_options)


                if this_i_theta != current_i_theta:
                    current_i_theta = this_i_theta


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
                                                                dset_2=obj.dset_rot, device=device_obj, unknown_type=unknown_type)
                        else:
                            obj_rot = obj.read_chunks_from_file(this_pos_batch, probe_size, dset_2=obj.dset_rot, device=device_obj, unknown_type=unknown_type)
                        opt.get_params_from_file(this_pos_batch, probe_size)
                    elif distribution_mode == 'distributed_object':
                        if subdiv_probe:
                            obj_rot = obj.read_chunks_from_distributed_object(probe_pos_int - np.array([safe_zone_width] * 2),
                                                                              this_ind_batch_allranks,
                                                                              minibatch_size, subprobe_size + np.array([safe_zone_width] * 2) * 2,
                                                                              device=device_obj, unknown_type=unknown_type, apply_to_arr_rot=True,
                                                                              dtype=cache_dtype, n_split=n_split_mpi_ata)
                            if isinstance(opt, CurveballOptimizer):
                                opt.z_chunk = opt.read_chunks_from_distributed_object(probe_pos_int - np.array([safe_zone_width] * 2),
                                                                                      this_ind_batch_allranks,
                                                                                      minibatch_size, subprobe_size + np.array([safe_zone_width] * 2) * 2,
                                                                                      device=device_obj, unknown_type=unknown_type, apply_to_arr_rot=True,
                                                                                      dtype=cache_dtype, n_split=n_split_mpi_ata)
                        else:
                            obj_rot = obj.read_chunks_from_distributed_object(probe_pos_int, this_ind_batch_allranks,
                                                                              minibatch_size, probe_size, device=device_obj,
                                                                              unknown_type=unknown_type, apply_to_arr_rot=True,
                                                                              dtype=cache_dtype, n_split=n_split_mpi_ata)
                            if isinstance(opt, CurveballOptimizer):
                                opt.z_chunk = opt.read_chunks_from_distributed_object(probe_pos_int, this_ind_batch_allranks,
                                                                                      minibatch_size, probe_size, device=device_obj,
                                                                                      unknown_type=unknown_type, apply_to_arr_rot=True,
                                                                                      dtype=cache_dtype, n_split=n_split_mpi_ata)
                    comm.Barrier()
                    print_flush('  Chunk reading done in {} s.'.format(time.time() - t_read_0), sto_rank, rank, **stdout_options)
                    obj.chunks = obj_rot

                # ================================================================================
                # Update weight for reweighted l1 if necessary
                # ================================================================================
                if reweighted_l1:
                    with w.no_grad():
                        if distribution_mode:
                            weight_l1 = w.max(obj.chunks) / (w.abs(obj.chunks) + 1e-4 * w.mean(obj.chunks))
                        else:
                            if i_batch % 10 == 0: weight_l1 = w.max(obj.arr) / (w.abs(obj.arr) + 1e-4 * w.mean(obj.arr))
                        reg_rwl1.update_l1_weight(weight_l1)

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
                    elif arg == 'probe_real' and not shared_probe_among_angles:
                        grad_func_args[arg] = probe_real[this_i_theta]
                    elif arg == 'probe_imag' and not shared_probe_among_angles:
                        grad_func_args[arg] = probe_imag[this_i_theta]
                    else:
                        try:
                            grad_func_args[arg] = optimizable_params[arg]
                        except:
                            grad_func_args[arg] = locals()[arg]
                comm.Barrier()
                print_flush('  Entering differentiation loop...', sto_rank, rank, **stdout_options)
                # Update the loss argument dictionary saved in ForwardModel class. Needed for CG but done for all
                # optimizers for now.
                forward_model.update_loss_args(grad_func_args)
                if isinstance(opt, CurveballOptimizer):
                    diff.get_l_h_hessian_and_h_x_jacobian_mvps(forward_model, 0, **grad_func_args)
                    grads = [opt.calculate_dz(diff, use_numpy=True)]
                    opt.calculate_beta_rho(diff, use_numpy=True)
                else:
                    grads = diff.get_gradients(**grad_func_args)
                comm.Barrier()
                print_flush('  Gradient calculation done in {} s.'.format(time.time() - t_grad_0), sto_rank, rank, **stdout_options)
                grads = list(grads)

                # ================================================================================
                # Save gradients to buffer, or write them to file.
                # ================================================================================
                if distribution_mode == 'shared_file':
                    obj_grads = grads[0]
                    t_grad_write_0 = time.time()
                    gradient.write_chunks_to_file(this_pos_batch, *w.split_channel(obj_grads), probe_size,
                                                  write_difference=False, dtype=cache_dtype)
                    print_flush('  Gradient writing done in {} s.'.format(time.time() - t_grad_write_0), 0, rank,
                                **stdout_options)
                elif distribution_mode == 'distributed_object':
                    obj_grads = w.to_numpy(grads[0])
                    t_grad_write_0 = time.time()
                    gradient.sync_chunks_to_distributed_object(obj_grads, probe_pos_int, this_ind_batch_allranks,
                                                               minibatch_size, probe_size, dtype=cache_dtype, n_split=n_split_mpi_ata)
                    comm.Barrier()
                    print_flush('  Gradient syncing done in {} s.'.format(time.time() - t_grad_write_0), 0, rank,
                                **stdout_options)
                else:
                    if initialize_gradients:
                        del gradient.arr
                        gradient.arr = w.zeros(grads[0].shape, requires_grad=False, device=device_obj)
                    gradient.arr += grads[0]
                    # If rotation is not done in the AD loop, the above gradient array is at theta, and needs to be
                    # rotated back to 0.
                    if rotate_out_of_loop:
                        if precalculate_rotation_coords:
                            coord_new = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                                           theta_ls[this_i_theta], reverse=True)
                        else:
                            coord_new = -theta_ls[this_i_theta]
                        # TODO: the rotated gradient should not be accumulated if rotate_out_of_loop
                        gradient.rotate_array(coord_new, interpolation=interpolation,
                                              precalculate_rotation_coords=precalculate_rotation_coords,
                                              override_device=device_obj, overwrite_arr=True)
                if rank == 0 and debug:
                    print_flush('  Average gradient is {} for rank 0.'.format(w.mean(grads[0])), 0, rank,
                                **stdout_options)

                # Initialize gradients for non-object variables if necessary.
                if initialize_gradients:
                    initialize_gradients = False
                    initialize_parameter_gradients(opt_ls, device_obj)

                opt_ls = update_parameter_gradients(opt_ls, grads)

                # if ((update_scheme == 'per angle' or distribution_mode) and not is_last_batch_of_this_theta):
                #     continue
                # else:
                #     initialize_gradients = True

                if distribution_mode is None:
                    if update_scheme == 'per angle' and not is_last_batch_of_this_theta:
                        continue
                    else:
                        initialize_gradients = True
                else:
                    shared_file_update_flag = False
                    if dist_mode_n_batch_per_update is None and not is_last_batch_of_this_theta:
                        continue
                    elif dist_mode_n_batch_per_update is not None and i_batch > 0 and i_batch % dist_mode_n_batch_per_update != 0:
                        continue
                    else:
                        initialize_gradients = True
                        shared_file_update_flag = True

                # ================================================================================
                # All reduce object gradient buffer.
                # ================================================================================
                if distribution_mode is None:
                    gradient.arr = comm.allreduce(gradient.arr)

                # ================================================================================
                # Update object function with optimizer if not distribution_mode; otherwise,
                # just save the gradient chunk into the gradient file.
                # ================================================================================
                with w.no_grad():
                    if distribution_mode is None and optimize_object:
                        if isinstance(opt, ScipyOptimizer):
                            obj.arr = opt.apply_gradient(obj.arr, forward_model=forward_model, differentiator=diff, **opt.options_dict)
                        else:
                            obj.arr = opt.apply_gradient(obj.arr, gradient, i_opt_batch, **opt.options_dict)
                        if isinstance(opt, CurveballOptimizer) and i_batch % 10 == 0:
                             opt.update_lambda(forward_model, grad_func_args)
                if distribution_mode is None:
                    w.reattach(obj.arr)

                # ================================================================================
                # Nonnegativity and phase/absorption-only constraints for non-shared-file-mode,
                # and update arrays in instance.
                # ================================================================================
                with w.no_grad():
                    malias = np if distribution_mode == 'distributed_object' else w
                    if distribution_mode is not 'shared_file' and obj.arr is not None:
                        if non_negativity and unknown_type != 'real_imag':
                            obj.arr = malias.clip(obj.arr, 0, None)
                        if unknown_type == 'delta_beta':
                            if object_type == 'absorption_only': obj.arr[:, :, :, 0] *= 0
                            if object_type == 'phase_only': obj.arr[:, :, :, 1] *= 0
                        elif unknown_type == 'real_imag':
                            if object_type == 'absorption_only':
                                delta, beta = malias.split_channel(obj.arr)
                                delta = malias.norm(delta, beta)
                                beta = beta * 0
                                obj.arr = malias.stack([delta, beta], -1)
                            if object_type == 'phase_only':
                                delta, beta = malias.split_channel(obj.arr)
                                obj_norm = malias.norm(delta, beta)
                                delta = delta / obj_norm
                                beta = beta / obj_norm
                                obj.arr = malias.stack([delta, beta], -1)
                    if update_using_external_algorithm is not None:
                        obj.update_using_external_algorithm(update_using_external_algorithm, locals(), device_obj)
                if distribution_mode is None:
                    w.reattach(obj.arr)

                # ================================================================================
                # Optimize probe and other parameters if necessary.
                # ================================================================================
                optimizable_params = update_parameters(opt_ls, optimizable_params, locals())

                # ================================================================================
                # For shared-file-mode, if finishing or above to move to a different angle,
                # rotate the gradient back, and use it to update the object at 0 deg. Then
                # update the object using gradient at 0 deg.
                # ================================================================================
                if distribution_mode and shared_file_update_flag:
                    if precalculate_rotation_coords:
                        coord_new = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                                       theta_ls[this_i_theta], reverse=True)
                    else:
                        coord_new = -theta_ls[this_i_theta]
                    print_flush('  Rotating gradient dataset back...', sto_rank, rank, **stdout_options)
                    t_rot_0 = time.time()
                    if distribution_mode == 'shared_file':
                        gradient.rotate_data_in_file(coord_new, interpolation=interpolation,
                                                     precalculate_rotation_coords=precalculate_rotation_coords)
                    elif distribution_mode == 'distributed_object':
                        gradient.rotate_array(coord_new, interpolation=interpolation,
                                              precalculate_rotation_coords=precalculate_rotation_coords,
                                              apply_to_arr_rot=False, overwrite_arr=True, override_backend='autograd',
                                              dtype=cache_dtype, override_device='cpu')
                    comm.Barrier()
                    print_flush('  Gradient rotation done in {} s.'.format(time.time() - t_rot_0), sto_rank, rank, **stdout_options)

                    t_apply_grad_0 = time.time()
                    if distribution_mode == 'shared_file' and optimize_object:
                        opt.apply_gradient_to_file(obj, gradient, i_batch=i_opt_batch, **optimizer_options_obj)
                        gradient.initialize_gradient_file()
                    elif distribution_mode == 'distributed_object' and obj.arr is not None and optimize_object:
                        obj.arr = opt.apply_gradient(obj.arr, gradient, i_opt_batch, use_numpy=True, **optimizer_options_obj)
                        gradient.initialize_distributed_array_with_zeros(dtype=cache_dtype)

                    comm.Barrier()
                    print_flush('  Object update done in {} s.'.format(time.time() - t_apply_grad_0), sto_rank, rank, **stdout_options)

                    comm.Barrier()
                    t0_nonify = time.time()
                    del obj.arr_rot
                    obj.arr_rot = None
                    gc.collect()
                    print_flush('  Invadidating obj.arr and garbage collection done in {}'.format(time.time() - t0_nonify), sto_rank, rank)

                # ================================================================================
                # Apply finite support mask if specified.
                # ================================================================================
                if mask is not None:
                    if distribution_mode is None or distribution_mode == 'distributed_object':
                        obj.apply_finite_support_mask_to_array(mask, unknown_type=unknown_type, device=device_obj)
                    elif distribution_mode == 'shared_file':
                        obj.apply_finite_support_mask_to_file(mask, unknown_type=unknown_type, device=device_obj)
                    print_flush('  Mask applied.', sto_rank, rank, **stdout_options)

                # ================================================================================
                # Update finite support mask if necessary.
                # ================================================================================
                if mask is not None and shrink_cycle is not None:
                    if i_batch % shrink_cycle == 0 and i_batch > 0:
                        if distribution_mode == 'shared_file':
                            mask.update_mask_file(obj, shrink_threshold)
                        else:
                            mask.update_mask_array(obj, shrink_threshold)
                        print_flush('  Mask updated.', sto_rank, rank, **stdout_options)

                # ================================================================================
                # Save intermediate object.
                # ================================================================================
                if save_intermediate and ((save_intermediate_level == 'epoch' and i_batch == n_batch - 1) \
                or save_intermediate_level == 'batch'):
                    create_directory_multirank(os.path.join(output_folder, 'intermediate', 'object'))
                    create_parameter_output_folders(opt_ls, output_folder)
                    if distribution_mode != 'distributed_object' and rank == 0 and is_last_batch_of_this_theta:
                        output_object(obj, distribution_mode, os.path.join(output_folder, 'intermediate', 'object'),
                                      unknown_type, full_output=False, i_epoch=i_epoch, i_batch=i_batch,
                                      save_history=save_history)
                        output_intermediate_parameters(opt_ls, optimizable_params, locals())
                    elif distribution_mode == 'distributed_object' and is_last_batch_of_this_theta:
                        output_object(obj, distribution_mode, os.path.join(output_folder, 'intermediate', 'object'),
                                      unknown_type, full_output=False, i_epoch=i_epoch, i_batch=i_batch,
                                      save_history=save_history)
                        if rank == 0:
                            output_intermediate_parameters(opt_ls, optimizable_params, locals())
                comm.Barrier()

                # ================================================================================
                # Finishing a batch.
                # ================================================================================
                current_loss = forward_model.current_loss
                print_flush('Minibatch/angle done in {} s; loss (rank 0) is {}.'.format(time.time() - t00, current_loss), sto_rank, rank, **stdout_options)

                gc.collect()
                if not cpu_only:
                    print_flush('GPU memory usage (current/peak): {:.2f}/{:.2f} MB; cache space: {:.2f} MB.'.format(
                        w.get_gpu_memory_usage_mb(), w.get_peak_gpu_memory_usage_mb(), w.get_gpu_memory_cache_mb()), sto_rank, rank, **stdout_options)
                f_conv.write('{},{},{},{}\n'.format(i_epoch, i_batch, current_loss, time.time() - t_zero))
                f_conv.flush()

                # ================================================================================
                # Update object optimizer's count.
                # ================================================================================
                if optimizer_batch_number_increment == 'angle':
                    if i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != current_i_theta:
                        i_opt_batch += 1
                elif optimizer_batch_number_increment == 'batch':
                    i_opt_batch += 1

            # ================================================================================
            # Stopping criterion.
            # ================================================================================
            if n_epochs == 'auto':
                    pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    time.time() - t0, time.time() - t_zero),
                sto_rank, rank, **stdout_options)
            i_epoch = i_epoch + 1

            # ================================================================================
            # Save reconstruction after an epoch.
            # ================================================================================
            if rank == 0:
                output_object(obj, distribution_mode, output_folder, unknown_type,
                              full_output=True, ds_level=ds_level)
                output_probe(optimizable_params['probe_real'], optimizable_params['probe_imag'], output_folder,
                             full_output=True, ds_level=ds_level)
            print_flush('Current iteration finished.', sto_rank, rank, **stdout_options)
        comm.Barrier()
