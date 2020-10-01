import numpy as np
import h5py
import warnings

import time

import adorym
import adorym.wrappers as w
import adorym.global_settings as global_settings
from adorym.constants import *
from adorym.misc import *
from adorym.util import *
from adorym.forward_model import PtychographyModel
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


warnings.warn('Module under construction.')



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
        backend='autograd', # Choose from 'autograd' or 'pytorch
        debug=False,
        t_max_min=None,
        # At the end of a batch, terminate the program with s tatus 0 if total time exceeds the set value.
        # Useful for working with supercomputers' job dependency system, where the dependent may start only
        # if the parent job exits with status 0.
        **kwargs,):

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()
    global_settings.backend = backend
    device_obj = None if cpu_only else gpu_index
    device_obj = w.get_device(device_obj)
    w.set_device(device_obj)

    stdout_options = {}
    sto_rank = 0


    class Subproblem():
        def __init__(self, device):
            self.device = device


    class PhaseRetrievalSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, rho=1, optimizer=None, prev_sp=None, next_sp=None, device=None):
            """
            Phase retrieval subproblem solver.

            :param whole_object_size: 3D shape of the object to be reconstructed.
            :param theta_ls: List of rotation angles in radians.
            :param device: Device object.
            :param rho: Weight of Lagrangian term.
            """
            super(PhaseRetrievalSubproblem, self).__init__(device)
            forwardmodel_args = {'loss_function_type': 'lsq',
                                 'distribution_mode': None,
                                 'device': device,
                                 'common_vars_dict': locals(),
                                 'raw_data_type': 'intensity'}
            self.forward_model = PtychographyModel(*forwardmodel_args)
            self.whole_object_size = whole_object_size
            self.theta_ls = theta_ls
            self.n_theta = len(theta_ls)
            self.rho = rho
            self.optimizer = optimizer
            self.prev_sp = prev_sp
            self.next_sp = next_sp
            assert self.prev_sp is None
            assert isinstance(self.next_sp, AlignmentSubproblem)

        def initialize(self, probe_init, prj, probe_pos=None, n_pos_ls=None, probe_pos_ls=None):
            """
            Initialize solver.

            :param probe_init: List. [probe_real_init, probe_imag_init]. Each is a Float array in
                               [n_modes, det_y, det_x].
            :param prj: H5py Dataset object. Pointer to raw data.
            :param probe_pos: List of probe positions. A list of length-2 tuples. If probe positions are different
                              among angles, put None.
            :param n_pos_ls: None or List of numbers of probe positions for each angle.
            :param probe_pos_ls: List of probe position lists for all angles. If probe positions are common for all
                                 angles, put None.
            """
            self.psi_theta_ls = []
            self.w_theta_ls = []
            self.lambda1_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                psi = initialize_object_for_dp([*self.whole_object_size[0:2], 2])
                psi = np.squeeze(psi)
                psi = w.create_variable(psi, requires_grad=False, device=self.device)
                self.psi_theta_ls.append(psi)
                w_ = initialize_object_for_dp([*self.whole_object_size[0:2], 2])
                w_ = np.squeeze(w_)
                w_ = w.create_variable(w_, requires_grad=False, device=self.device)
                self.w_theta_ls.append(w_)
            self.psi_theta_ls = w.stack(self.psi_theta_ls)
            self.w_theta_ls = w.stack(self.w_theta_ls)
            self.lambda1_theta_ls = w.stack(self.lambda1_theta_ls)
            self.grad_psi = w.zeros_like(self.psi_theta_ls[0], requires_grad=False, device=self.device)
            self.probe_real = w.create_variable(probe_init[0], requires_grad=False, device=self.device)
            self.probe_imag = w.create_variable(probe_init[1], requires_grad=False, device=self.device)
            self.probe_size = probe_init[0].shape[1:]
            self.n_probe_modes = probe_init[0].shape[0]
            self.prj = prj
            self.probe_pos = probe_pos
            self.n_pos_ls = n_pos_ls
            self.probe_pos_ls = probe_pos_ls
            self.optimizer.create_param_arrays(self.psi_theta_ls[0].shape, device=self.device)
            self.optimizer.set_index_in_grad_return(0)

        def correction_shift(self, patch_ls, probe_pos_correction):
            patch_ls_new = []
            for i, patch in enumerate(patch_ls):
                patch_real, patch_imag = realign_image_fourier(patch[:, :, 0], patch[:, :, 1], probe_pos_correction[i],
                                                               axes=(0, 1), device=self.device)
                patch_ls_new.append(w.stack([patch_real, patch_imag], axis=-1))
            return patch_ls_new

        def forward(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch, prj,
                     probe_pos_correction, this_ind_batch):
            # Shift object function (instead of probe as in the 3D case of Adorym, so use negative pos_correction).
            pos_correction_batch = probe_pos_correction[this_i_theta, this_ind_batch]
            patches = self.correction_shift(patches, -pos_correction_batch)
            ex_int = w.zeros([len(patches), *self.probe_size], requires_grad=False, device=self.device)
            for i_mode in range(self.n_probe_modes):
                this_probe_real = probe_real[i_mode, :, :]
                this_probe_imag = probe_imag[i_mode, :, :]
                wave_real, wave_imag = w.complex_mul(patches[:, :, :, 0], patches[:, :, :, 1], this_probe_real, this_probe_imag)
                wave_real, wave_imag = w.fft2_and_shift(wave_real, wave_imag, axes=(1, 2))
                ex_int = ex_int + wave_real ** 2 + wave_imag ** 2
            y_pred_ls = w.sqrt(ex_int)
            return y_pred_ls

        def get_data(self, this_i_theta, this_ind_batch, theta_downsample=None, ds_level=1):
            if theta_downsample is None: theta_downsample = 1
            this_prj_batch = self.prj[this_i_theta * theta_downsample, this_ind_batch]
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
            if ds_level > 1:
                this_prj_batch = this_prj_batch[:, ::ds_level, ::ds_level]
            return this_prj_batch

        def get_part1_loss(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                     probe_pos_correction, this_ind_batch):
            y_pred_ls = self.forward(patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                                     probe_pos_correction, this_ind_batch)
            y_ls = self.get_data(this_i_theta, this_ind_batch)
            loss = w.mean((y_pred_ls - y_ls) ** 2)
            return loss

        def get_part1_grad(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                     probe_pos_correction, this_ind_batch):
            """
            Calculate gradient of patches in a minibatch.

            :return: A stack of 2D gradients.
            """
            y_pred_ls = self.forward(patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                                     probe_pos_correction, this_ind_batch)
            y_ls = self.get_data(this_i_theta, this_ind_batch)
            g = (y_pred_ls - y_ls) * w.sign(y_pred_ls)
            g_real, g_imag = w.ishift_and_ifft2(w.real(g), w.imag(g))
            pos_correction_batch = probe_pos_correction[this_i_theta, this_ind_batch]
            g = self.correction_shift(w.stack([g_real, g_imag], axis=-1), pos_correction_batch)
            g_real, g_imag = w.split_channel(g)
            patches_real, patches_imag = w.split_channel(patches)

            #TODO: implement multimode
            g_psi_real, g_psi_imag = w.complex_mul(g_real, g_imag, probe_real[0], -probe_imag[0])
            g_p_real, g_p_imag = w.complex_mul(g_real, g_imag, patches_real, -patches_imag)
            return (g_psi_real, g_psi_imag), (g_p_real, g_p_imag)

        def get_part2_grad(self, this_i_theta):
            return self.rho * (self.psi_theta_ls[this_i_theta] - self.w_theta_ls[this_i_theta] +
                               self.lambda1_theta_ls[this_i_theta] / self.rho)

        def solve(self, n_iterations=5, minibatch_size=23, randomize_probe_pos=False):
            """Solve subproblem.

            :param n_iterations: Int. Number of inner iterations.
            :param minibatch_size: Int. Number of diffraction patterns per batch.
            """
            self.lambda1_theta_ls = self.next_sp.lambda1_theta_ls
            common_probe_pos = True if self.n_pos_ls is None else False
            if common_probe_pos:
                probe_pos_int = np.round(self.probe_pos).astype(int)
                probe_pos_correction = w.create_variable(np.tile(self.probe_pos - probe_pos_int, [self.n_theta, 1, 1]),
                                                         requires_grad=False, device=self.device)
            else:
                probe_pos_int_ls = [np.round(probe_pos).astype(int) for probe_pos in self.probe_pos_ls]
                n_pos_max = np.max([len(poses) for poses in self.probe_pos_ls])
                probe_pos_correction = np.zeros([self.n_theta, n_pos_max, 2])
                for j, (probe_pos, probe_pos_int) in enumerate(zip(self.probe_pos_ls, probe_pos_int_ls)):
                    probe_pos_correction[j, :len(probe_pos)] = probe_pos - probe_pos_int
                probe_pos_correction = w.create_variable(probe_pos_correction, device=self.device)
            n_tot_per_batch = minibatch_size * n_ranks
            for i_iteration in range(n_iterations):
                theta_ind_ls = np.arange(self.n_theta)
                np.random.shuffle(theta_ind_ls)
                comm.Bcast(theta_ind_ls, root=0)
                # ================================================================================
                # Put diffraction spots from all angles together, and divide into minibatches.
                # ================================================================================
                for i, i_theta in enumerate(theta_ind_ls):
                    n_pos = len(self.probe_pos) if common_probe_pos else self.n_pos_ls[i_theta]
                    spots_ls = range(n_pos)
                    if randomize_probe_pos:
                        spots_ls = np.random.choice(spots_ls, len(spots_ls), replace=False)
                    # ================================================================================
                    # Append randomly selected diffraction spots if necessary, so that a rank won't be given
                    # spots from different angles in one batch.
                    # When using shared file object, we must also ensure that all ranks deal with data at the
                    # same angle at a time.
                    # ================================================================================
                    if n_pos % minibatch_size != 0:
                        spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-n_pos % minibatch_size],
                                                                        minibatch_size - (n_pos % minibatch_size),
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

                current_i_theta = -1
                initialize_gradients = True
                shared_file_update_flag = False

                for i_batch in range(0, n_batch):
                    # ================================================================================
                    # Initialize batch.
                    # ================================================================================
                    print_flush('Iter {}, batch {} of {} started.'.format(i_iteration, i_batch, n_batch), sto_rank, rank,
                                **stdout_options)
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

                    patch_ls = self.get_patches(self.psi_theta_ls[this_i_theta], this_pos_batch)
                    (grad_psi_real_ls, grad_psi_imag_ls), (grad_p_real_ls, grad_p_imag_ls) = \
                        self.get_part1_grad(patch_ls, self.probe_real, self.probe_imag, this_i_theta, this_pos_batch,
                                            probe_pos_correction, this_ind_batch)
                    grad_psi_ls = w.stack([grad_psi_real_ls, grad_psi_imag_ls], axis=-1)
                    self.grad_psi = self.replace_grad_patches(grad_psi_ls, self.grad_psi, this_pos_batch, initialize=True)
                    self.psi_theta_ls[this_i_theta] = self.optimizer.apply_gradient(self.psi_theta_ls[this_i_theta],
                                                                               self.grad_psi,
                                                                               i_batch + n_batch * i_iteration,
                                                                               **self.optimizer.options_dict)
                    # TODO: update probe

                    if is_last_batch_of_this_theta:
                        # Calculate gradient of the second term of the loss upon finishing each angle.
                        self.grad_psi = self.get_part2_grad(this_i_theta)
                        self.psi_theta_ls[this_i_theta] = self.optimizer.apply_gradient(self.psi_theta_ls[this_i_theta],
                                                                                   self.grad_psi,
                                                                                   i_batch + n_batch * i_iteration,
                                                                                   **self.optimizer.options_dict)

        def get_patches(self, psi, this_pos_batch_int):
            """
            Get a list of psi patches.

            :param psi: Tensor. Tensor in shape [psi_y, psi_x, 2].
            :param this_pos_batch_int: Tensor of Int.
            :return: A list of patches.
            """
            patch_ls = []
            psi, pad_arr = pad_object(psi, self.whole_object_size, this_pos_batch_int, self.probe_size)
            for this_pos_int in this_pos_batch_int:
                this_pos_int = this_pos_int + pad_arr[:, 0]
                patch = psi[this_pos_int[0]:this_pos_int[0] + self.probe_size[0],
                            this_pos_int[1]:this_pos_int[1] + self.probe_size[1], :]
                patch_ls.append(patch)
            patch_ls = w.stack(patch_ls)
            return patch_ls

        def replace_grad_patches(self, grad_ls, grad_psi, this_pos_batch_int, initialize=True):
            """
            Add patch gradients into full-psi gradient array.

            :param grad_ls: List.
            :param grad_psi: Tensor.
            :param this_pos_batch_int: Tensor of Int.
            :param initialize: Bool. If True, grad_psi will be set to 0 before adding back gradients.
            :return: Tensor.
            """
            if initialize:
                grad_psi[...] = 0
            init_shape = grad_psi.shape
            grad_psi, pad_arr = pad_object(grad_psi, self.whole_object_size, this_pos_batch_int, self.probe_size)
            for this_grad, this_pos_int in zip(grad_ls, this_pos_batch_int):
                this_pos_int = this_pos_int + pad_arr[:, 0]
                grad_psi[this_pos_int[0]:this_pos_int[0] + self.probe_size[0],
                         this_pos_int[1]:this_pos_int[1] + self.probe_size[1], :] += this_grad
            grad_psi = grad_psi[pad_arr[0, 0]:pad_arr[0, 0] + init_shape[0],
                                pad_arr[1, 0]:pad_arr[1, 0] + init_shape[1]]
            return grad_psi

        def update_dual(self):
            self.lambda1_theta_ls = self.lambda1_theta_ls + self.rho * (self.psi_theta_ls - self.next_sp.w_theta_ls)


    class AlignmentSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, rho=1, optimizer=None, prev_sp=None, next_sp=None, device=None):
            """
            Alignment subproblem solver.

            :param whole_object_size: 3D shape of the object to be reconstructed.
            :param theta_ls: List of rotation angles in radians.
            :param device: Device object.
            :param rho: Weight of Lagrangian term.
            """
            super(AlignmentSubproblem, self).__init__(device)
            self.whole_object_size = whole_object_size
            self.theta_ls = theta_ls
            self.n_theta = len(theta_ls)
            self.rho = rho
            self.optimizer = optimizer
            self.prev_sp = prev_sp
            self.next_sp = next_sp
            assert isinstance(self.prev_sp, PhaseRetrievalSubproblem)
            assert isinstance(self.next_sp, BackpropSubproblem)

        def initialize(self):
            """
            Initialize solver.

            """
            self.w_theta_ls = []
            self.lambda1_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                w_ = initialize_object_for_dp([*self.whole_object_size[0:2], 2])
                w_ = np.squeeze(w_)
                w_ = w.create_variable(w_, requires_grad=False, device=self.device)
                self.w_theta_ls.append(w_)
                lmbda1 = initialize_object_for_dp([*self.whole_object_size[0:2], 2])
                lmbda1 = np.squeeze(lmbda1)
                lmbda1 = w.create_variable(lmbda1, requires_grad=False, device=self.device)
                self.lambda1_theta_ls.append(lmbda1)
            self.w_theta_ls = w.stack(self.w_theta_ls)
            self.lambda1_theta_ls = w.stack(self.lambda1_theta_ls)

        def forward(self, w):
            """
            Operator t.

            :param w: Tensor.
            :return:
            """
            return w

        def solve(self):
            self.psi_theta_ls = self.prev_sp.psi_theta_ls
            self.w_theta_ls = self.psi_theta_ls


    class BackpropSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, binning, energy_ev, psize_cm,
                     rho=1, optimizer=None, prev_sp=None, next_sp=None, device=None):
            """
            Alignment subproblem solver.

            :param whole_object_size: 3D shape of the object to be reconstructed.
            :param theta_ls: List of rotation angles in radians.
            :param device: Device object.
            :param rho: Weight of Lagrangian term.
            """
            super(BackpropSubproblem, self).__init__(device)
            self.whole_object_size = whole_object_size
            self.theta_ls = theta_ls
            self.n_theta = len(theta_ls)
            self.binnning = binning
            self.energy_ev = energy_ev
            self.psize_cm = psize_cm
            self.rho = rho
            self.optimizer = optimizer
            self.prev_sp = prev_sp
            self.next_sp = next_sp
            assert isinstance(prev_sp, AlignmentSubproblem)
            assert isinstance(next_sp, TomographySubproblem)

        def initialize(self):
            """
            Initialize solver.

            :param prev_sp: AlignmentSubproblem object.
            """
            self.u_theta_ls = []
            self.lambda2_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                u = initialize_object_for_dp([*self.whole_object_size[0:3], 2])
                u = np.squeeze(u)
                u = w.create_variable(u, requires_grad=False, device=None) # Keep on RAM, not GPU
                self.u_theta_ls.append(u)
                lmbda2 = initialize_object_for_dp([*self.whole_object_size[0:2], 2])
                lmbda2 = np.squeeze(lmbda2)
                lmbda2 = w.create_variable(lmbda2, requires_grad=False, device=self.device)
                self.lambda2_theta_ls.append(lmbda2)
            self.u_theta_ls = w.stack(self.u_theta_ls)
            self.lambda2_theta_ls = w.stack(self.lambda2_theta_ls)
            self.optimizer.create_param_arrays([*self.whole_object_size, 2], device=self.device)
            self.optimizer.set_index_in_grad_return(0)

        def forward(self, u_ls, energy_ev, psize_cm, return_intermediate_wavefields=False,
                    return_binned_modulators=False):
            """
            Operator g.

            :param u_ls: Tensor. A stack of u_theta chunks in shape [n_chunks, chunk_y, chunk_x, chunk_z, 2].
                         All chunks have to be from the same theta.
            :return: Exiting wavefields.
            """
            patch_shape = u_ls[0].shape[:3]
            probe_real = w.ones(patch_shape[:2], requires_grad=False, device=self.device)
            probe_imag = w.zeros(patch_shape[:2], requires_grad=False, device=self.device)
            res = multislice_propagate_batch(u_ls, probe_real, probe_imag, energy_ev, psize_cm,
                                             psize_cm, free_prop_cm=0, binning=self.binnning,
                                             return_intermediate_wavefields=return_intermediate_wavefields,
                                             return_binned_modulators=return_binned_modulators)
            return res

        def get_grad(self, u_ls, w_ls, x, lambda2_ls, lambda3_ls, theta, energy_ev, psize_cm):
            lmbda_nm = 1240. / energy_ev
            delta_nm = psize_cm * 1e7
            k = 2. * PI * delta_nm / lmbda_nm

            exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls, expu_real_ls, expu_imag_ls = \
                self.forward(u_ls, energy_ev, psize_cm, return_intermediate_wavefields=True,
                             return_binned_modulators=True)
            _, _, psi_backward_real_ls, psi_backward_imag_ls = \
                self.forward(w.stack([-u_ls[:, :, :, ::-1, 0], u_ls[:, :, :, ::-1, 1]], axis=-1),
                             energy_ev, psize_cm, return_intermediate_wavefields=True)
            grad_real, grad_imag = w.complex_mul(w.stack(psi_forward_real_ls),  w.stack(psi_forward_imag_ls),
                                                 w.stack(psi_backward_real_ls[::-1]), w.stack(psi_backward_imag_ls[::-1]))
            grad_real, grad_imag = w.complex_mul(grad_real, grad_imag, expu_real_ls, expu_imag_ls)
            temp = w_ls - w.stack([exit_real, exit_imag], axis=-1) + lambda2_ls / self.rho
            temp_real, temp_imag = w.split_channel(temp)
            grad_real, grad_imag = -self.rho * w.complex_mul(grad_real, grad_imag, temp_real, temp_imag)
            grad_delta = -k * grad_imag
            grad_beta = k * grad_real

            temp = self.next_sp.rho * (u_ls - self.next_sp.forward(x, theta) + lambda3_ls / self.next_sp.rho)
            temp_real, temp_imag = w.split_channel(temp)
            grad_delta = grad_delta - temp_real
            grad_beta = grad_beta - temp_imag

            return grad_delta, grad_beta

        def solve(self, n_iterations=5):
            self.w_theta_ls = self.prev_sp.w_theta_ls
            self.lambda3_theta_ls = self.next_sp.lambda3_theta_ls

            for i_iteration in range(n_iterations):
                theta_ind_ls = range(self.n_theta)
                np.random.shuffle(theta_ind_ls)
                for i, i_theta in enumerate(theta_ind_ls):
                    u_ls = self.u_theta_ls[i_theta]
                    w_ls = self.w_theta_ls[i_theta]
                    x = self.next_sp.x
                    lambda2_ls = self.lambda2_theta_ls[i_theta]
                    lambda3_ls = self.lambda3_theta_ls[i_theta]
                    theta = self.theta_ls[i_theta]
                    grad_u = self.get_grad(u_ls, w_ls, x, lambda2_ls, lambda3_ls, theta, self.energy_ev, self.psize_cm)
                    self.u_theta_ls[i_theta] = self.optimizer.apply_gradient(self.u_theta_ls[i_theta], grad_u,
                                                                             i_iteration,
                                                                             **self.optimizer.options_dict)


    class TomographySubproblem(Subproblem):
        pass
