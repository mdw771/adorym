import numpy as np
import h5py

import warnings
import time
import inspect

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
        rho_aln=1, rho_bkp=1, rho_tmo=1,
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
        optimizer_phr=None,
        optimizer_phr_probe=None,
        optimizer_aln=None,
        optimizer_bkp=None,
        optimizer_tmo=None,
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

    class Subproblem():
        def __init__(self, device):
            self.device = device
            self.total_iter = 0
            self.forward_current = None

        def set_dependencies(self, prev_sp=None, next_sp=None):
            self.prev_sp = prev_sp
            self.next_sp = next_sp
            if isinstance(self, PhaseRetrievalSubproblem):
                assert self.prev_sp is None
                assert isinstance(self.next_sp, AlignmentSubproblem)
            elif isinstance(self, AlignmentSubproblem):
                assert isinstance(self.prev_sp, PhaseRetrievalSubproblem)
                assert isinstance(self.next_sp, BackpropSubproblem)
            elif isinstance(self, BackpropSubproblem):
                assert isinstance(self.prev_sp, AlignmentSubproblem)
                assert isinstance(self.next_sp, TomographySubproblem)
            elif isinstance(self, TomographySubproblem):
                assert isinstance(self.prev_sp, BackpropSubproblem)
                assert self.next_sp is None


    class PhaseRetrievalSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, rho=1., theta_downsample=None, optimizer=None, device=None,
                     minibatch_size=23, probe_update_delay=30, optimize_probe=False, probe_optimizer=None,
                     common_probe=True):
            """
            Phase retrieval subproblem solver.

            :param whole_object_size: 3D shape of the object to be reconstructed.
            :param theta_ls: List of rotation angles in radians.
            :param device: Device object.
            :param rho: Weight of Lagrangian term.
            :param optimizer: adorym.Optimizer object for object function.
            :param probe_optimizer: adorym.Optimizer object for probe functions.
            :param probe_update_delay: Int. Number of subproblem iterations after which probes can be updated.
            :param common_probe: Whether to use the same exiting plane probe for all positions. Due to object-probe
                                 coupling, allowing different probes for different positions is more physically
                                 accurate for strongly scattering objects, but requires more memory.
            """
            super(PhaseRetrievalSubproblem, self).__init__(device)
            self.whole_object_size = whole_object_size
            self.theta_ls = theta_ls
            self.n_theta = len(theta_ls)
            self.theta_downsample = theta_downsample
            self.rho = rho
            self.optimizer = optimizer
            self.optimize_probe = optimize_probe
            self.probe_optimizer = probe_optimizer
            self.common_probe = common_probe
            self.minibatch_size = minibatch_size
            self.probe_update_delay = probe_update_delay

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
            self.probe_pos = probe_pos
            self.n_pos_ls = n_pos_ls
            self.probe_pos_ls = probe_pos_ls
            if probe_pos is not None:
                self.n_pos_ls = [len(probe_pos)] * self.n_theta
            self.psi_theta_ls = []
            if not self.common_probe:
                self.probe_real_ls = [[None] * self.n_pos_ls[i] for i in range(self.n_theta)]
                self.probe_imag_ls = [[None] * self.n_pos_ls[i] for i in range(self.n_theta)]
            for i, theta in enumerate(self.theta_ls):
                psi_real, psi_imag = initialize_object_for_dp([*self.whole_object_size[0:2], 1],
                                                              random_guess_means_sigmas=(1, 0, 0, 0),
                                                              verbose=False)
                psi_real = psi_real[:, :, 0]
                psi_imag = psi_imag[:, :, 0]
                psi = w.create_variable(np.stack([psi_real, psi_imag], axis=-1), requires_grad=False, device=self.device)
                self.psi_theta_ls.append(psi)
            self.psi_theta_ls = w.stack(self.psi_theta_ls)
            self.probe_real = w.create_variable(probe_init[0], requires_grad=False, device=self.device)
            self.probe_imag = w.create_variable(probe_init[1], requires_grad=False, device=self.device)
            self.probe_size = probe_init[0].shape[1:]
            self.n_probe_modes = probe_init[0].shape[0]
            self.prj = prj
            self.optimizer.create_param_arrays(self.psi_theta_ls.shape, device=self.device)
            self.optimizer.set_index_in_grad_return(0)
            if self.optimize_probe:
                if self.common_probe:
                    self.probe_optimizer.create_param_arrays([*self.probe_real.shape, 2], device=self.device)
                else:
                    self.probe_optimizer.create_param_arrays([n_theta, max(self.n_pos_ls), *self.probe_real.shape, 2],
                                                             device=self.device)
                self.probe_optimizer.set_index_in_grad_return(0)

        def correction_shift(self, patch_ls, probe_pos_correction):
            patch_ls_new = []
            for i, patch in enumerate(patch_ls):
                patch_real, patch_imag = realign_image_fourier(patch[:, :, 0], patch[:, :, 1], probe_pos_correction[i],
                                                               axes=(0, 1), device=self.device)
                patch_ls_new.append(w.stack([patch_real, patch_imag], axis=-1))
            patch_ls_new = w.stack(patch_ls_new)
            return patch_ls_new

        def forward(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                     probe_pos_correction, this_ind_batch):
            """
            Calculate diffraction pattern of patches in a minibatch.

            :param: patches: A stack of psi patches in [n_batch, y, x, 2].
            :param: probe_real: Real part of probe. If common_probe == True, this should be in shape [n_modes, y, x].
                                Otherwise, it should be in [n_batch, n_modes, y, x].
            :return: A list of diffraction magnitudes, a stack of real/imaginary far-field wavefield in
                     [n_batch, n_modes, y, x].
            """
            # Shift object function (instead of probe as in the 3D case of Adorym, so use negative pos_correction).
            pos_correction_batch = probe_pos_correction[this_i_theta, this_ind_batch]
            patches = self.correction_shift(patches, -pos_correction_batch)
            ex_int = w.zeros([len(patches), *self.probe_size], requires_grad=False, device=self.device)
            det_real_mode_ls = w.zeros([len(patches), self.n_probe_modes, *self.probe_size],
                                       requires_grad=False, device=self.device)
            det_imag_mode_ls = w.zeros([len(patches), self.n_probe_modes, *self.probe_size],
                                       requires_grad=False, device=self.device)
            for i_mode in range(self.n_probe_modes):
                slicer = [slice(None)] * (len(probe_real.shape) - 3) + [i_mode, slice(None), slice(None)]
                this_probe_mode_real = probe_real[slicer]
                this_probe_mode_imag = probe_imag[slicer]
                wave_real, wave_imag = w.complex_mul(patches[:, :, :, 0], patches[:, :, :, 1], this_probe_mode_real, this_probe_mode_imag)
                wave_real, wave_imag = w.fft2_and_shift(wave_real, wave_imag, axes=(1, 2))
                det_real_mode_ls[:, i_mode, :, :] = wave_real
                det_imag_mode_ls[:, i_mode, :, :] = wave_imag
                ex_int = ex_int + wave_real ** 2 + wave_imag ** 2
            y_pred_ls = w.sqrt(ex_int)
            return y_pred_ls, det_real_mode_ls, det_imag_mode_ls

        def get_data(self, this_i_theta, this_ind_batch, theta_downsample=None, ds_level=1):
            if theta_downsample is None: theta_downsample = 1
            this_prj_batch = self.prj[this_i_theta * theta_downsample, this_ind_batch]
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
            if ds_level > 1:
                this_prj_batch = this_prj_batch[:, ::ds_level, ::ds_level]
            return this_prj_batch

        def get_part1_loss(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                     probe_pos_correction, this_ind_batch):
            y_pred_ls, _, _ = self.forward(patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                                           probe_pos_correction, this_ind_batch)
            y_ls = self.get_data(this_i_theta, this_ind_batch, self.theta_downsample)
            loss = w.sum((y_pred_ls - y_ls) ** 2)
            return loss

        def get_part1_grad(self, patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                     probe_pos_correction, this_ind_batch, epsilon=1e-11):
            """
            Calculate gradient of patches in a minibatch.

            :param: patches: A stack of psi patches in [n_batch, y, x, 2].
            :param: probe_real: Real part of probe. If common_probe == True, this should be in shape [n_modes, y, x].
                                Otherwise, it should be in [n_batch, n_modes, y, x].
            :return: A stack of 2D gradients.
            """
            # y_pred_ls is in [n_batch, y, x];
            # y_real/imag_mode_ls is in [n_batch, n_modes, y, x].
            y_pred_ls, y_real_mode_ls, y_imag_mode_ls = \
                self.forward(patches, probe_real, probe_imag, this_i_theta, this_pos_batch,
                             probe_pos_correction, this_ind_batch)
            y_ls = self.get_data(this_i_theta, this_ind_batch, theta_downsample=theta_downsample)
            g = (y_pred_ls - y_ls)
            this_loss = w.sum(g ** 2)

            g_psi_real = w.zeros(patches.shape[:-1], requires_grad=False, device=self.device)
            g_psi_imag = w.zeros(patches.shape[:-1], requires_grad=False, device=self.device)
            # g_p_real/imag[slicer] is in either [n_modes, y, x] or [n_batch, n_modes, y, x];
            g_p_real = w.zeros(probe_real.shape, requires_grad=False, device=self.device)
            g_p_imag = w.zeros(probe_imag.shape, requires_grad=False, device=self.device)
            # patches_real/imag is in [n_batch, y, x].
            patches_real, patches_imag = w.split_channel(patches)

            for i_mode in range(self.n_probe_modes):
                # g_real/imag is in [n_batch, y, x].
                g_real, g_imag = g * y_real_mode_ls[:, i_mode, :, :] / (y_pred_ls + epsilon), \
                                 g * y_imag_mode_ls[:, i_mode, :, :] / (y_pred_ls + epsilon)
                g_real, g_imag = w.ishift_and_ifft2(g_real, g_imag)
                pos_correction_batch = probe_pos_correction[this_i_theta, this_ind_batch]
                g = self.correction_shift(w.stack([g_real, g_imag], axis=-1), pos_correction_batch)

                g_real, g_imag = w.split_channel(g)
                # g_p_m_real/imag is in [n_batch, y, x].
                g_p_m_real, g_p_m_imag = w.complex_mul(g_real, g_imag, patches_real, -patches_imag)
                if len(probe_real) == 4:
                    g_p_real[:, i_mode, :, :] = g_p_m_real
                    g_p_imag[:, i_mode, :, :] = g_p_m_imag
                else:
                    g_p_real[i_mode, :, :] = w.mean(g_p_m_real, axis=0)
                    g_p_imag[i_mode, :, :] = w.mean(g_p_m_imag, axis=0)

                slicer = [slice(None)] * (len(probe_real.shape) - 3) + [i_mode, slice(None), slice(None)]
                # probe_real/imag[slicer] is in either [y, x] or [n_batch, y, x];
                # g_psi_m_real/imag is always in [n_batch, y, x].
                g_psi_m_real, g_psi_m_imag = w.complex_mul(g_real, g_imag, probe_real[slicer], -probe_imag[slicer])
                g_psi_real = g_psi_real + g_psi_m_real
                g_psi_imag = g_psi_imag + g_psi_m_imag

            return (g_psi_real, g_psi_imag), (g_p_real, g_p_imag), this_loss

        def get_part2_loss(self, psi_ls, w_ls, lambda1_ls):
            return self.get_part2_grad(psi_ls, w_ls, lambda1_ls)[1]

        def get_part2_grad(self, psi_ls, w_ls, lambda1_ls):
            grad = self.next_sp.rho * (psi_ls - w_ls + lambda1_ls / self.next_sp.rho)
            l_real, l_imag = w.split_channel(grad)
            this_loss = w.sum(l_real ** 2 + l_imag ** 2) / self.next_sp.rho
            return grad, this_loss

        def solve(self, n_iterations=5, randomize_probe_pos=False):
            """Solve subproblem.

            :param n_iterations: Int. Number of inner iterations.
            """
            self.last_iter_part1_loss = 0
            self.last_iter_part2_loss = 0
            lambda1_theta_ls = self.next_sp.lambda1_theta_ls
            w_theta_ls = self.next_sp.w_theta_ls
            grad_psi = w.zeros_like(self.psi_theta_ls[0], requires_grad=False, device=self.device)
            common_probe_pos = True if self.probe_pos_ls is None else False
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
            n_tot_per_batch = self.minibatch_size * n_ranks
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
                    if n_pos % self.minibatch_size != 0:
                        spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-n_pos % self.minibatch_size],
                                                                        self.minibatch_size - (n_pos % self.minibatch_size),
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
                    print_flush('  PHR: Iter {}, batch {} of {} started.'.format(i_iteration, i_batch, n_batch),
                                sto_rank, rank, same_line=True, **stdout_options)
                    starting_batch = 0

                    # ================================================================================
                    # Get scan position, rotation angle indices, and raw data for current batch.
                    # ================================================================================
                    t00 = time.time()
                    if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                        n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                        ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                    this_ind_batch_allranks = ind_list_rand[i_batch]
                    this_i_theta = this_ind_batch_allranks[rank * self.minibatch_size, 0]
                    this_ind_batch = np.sort(this_ind_batch_allranks[rank * self.minibatch_size:(rank + 1) * self.minibatch_size, 1])
                    probe_pos_int = probe_pos_int if common_probe_pos else probe_pos_int_ls[this_i_theta]
                    this_pos_batch = probe_pos_int[this_ind_batch]
                    is_last_batch_of_this_theta = i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != this_i_theta
                    comm.Barrier()

                    if not self.common_probe:
                        this_probe_real = []
                        this_probe_imag = []
                        for ind in this_ind_batch:
                            if self.probe_real_ls[this_i_theta][ind] is not None:
                                this_probe_real.append(self.probe_real_ls[this_i_theta][ind])
                                this_probe_imag.append(self.probe_imag_ls[this_i_theta][ind])
                            else:
                                this_probe_real.append(self.probe_real)
                                this_probe_imag.append(self.probe_imag)
                        this_probe_real = w.stack(this_probe_real)
                        this_probe_imag = w.stack(this_probe_imag)
                    else:
                        this_probe_real = self.probe_real
                        this_probe_imag = self.probe_imag
                    patch_ls = self.get_patches(self.psi_theta_ls[this_i_theta], this_pos_batch)
                    (grad_psi_patch_real_ls, grad_psi_patch_imag_ls), (grad_p_real, grad_p_imag), this_loss = \
                        self.get_part1_grad(patch_ls, this_probe_real, this_probe_imag, this_i_theta, this_pos_batch,
                                            probe_pos_correction, this_ind_batch)
                    grad_psi_patch_ls = w.stack([grad_psi_patch_real_ls, grad_psi_patch_imag_ls], axis=-1)
                    grad_psi[...] = 0
                    grad_psi = self.replace_grad_patches(grad_psi_patch_ls, grad_psi, this_pos_batch, initialize=True)
                    self.psi_theta_ls[this_i_theta] = self.optimizer.apply_gradient(self.psi_theta_ls[this_i_theta],
                                                                               w.cast(grad_psi, 'float32'),
                                                                               i_batch + n_batch * i_iteration,
                                                                               params_slicer=[this_i_theta],
                                                                               **self.optimizer.options_dict)
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part1_loss = self.last_iter_part1_loss + this_loss

                    if self.optimize_probe and self.total_iter > self.probe_update_delay:
                        p = w.stack([this_probe_real, this_probe_imag], axis=-1)
                        grad_p = w.stack([grad_p_real, grad_p_imag], axis=-1)
                        if self.common_probe:
                            params_slicer = [slice(None)]
                        else:
                            params_slicer = [this_i_theta, this_ind_batch]
                        p = self.probe_optimizer.apply_gradient(p, w.cast(grad_p, 'float32'), i_batch + n_batch * i_iteration,
                                                                params_slicer=params_slicer,
                                                                **self.optimizer.options_dict)
                        if self.common_probe:
                            self.probe_real, self.probe_imag = w.split_channel(p)
                        else:
                            pr, pi = w.split_channel(p)
                            for i, ind in enumerate(this_ind_batch):
                                self.probe_real_ls[this_i_theta][ind] = pr[i]
                                self.probe_imag_ls[this_i_theta][ind] = pi[i]

                    if is_last_batch_of_this_theta:
                        # Calculate gradient of the second term of the loss upon finishing each angle.
                        psi_ls = self.psi_theta_ls[this_i_theta]
                        w_ls = w_theta_ls[this_i_theta]
                        lambda1_ls = lambda1_theta_ls[this_i_theta]
                        grad_psi, this_loss = self.get_part2_grad(psi_ls, w_ls, lambda1_ls)
                        a = self.optimizer.apply_gradient(self.psi_theta_ls[this_i_theta], w.cast(grad_psi, 'float32'),
                                                          i_batch=self.total_iter, **self.optimizer.options_dict,
                                                          params_slicer=[this_i_theta])
                        self.psi_theta_ls[this_i_theta] = a
                        if i_iteration == n_iterations - 1:
                            self.last_iter_part2_loss = self.last_iter_part2_loss + this_loss
                self.total_iter += 1
            # self.last_iter_part1_loss /= n_batch
            # self.last_iter_part2_loss /= n_theta

        def get_patches(self, psi, this_pos_batch_int):
            """
            Get a list of psi patches.

            :param psi: Tensor. Tensor in shape [psi_y, psi_x, 2].
            :param this_pos_batch_int: Tensor of Int.
            :return: A list of patches.
            """
            patch_ls = []
            psi = psi[:, :, None, :]
            psi, pad_arr = pad_object_edge(psi, self.whole_object_size, this_pos_batch_int, self.probe_size)
            psi = psi[:, :, 0, :]
            for this_pos_int in this_pos_batch_int:
                this_pos_int = this_pos_int + pad_arr[:, 0]
                patch = psi[this_pos_int[0]:this_pos_int[0] + self.probe_size[0],
                            this_pos_int[1]:this_pos_int[1] + self.probe_size[1], :]
                patch_ls.append(patch)
            patch_ls = w.stack(patch_ls)
            return patch_ls

        def replace_grad_patches(self, grad_patch_ls, grad_psi, this_pos_batch_int, initialize=True):
            """
            Add patch gradients into full-psi gradient array.

            :param grad_patch_ls: List.
            :param grad_psi: Tensor.
            :param this_pos_batch_int: Tensor of Int.
            :param initialize: Bool. If True, grad_psi will be set to 0 before adding back gradients.
            :return: Tensor.
            """
            if initialize:
                grad_psi[...] = 0
            init_shape = grad_psi.shape
            grad_psi, pad_arr = pad_object(grad_psi, self.whole_object_size, this_pos_batch_int, self.probe_size)
            for this_grad, this_pos_int in zip(grad_patch_ls, this_pos_batch_int):
                this_pos_int = this_pos_int + pad_arr[:, 0]
                grad_psi[this_pos_int[0]:this_pos_int[0] + self.probe_size[0],
                         this_pos_int[1]:this_pos_int[1] + self.probe_size[1], :] += this_grad
            grad_psi = grad_psi[pad_arr[0, 0]:pad_arr[0, 0] + init_shape[0],
                                pad_arr[1, 0]:pad_arr[1, 0] + init_shape[1]]
            return grad_psi


    class AlignmentSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, rho=1., optimizer=None, device=None):
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
            forward_model = adorym.ForwardModel()
            args = inspect.getfullargspec(self.get_loss).args
            args.pop(0)
            forward_model.argument_ls = args
            forward_model.get_loss_function = lambda: self.get_loss
            self.optimizer.forward_model = forward_model

        def initialize(self):
            """
            Initialize solver.

            """
            self.w_theta_ls = []
            self.lambda1_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                w_real, w_imag = initialize_object_for_dp([*self.whole_object_size[0:2], 1],
                                                              random_guess_means_sigmas=(1, 0, 0, 0),
                                                              verbose=False)
                w_real = w_real[:, :, 0]
                w_imag = w_imag[:, :, 0]
                w_ = w.create_variable(np.stack([w_real, w_imag], axis=-1), requires_grad=False,
                                        device=self.device)
                self.w_theta_ls.append(w_)
                lmbda1 = w.zeros([*self.whole_object_size[0:2], 2], requires_grad=False, device=self.device)
                self.lambda1_theta_ls.append(lmbda1)
            self.w_theta_ls = w.stack(self.w_theta_ls)
            self.lambda1_theta_ls = w.stack(self.lambda1_theta_ls)
            self.optimizer.create_param_arrays(self.w_theta_ls.shape, device=self.device)

        def forward(self, w_ls):
            """
            Operator t.

            :param w_ls: Tensor.
            :return:
            """
            return w_ls

        def get_loss(self, w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls):

            this_part1_loss, this_part2_loss = self.get_grad(w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls,
                                                             recalculate_multislice=True,
                                                             return_multislice_results=False)[1]
            this_loss = this_part1_loss + this_part2_loss
            return this_loss

        def get_grad(self, w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls, recalculate_multislice=False,
                     return_multislice_results=False):
            """
            Get gradient.

            :param w: Tensor in [n_batch, y, x, 2].
            :return: Gradient.
            """
            if recalculate_multislice:
                self.exit_real, self.exit_imag, self.psi_forward_real_ls, self.psi_forward_imag_ls = \
                    self.next_sp.forward(u_ls, self.next_sp.energy_ev, self.next_sp.psize_cm,
                                         return_intermediate_wavefields=True)
            g_u = w.stack([self.exit_real, self.exit_imag], axis=-1)

            # t(w) is currently assumed to be identity matrix.
            temp = self.rho * (psi_ls - w_ls + lambda1_ls / self.rho)
            grad = -temp
            lr, li = w.split_channel(temp)
            this_part1_loss = w.sum(lr ** 2 + li ** 2) / self.rho

            temp = self.next_sp.rho * (w_ls - g_u + lambda2_ls / self.next_sp.rho)
            grad = grad + temp
            lr, li = w.split_channel(temp)
            this_part2_loss = w.sum(lr ** 2 + li ** 2) / self.next_sp.rho

            if return_multislice_results:
                return grad, self.exit_real, self.exit_imag, self.psi_forward_real_ls, self.psi_forward_imag_ls, \
                       (this_part1_loss, this_part2_loss)
            else:
                return grad, (this_part1_loss, this_part2_loss)

        def solve(self, n_iterations=3):
            self.last_iter_part1_loss = 0
            self.last_iter_part2_loss = 0
            psi_theta_ls = self.prev_sp.psi_theta_ls
            u_theta_ls = self.next_sp.u_theta_ls
            lambda2_theta_ls = self.next_sp.lambda2_theta_ls
            theta_ind_ls = np.arange(self.n_theta).astype(int)
            self.exit_real_ls = []
            self.exit_imag_ls = []
            self.psi_forward_real_ls_ls = []
            self.psi_forward_imag_ls_ls = []
            for i, i_theta in enumerate(theta_ind_ls):
                for i_iteration in range(n_iterations):
                    print_flush('  ALN: Iter {}, theta {} started.'.format(i_iteration, i),
                                sto_rank, rank, same_line=True, **stdout_options)
                    u_ls = u_theta_ls[i_theta:i_theta + 1]
                    w_ls = self.w_theta_ls[i_theta:i_theta + 1]
                    psi_ls = psi_theta_ls[i_theta:i_theta + 1]
                    lambda1_ls = self.lambda1_theta_ls[i_theta:i_theta + 1]
                    lambda2_ls = lambda2_theta_ls[i_theta:i_theta + 1]
                    loss_func_args = {'w_ls': w_ls, 'u_ls': u_ls, 'psi_ls': psi_ls, 'lambda1_ls': lambda1_ls,
                                      'lambda2_ls': lambda2_ls}
                    self.optimizer.forward_model.update_loss_args(loss_func_args)

                    if i_iteration == 0:
                        grad, er, ei, psir, psii, (this_part1_loss, this_part2_loss) = \
                            self.get_grad(w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls, recalculate_multislice=True,
                                          return_multislice_results=True)
                        # ======DEBUG======
                        # if i_theta == 0 and i_iteration == 0:
                        #     import matplotlib.pyplot as plt
                        #     fig, axes = plt.subplots(1, 2)
                        #     a1 = axes[0].imshow(grad[0, :, :, 0])
                        #     plt.colorbar(a1, ax=axes[0])
                        #     a2 = axes[1].imshow(grad[0, :, :, 1])
                        #     plt.colorbar(a2, ax=axes[1])
                        #     plt.savefig(os.path.join(output_folder, 'intermediate', 'grads', 'align_grad_{}.png'.format(i_epoch)),
                        #                 format='png')
                        #     # plt.show()
                        # =================
                        self.exit_real_ls.append(er)
                        self.exit_imag_ls.append(ei)
                        self.psi_forward_real_ls_ls.append(psir)
                        self.psi_forward_imag_ls_ls.append(psii)
                    else:
                        grad, (this_part1_loss, this_part2_loss) = self.get_grad(w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls, recalculate_multislice=False,
                                             return_multislice_results=False)
                    self.optimizer.forward_model.current_loss = this_part1_loss + this_part2_loss
                    w_ls = self.optimizer.apply_gradient(w_ls, w.cast(grad, 'float32'), i_batch=self.total_iter,
                                                         params_slicer=[slice(i_theta, i_theta + 1)],
                                                         **self.optimizer.options_dict)
                    self.w_theta_ls[i_theta:i_theta + 1] = w_ls
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part1_loss = self.last_iter_part1_loss + this_part1_loss
                        self.last_iter_part2_loss = self.last_iter_part2_loss + this_part2_loss
            self.total_iter += 1
            # self.last_iter_part1_loss /= n_theta
            # self.last_iter_part2_loss /= n_theta

        def update_dual(self):
            r = (self.prev_sp.psi_theta_ls - self.forward(self.w_theta_ls))
            self.lambda1_theta_ls = self.lambda1_theta_ls + self.rho * r
            rr, ri = w.split_channel(r)
            self.rsquare = w.mean(rr ** 2 + ri ** 2)


    class BackpropSubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, binning, energy_ev, psize_cm,
                     rho=1., optimizer=None, device=None):
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
            self.binning = binning
            self.energy_ev = energy_ev
            self.psize_cm = psize_cm
            self.rho = rho
            self.optimizer = optimizer
            forward_model = adorym.ForwardModel()
            args = inspect.getfullargspec(self.get_loss).args
            args.pop(0)
            forward_model.argument_ls = args
            forward_model.get_loss_function = lambda: self.get_loss
            self.optimizer.forward_model = forward_model

        def initialize(self):
            """
            Initialize solver.

            :param prev_sp: AlignmentSubproblem object.
            """
            self.u_theta_ls = []
            self.lambda2_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                u_delta, u_beta = initialize_object_for_dp(self.whole_object_size,
                                                           # random_guess_means_sigmas=[8.7e-7, 5.1e-8, 1e-7, 1e-8],
                                                           random_guess_means_sigmas=[0, 0, 0, 0],
                                                           verbose=False)
                u = w.create_variable(np.stack([u_delta, u_beta], axis=-1),
                                      requires_grad=False, device=None) # Keep on RAM, not GPU
                self.u_theta_ls.append(u)
                lmbda2 = w.zeros([*self.whole_object_size[0:2], 2], requires_grad=False, device=self.device)
                self.lambda2_theta_ls.append(lmbda2)
            self.u_theta_ls = w.stack(self.u_theta_ls)
            self.lambda2_theta_ls = w.stack(self.lambda2_theta_ls)
            self.optimizer.create_param_arrays(self.u_theta_ls.shape, device=self.device)
            self.optimizer.set_index_in_grad_return(0)

        def forward(self, u_ls, energy_ev, psize_cm, return_intermediate_wavefields=False):
            """
            Operator g.

            :param u_ls: Tensor. A stack of u_theta chunks in shape [n_chunks, chunk_y, chunk_x, chunk_z, 2].
                         All chunks have to be from the same theta.
            :return: Exiting wavefields.
            """
            patch_shape = u_ls[0].shape[1:4]
            n_batch = len(u_ls)
            probe_real = w.ones([n_batch, *patch_shape[:2]], requires_grad=False, device=self.device, dtype='float64')
            probe_imag = w.zeros([n_batch, *patch_shape[:2]], requires_grad=False, device=self.device, dtype='float64')
            res = multislice_propagate_batch(u_ls, probe_real, probe_imag, energy_ev, psize_cm,
                                             psize_cm, free_prop_cm=0, binning=self.binning,
                                             return_intermediate_wavefields=return_intermediate_wavefields)
            return res

        def backprop(self, u_ls, probe_real, probe_imag, energy_ev, psize_cm, return_intermediate_wavefields=False):
            """
            Hermitian of operator g.

            :param u_ls: Tensor. A stack of u_theta chunks in shape [n_chunks, chunk_y, chunk_x, chunk_z, 2].
                         All chunks have to be from the same theta.
            :return: Exiting wavefields.
            """
            patch_shape = u_ls[0].shape[1:4]
            res = multislice_backpropagate_batch(u_ls, probe_real, probe_imag, energy_ev, psize_cm,
                                                 psize_cm, free_prop_cm=0, binning=self.binning,
                                                 return_intermediate_wavefields=return_intermediate_wavefields)
            return res

        def get_loss(self, u_ls, w_ls, x, lambda2_ls, lambda3_ls, theta, energy_ev, psize_cm):
            lmbda_nm = 1240. / energy_ev
            delta_nm = psize_cm * 1e7
            k = 2. * PI * delta_nm / lmbda_nm
            n_slices = u_ls.shape[-2]

            exit_real, exit_imag = self.forward(u_ls, energy_ev, psize_cm, return_intermediate_wavefields=False)

            grad = -self.rho * (w_ls - w.stack([exit_real, exit_imag], axis=-1) + lambda2_ls / self.rho)
            grad_real, grad_imag = w.split_channel(grad)
            this_part1_loss = w.sum(grad_real ** 2 + grad_imag ** 2) / self.rho

            temp = self.next_sp.rho * (u_ls - self.next_sp.forward(x, theta) + lambda3_ls / self.next_sp.rho)
            temp_delta, temp_beta = w.split_channel(temp)
            this_part2_loss = w.sum(temp_delta ** 2 + temp_beta ** 2) / self.next_sp.rho

            this_loss = this_part1_loss + this_part2_loss
            return this_loss

        def get_grad(self, u_ls, w_ls, x, lambda2_ls, lambda3_ls, theta, energy_ev, psize_cm,
                     i_theta, get_multislice_results_from_prevsp=False):
            lmbda_nm = 1240. / energy_ev
            delta_nm = psize_cm * 1e7
            k = 2. * PI * delta_nm / lmbda_nm
            n_slices = u_ls.shape[-2]

            # Forward pass, store psi'.
            # If this is the first ieration of this subproblem, try to get forward propagation results
            # from the previous subproblem.

            if get_multislice_results_from_prevsp:
                try:
                    exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls = \
                        (self.prev_sp.exit_real_ls[i_theta], self.prev_sp.exit_imag_ls[i_theta],
                         self.prev_sp.psi_forward_real_ls_ls[i_theta], self.prev_sp.psi_forward_imag_ls_ls[i_theta])
                except:
                    warnings.warn('Cannot get multislice results from previous subproblem.')
                    exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls = \
                        self.forward(u_ls, energy_ev, psize_cm, return_intermediate_wavefields=True)
            else:
                exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls = \
                    self.forward(u_ls, energy_ev, psize_cm, return_intermediate_wavefields=True)
            psi_forward_real_ls = w.stack(psi_forward_real_ls, axis=3)
            psi_forward_imag_ls = w.stack(psi_forward_imag_ls, axis=3)

            # Calculate dL / dg = -rho * [w - g(u) + lambda/rho]
            grad = -self.rho * (w_ls - w.stack([exit_real, exit_imag], axis=-1) + lambda2_ls / self.rho)
            grad_real, grad_imag = w.split_channel(grad)
            this_part1_loss = w.sum(grad_real ** 2 + grad_imag ** 2) / self.rho

            # Back propagate and get psi''.
            _, _, psi_backward_real_ls, psi_backward_imag_ls = \
                self.backprop(u_ls, grad_real, grad_imag, energy_ev, psize_cm, return_intermediate_wavefields=True)
            psi_backward_real_ls = w.stack(psi_backward_real_ls[::-1], axis=3)
            psi_backward_imag_ls = w.stack(psi_backward_imag_ls[::-1], axis=3)

            # Calculate dL / d[exp(iku)] = psi''psi'*.
            grad_real, grad_imag = w.complex_mul(psi_backward_real_ls, psi_backward_imag_ls,
                                                 psi_forward_real_ls, -psi_forward_imag_ls)
            if self.binning > 1:
                grad_real = w.repeat(grad_real, self.binning, axis=3)
                grad_imag = w.repeat(grad_imag, self.binning, axis=3)
                if grad_real.shape[-1] > n_slices:
                    grad_real = grad_real[:, :, :, :n_slices]
                    grad_imag = grad_imag[:, :, :, :n_slices]

            # Calculate first term of dL / d(delta/beta).
            expu_herm_real, expu_herm_imag = w.exp_complex(-k * u_ls[:, :, :, :, 1], k * u_ls[:, :, :, :, 0])
            grad_real, grad_imag = w.complex_mul(grad_real, grad_imag, expu_herm_real, expu_herm_imag)
            grad_delta = -grad_imag * k
            grad_beta = -grad_real * k

            # Calculate second term of dL / d(delta/beta).
            temp = self.next_sp.rho * (u_ls - self.next_sp.forward(x, theta) + lambda3_ls / self.next_sp.rho)
            temp_delta, temp_beta = w.split_channel(temp)
            this_part2_loss = w.sum(temp_delta ** 2 + temp_beta ** 2) / self.next_sp.rho
            grad_delta = grad_delta + temp_delta
            grad_beta = grad_beta + temp_beta

            return w.stack([grad_delta, grad_beta], axis=-1), (this_part1_loss, this_part2_loss)

        def solve(self, n_iterations=3):
            self.last_iter_part1_loss = 0
            self.last_iter_part2_loss = 0
            w_theta_ls = self.prev_sp.w_theta_ls
            lambda3_theta_ls = self.next_sp.lambda3_theta_ls
            x = self.next_sp.x

            for i_iteration in range(n_iterations):
                theta_ind_ls = np.arange(self.n_theta).astype(int)
                np.random.shuffle(theta_ind_ls)
                for i, i_theta in enumerate(theta_ind_ls):
                    print_flush('  BKP: Iter {}, theta {} started.'.format(i_iteration, i),
                                sto_rank, rank, same_line=True, **stdout_options)
                    u_ls = w.reshape(self.u_theta_ls[i_theta], [1, *self.u_theta_ls[i_theta].shape])
                    w_ls = w.reshape(w_theta_ls[i_theta], [1, *w_theta_ls[i_theta].shape])
                    lambda2_ls = w.reshape(self.lambda2_theta_ls[i_theta], [1, *self.lambda2_theta_ls[i_theta].shape])
                    lambda3_ls = w.reshape(lambda3_theta_ls[i_theta], [1, *lambda3_theta_ls[i_theta].shape])
                    theta = self.theta_ls[i_theta]
                    loss_func_args = {'u_ls': u_ls, 'w_ls': w_ls, 'x': x, 'lambda2_ls': lambda2_ls,
                                      'lambda3_ls': lambda3_ls, 'theta': theta, 'energy_ev': self.energy_ev,
                                      'psize_cm': self.psize_cm}
                    self.optimizer.forward_model.update_loss_args(loss_func_args)

                    grad_u, (this_part1_loss, this_part2_loss) = \
                        self.get_grad(u_ls, w_ls, x, lambda2_ls, lambda3_ls, theta, self.energy_ev, self.psize_cm,
                                      i_theta=i_theta, get_multislice_results_from_prevsp=(i_iteration == 0))
                    self.optimizer.forward_model.current_loss = this_part1_loss + this_part2_loss
                    # ======DEBUG======
                    # if i_theta == 0 and i_iteration == 0:
                    #     import matplotlib.pyplot as plt
                    #     fig, axes = plt.subplots(1, 2)
                    #     a1 = axes[0].imshow(grad_u[0, 128, :, :, 0])
                    #     plt.colorbar(a1, ax=axes[0])
                    #     a2 = axes[1].imshow(grad_u[0, 128, :, :, 1])
                    #     plt.colorbar(a2, ax=axes[1])
                    #     plt.savefig(os.path.join(output_folder, 'intermediate', 'grads', 'bp_grad_{}.png'.format(i_epoch)), format='png')
                    #     # plt.show()
                    # =================
                    self.u_theta_ls[i_theta:i_theta + 1] = \
                        self.optimizer.apply_gradient(u_ls,
                                                      w.cast(grad_u, 'float32'),
                                                      i_batch=self.total_iter,
                                                      params_slicer=[slice(i_theta, i_theta + 1)],
                                                      **self.optimizer.options_dict)
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part1_loss = self.last_iter_part1_loss + this_part1_loss
                        self.last_iter_part2_loss = self.last_iter_part2_loss + this_part2_loss
                self.total_iter += 1
            # self.last_iter_part1_loss = self.last_iter_part1_loss / n_theta
            # self.last_iter_part2_loss = self.last_iter_part2_loss / n_theta

        def update_dual(self):
            r = 0
            self.rsquare = 0
            for i, theta in enumerate(self.theta_ls):
                u_ls = w.reshape(self.u_theta_ls[i], [1, *self.u_theta_ls[i].shape])
                gn_real, gn_imag = self.forward(u_ls, self.energy_ev, self.psize_cm,
                                                return_intermediate_wavefields=False)
                gn = w.stack([gn_real, gn_imag], axis=-1)
                this_r = self.prev_sp.w_theta_ls[i] - gn
                self.lambda2_theta_ls[i] = self.lambda2_theta_ls[i] + self.rho * this_r
                rr, ri = w.split_channel(this_r)
                self.rsquare = self.rsquare + w.mean(rr ** 2 + ri ** 2)
            self.rsquare = self.rsquare / n_theta

    class TomographySubproblem(Subproblem):
        def __init__(self, whole_object_size, theta_ls, rho=1., optimizer=None, device=None):
            """
            Tomography subproblem solver.

            :param whole_object_size: 3D shape of the object to be reconstructed.
            :param theta_ls: List of rotation angles in radians.
            :param device: Device object.
            :param rho: Weight of Lagrangian term.
            """
            super(TomographySubproblem, self).__init__(device)
            self.whole_object_size = whole_object_size
            self.theta_ls = theta_ls
            self.n_theta = len(theta_ls)
            self.rho = rho
            self.optimizer = optimizer
            forward_model = adorym.ForwardModel()
            args = inspect.getfullargspec(self.get_loss).args
            args.pop(0)
            forward_model.argument_ls = args
            forward_model.get_loss_function = lambda: self.get_loss
            self.optimizer.forward_model = forward_model

        def initialize(self):
            """
            Initialize solver.

            :param prev_sp: AlignmentSubproblem object.
            """
            obj_delta, obj_beta = initialize_object_for_dp(self.whole_object_size,
                                                           # random_guess_means_sigmas=[8.7e-7, 5.1e-8, 1e-7, 1e-8],
                                                           random_guess_means_sigmas=[0, 0, 0, 0],
                                                           verbose=False)
            self.x = w.create_variable(np.stack([obj_delta, obj_beta], axis=-1), requires_grad=False, device=self.device)
            self.lambda3_theta_ls = []
            for i, theta in enumerate(self.theta_ls):
                lmbda3 = w.zeros([*self.whole_object_size, 2])
                lmbda3 = w.create_variable(lmbda3, requires_grad=False, device=self.device)
                self.lambda3_theta_ls.append(lmbda3)
            self.lambda3_theta_ls = w.stack(self.lambda3_theta_ls)
            self.optimizer.create_param_arrays(self.x.shape, device=self.device)
            self.optimizer.set_index_in_grad_return(0)

        def forward(self, x, theta):
            return w.rotate(x, theta, axis=0, device=self.device)

        def get_loss(self, x, u, lambda3, theta):
            grad = u - self.forward(x, theta) + lambda3 / self.rho
            grad_delta, grad_beta = w.split_channel(grad)
            this_loss = w.sum(grad_delta ** 2 + grad_beta ** 2) * self.rho
            return this_loss

        def get_grad(self, x, u, lambda3, theta):
            grad = u - self.forward(x, theta) + lambda3 / self.rho
            grad_delta, grad_beta = w.split_channel(grad)
            this_loss = w.sum(grad_delta ** 2 + grad_beta ** 2) * self.rho
            grad = self.forward(grad, -theta)
            grad = -self.rho * grad
            return grad, this_loss

        def solve(self, n_iterations=3):
            self.last_iter_part1_loss = 0
            theta_ind_ls = np.arange(self.n_theta).astype(int)
            u_theta_ls = self.prev_sp.u_theta_ls
            for i_iteration in range(n_iterations):
                np.random.shuffle(theta_ind_ls)
                for i, i_theta in enumerate(theta_ind_ls):
                    print_flush('  TMO: Iter {}, theta {} started.'.format(i_iteration, i),
                                sto_rank, rank, same_line=True, **stdout_options)
                    u = u_theta_ls[i_theta]
                    lambda3 = self.lambda3_theta_ls[i_theta]
                    theta = self.theta_ls[i_theta]
                    loss_func_args = {'x': self.x, 'u': u, 'lambda3': lambda3, 'theta': theta}
                    self.optimizer.forward_model.update_loss_args(loss_func_args)

                    grad, this_loss = self.get_grad(self.x, u, lambda3, theta)
                    self.optimizer.forward_model.current_loss = this_loss
                    # ======DEBUG======
                    # if i_theta == 0 and i_iteration == 0:
                    #     import matplotlib.pyplot as plt
                    #     fig, axes = plt.subplots(1, 2)
                    #     a1 = axes[0].imshow(grad[128, :, :, 0])
                    #     plt.colorbar(a1, ax=axes[0])
                    #     a2 = axes[1].imshow(grad[128, :, :, 1])
                    #     plt.colorbar(a2, ax=axes[1])
                    #     plt.savefig(os.path.join(output_folder, 'intermediate', 'grads', 'tomo_grad_{}.png'.format(i_epoch)),
                    #                 format='png')
                    #     # plt.show()
                    # =================
                    self.x = self.optimizer.apply_gradient(self.x, w.cast(grad, 'float32'), i_batch=self.total_iter,
                                                           **self.optimizer.options_dict)
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part1_loss = self.last_iter_part1_loss + this_loss
                self.total_iter += 1
            # self.last_iter_part1_loss = self.last_iter_part1_loss / n_theta

        def update_dual(self):
            self.rsquare = 0
            u_theta_ls = self.prev_sp.u_theta_ls
            for i, theta in enumerate(self.theta_ls):
                u = u_theta_ls[i]
                this_r = u - self.forward(self.x, theta)
                self.lambda3_theta_ls[i] = self.lambda3_theta_ls[i] + self.rho * this_r
                rr, ri = w.split_channel(this_r)
                self.rsquare = self.rsquare + w.mean(rr ** 2 + ri ** 2)
            self.rsquare = self.rsquare / n_theta

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

    for ds_level in range(multiscale_level - 1, -1, -1):
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
        # Unify random seed for all threads.
        # ================================================================================
        comm.Barrier()
        seed = int(time.time() / 60)
        seed = comm.bcast(seed, root=0)
        np.random.seed(seed)

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
                                                                rescale_intensity=rescale_probe_intensity,
                                                                save_path=save_path, fname=fname,
                                                                extra_defocus_cm=probe_extra_defocus_cm,
                                                                raw_data_type=raw_data_type, stdout_options=stdout_options,
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

        # Forward propagate probe to exiting plane.
        voxel_nm = np.array([psize_cm] * 3) * 1.e7
        probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, psize_cm * this_obj_size[2] * 1e7,
                                                   lmbda_nm, voxel_nm, override_backend='autograd')

        # ================================================================================
        # Declare and initialize subproblems.
        # ================================================================================
        sp_phr = PhaseRetrievalSubproblem(obj_size, theta_ls, theta_downsample=theta_downsample,
                                          probe_update_delay=30,
                                          rho=1e-3, optimizer=optimizer_phr, device=device_obj,
                                          common_probe=False, minibatch_size=minibatch_size,
                                          optimize_probe=True, probe_optimizer=optimizer_phr_probe)
        sp_phr.initialize([probe_real, probe_imag], prj, probe_pos=probe_pos)

        sp_aln = AlignmentSubproblem(obj_size, theta_ls, rho=rho_aln, optimizer=optimizer_aln, device=device_obj)
        sp_aln.initialize()

        sp_bkp = BackpropSubproblem(obj_size, theta_ls, binning=binning, energy_ev=energy_ev, psize_cm=psize_cm,
                                   rho=rho_bkp, optimizer=optimizer_bkp, device=device_obj)
        sp_bkp.initialize()

        sp_tmo = TomographySubproblem(obj_size, theta_ls, rho=rho_tmo, optimizer=optimizer_tmo, device=device_obj)
        sp_tmo.initialize()

        sp_phr.set_dependencies(None, sp_aln)
        sp_aln.set_dependencies(sp_phr, sp_bkp)
        sp_bkp.set_dependencies(sp_aln, sp_tmo)
        sp_tmo.set_dependencies(sp_bkp, None)

        # ================================================================================
        # Enter ADMM iterations.
        # ================================================================================
        for i_epoch in range(n_epochs):
            t0 = time.time()
            t00 = time.time()

            # ####### DEBUG #######
            # ff = h5py.File('/home/beams/B282788/Data/programs/adorym_dev/demos/adhesin/data_adhesin_360_soft_4d.h5')
            # dd = ff['exchange/data']
            # psi_ls = []
            # for img in dd[::theta_downsample]:
            #     psi = np.stack([img[0].real, img[0].imag], axis=-1)
            #     psi_ls.append(psi)
            # sp_phr.psi_theta_ls = w.create_variable(np.stack(psi_ls), requires_grad=False)
            ###
            # psi_mag_ls = []
            # psi_phase_ls = []
            # for i in range(50):
            #     psi_mag_ls.append(dxchange.read_tiff(os.path.join(output_folder, 'intermediate', 'ptycho',
            #                                                       'psi_{}_mag_65_0.tiff'.format(i))))
            #     psi_phase_ls.append(dxchange.read_tiff(os.path.join(output_folder, 'intermediate', 'ptycho',
            #                                                       'psi_{}_phase_65_0.tiff'.format(i))))
            # psi_mag_ls = np.stack(psi_mag_ls)
            # psi_phase_ls = np.stack(psi_phase_ls)
            # psi_real_ls, psi_imag_ls = mag_phase_to_real_imag(psi_mag_ls, psi_phase_ls)
            # sp_phr.psi_theta_ls = w.create_variable(np.stack([psi_real_ls, psi_imag_ls], axis=-1), requires_grad=False)

            sp_phr.solve(n_iterations=1)
            print_flush('PHR done in {} s. Loss: {}/{}.'.format(time.time() - t00, sp_phr.last_iter_part1_loss,
                                                                sp_phr.last_iter_part2_loss), sto_rank, rank)
            t00 = time.time()
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(sp_phr.psi_theta_ls[0, :, :, 0])
            # axes[1].imshow(sp_phr.psi_theta_ls[0, :, :, 1])
            # axes[2].imshow(w.arctan2(sp_phr.psi_theta_ls[0, :, :, 1], sp_phr.psi_theta_ls[0, :, :, 0]))
            # plt.show()
            sp_aln.solve(n_iterations=2)
            print_flush('ALN done in {} s. Loss: {}/{}.'.format(time.time() - t00, sp_aln.last_iter_part1_loss,
                                                                sp_aln.last_iter_part2_loss), sto_rank, rank)
            # ff = h5py.File('/home/beams/B282788/Data/programs/adorym_dev/demos/adhesin/data_adhesin_360_soft_4d.h5')
            # dd = ff['exchange/data']
            # w_ls = []
            # for img in dd[::theta_downsample]:
            #     ww = np.stack([img[0].real, img[0].imag], axis=-1)
            #     w_ls.append(ww)
            # sp_aln.w_theta_ls = w.create_variable(np.stack(w_ls), requires_grad=False)
            #######################


            # ####### DEBUG #######
            # grid_delta = np.load('/home/beams/B282788/Data/programs/adorym_dev/demos/adhesin/phantom/grid_delta.npy')
            # grid_beta = np.load('/home/beams/B282788/Data/programs/adorym_dev/demos/adhesin/phantom/grid_beta.npy')
            # x = np.stack([grid_delta, grid_beta], axis=-1)
            # x = w.create_variable(x, requires_grad=False)
            # for i, theta in enumerate(sp_bkp.theta_ls):
            #     sp_bkp.u_theta_ls[i] = w.rotate(x, theta, axis=0)
            #######################


            t00 = time.time()
            sp_bkp.solve(n_iterations=2)
            print_flush('BKP done in {} s. Loss: {}/{}.'.format(time.time() - t00, sp_bkp.last_iter_part1_loss,
                                                                sp_bkp.last_iter_part2_loss), sto_rank, rank)
            t00 = time.time()
            sp_tmo.solve(n_iterations=2)
            print_flush('TMO done in {} s. Loss: {}.'.format(time.time() - t00, sp_tmo.last_iter_part1_loss),
                        sto_rank, rank)

            t00 = time.time()
            sp_aln.update_dual()
            print_flush('ALN dual update done in {} s. R^2 = {}.'.format(time.time() - t00, sp_aln.rsquare), sto_rank, rank)
            t00 = time.time()
            sp_bkp.update_dual()
            print_flush('BKP dual update done in {} s. R^2 = {}.'.format(time.time() - t00, sp_bkp.rsquare), sto_rank, rank)
            t00 = time.time()
            sp_tmo.update_dual()
            print_flush('TMO dual update done in {} s. R^2 = {}.'.format(time.time() - t00, sp_tmo.rsquare), sto_rank, rank)

            # ================================================================================
            # Save reconstruction after an epoch.
            # ================================================================================
            if rank == 0:
                obj = adorym.ObjectFunction(obj_size)
                obj.arr = sp_tmo.x
                output_object(obj, distribution_mode, os.path.join(output_folder, 'intermediate', 'object'),
                              unknown_type, full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)
                output_probe(sp_phr.probe_real, sp_phr.probe_imag, os.path.join(output_folder, 'intermediate', 'probe'),
                             full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)
                for i_theta, theta in enumerate(theta_ls):
                    output_probe(sp_phr.psi_theta_ls[i_theta][:, :, 0], sp_phr.psi_theta_ls[i_theta][:, :, 1],
                                 os.path.join(output_folder, 'intermediate', 'ptycho'), custom_name='psi_{}'.format(i_theta),
                                 full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)

            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                                  time.time() - t0, time.time() - t_zero),
                sto_rank, rank, **stdout_options)