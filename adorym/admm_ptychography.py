import numpy as np
import h5py

import warnings
import time
import inspect
import pickle

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


class Subproblem():
    def __init__(self, device):
        self.device = device
        self.total_iter = 0
        self.forward_current = None
        self.output_folder = None
        self.temp_folder = None

    def set_dependencies(self, prev_sp=None, next_sp=None):
        self.prev_sp = prev_sp
        self.next_sp = next_sp

    def save_variable(self, var, fname, to_numpy=True):
        if to_numpy:
            var = w.to_numpy(var)
        np.save(os.path.join(self.temp_folder, fname), var)

    def load_variable(self, fname, create_variable=True):
        if len(fname) < 4 or fname[-4:] != '.npy':
            fname += '.npy'
        var = np.load(os.path.join(self.temp_folder, fname))
        if create_variable:
            var = w.create_variable(var, requires_grad=False, device=self.device)
        return var

    def load_mmap(self, fname, mode='r'):
        if len(fname) < 4 or fname[-4:] != '.npy':
            fname += '.npy'
        var = np.load(os.path.join(self.temp_folder, fname), allow_pickle=True, mmap_mode=mode)
        return var

    def setup_temp_folder(self, output_folder):
        self.output_folder = output_folder
        self.temp_folder = os.path.join(output_folder, 'tmp')
        if rank == 0:
            if not os.path.exists(self.temp_folder):
                os.makedirs(self.temp_folder)


class PhaseRetrievalSubproblem(Subproblem):
    def __init__(self, whole_object_size, rho=1., theta_downsample=None, optimizer=None, device=None,
                 minibatch_size=23, probe_update_delay=30, optimize_probe=False, probe_optimizer=None,
                 common_probe=True, randomize_probe_pos=False, stdout_options={}):
        """
        Phase retrieval subproblem solver.

        :param whole_object_size: 3D shape of the object to be reconstructed.
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
        self.theta_downsample = theta_downsample
        self.rho = rho
        self.optimizer = optimizer
        self.optimize_probe = optimize_probe
        self.probe_optimizer = probe_optimizer
        self.common_probe = common_probe
        self.minibatch_size = minibatch_size
        self.probe_update_delay = probe_update_delay
        self.stdout_options = stdout_options
        self.randomize_probe_pos = randomize_probe_pos

    def initialize(self, probe_init, prj, theta_ls=None, probe_pos=None, n_pos_ls=None, probe_pos_ls=None,
                   output_folder='.'):
        """
        Initialize solver.

        :param probe_init: List. [probe_real_init, probe_imag_init]. Each is a Float array in
                           [n_modes, det_y, det_x].
        :param theta_ls: List of rotation angles in radians.
        :param prj: H5py Dataset object. Pointer to raw data.
        :param probe_pos: List of probe positions. A list of length-2 tuples. If probe positions are different
                          among angles, put None.
        :param n_pos_ls: None or List of numbers of probe positions for each angle.
        :param probe_pos_ls: List of probe position lists for all angles. If probe positions are common for all
                             angles, put None.
        """
        self.theta_ls = theta_ls
        self.n_theta = len(theta_ls)
        self.setup_temp_folder(output_folder); comm.Barrier()

        # Divide ranks over angles, and create local communicator for ranks processing the same set of angles.
        self.theta_rank_ls, self.rank_group_ls = self.allocate_over_theta()
        self.n_theta_local = len(self.theta_rank_ls[rank])
        self.theta_ind_ls_local = self.theta_rank_ls[rank]
        self.local_comm = comm.Split(self.rank_group_ls[rank], rank)
        self.local_rank = self.local_comm.Get_rank()
        self.n_local_ranks = self.local_comm.Get_size()

        self.probe_pos = probe_pos
        self.n_pos_ls = n_pos_ls
        self.probe_pos_ls = probe_pos_ls
        if probe_pos is not None:
            self.n_pos_ls = [len(probe_pos)] * self.n_theta
        self.psi_theta_ls = []
        if not self.common_probe:
            self.probe_real_ls = [[None] * self.n_pos_ls[i] for i in range(self.n_theta)]
            self.probe_imag_ls = [[None] * self.n_pos_ls[i] for i in range(self.n_theta)]
        for i, theta in enumerate(self.theta_ind_ls_local):
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
                self.probe_optimizer.create_param_arrays([self.n_theta, max(self.n_pos_ls), *self.probe_real.shape, 2],
                                                         device=self.device)
            self.probe_optimizer.set_index_in_grad_return(0)

    def allocate_over_theta(self):
        """
        Allocate ranks over MPI ranks.

        :return: A tuple of two lists. Both lists are of length n_ranks, ordered by rank indices.
                 In the first list, each element is a sub-list indicating the indices of angle(s) that the rank
                 should process.
                 In the second list, each element is an integer indicating the index of rank group that the rank
                 belongs to.
        """
        theta_rank_ls = []
        if n_ranks <= self.n_theta:
            surplus = self.n_theta % n_ranks
            i_theta = 0
            mean_thetas_rank = self.n_theta // n_ranks
            for i_rank in range(n_ranks):
                this_theta_rank = list(range(i_theta, i_theta + mean_thetas_rank))
                i_theta += mean_thetas_rank
                if surplus > 0:
                    this_theta_rank += [i_theta]
                    i_theta += 1
                    surplus -= 1
                theta_rank_ls.append(this_theta_rank)
            rank_group_ls = list(range(n_ranks))
        else:
            mean_ranks_theta = np.round(n_ranks / self.n_theta).astype(int)
            theta_rank_ls = np.arange(self.n_theta).astype(int)
            theta_rank_ls = np.repeat(theta_rank_ls, mean_ranks_theta)
            theta_rank_ls = theta_rank_ls.tolist()
            size0 = self.n_theta * mean_ranks_theta
            if size0 > n_ranks:
                theta_rank_ls = theta_rank_ls[:-(size0 - n_ranks)]
            elif size0 < n_ranks:
                theta_rank_ls = theta_rank_ls + [theta_rank_ls[-1]] * (n_ranks - size0)
            theta_rank_ls = np.reshape(theta_rank_ls, [n_ranks, 1]).tolist()
            rank_group_ls = np.reshape(theta_rank_ls, [-1]).tolist()
        return theta_rank_ls, rank_group_ls

    def locate_theta_data(self, i_theta):
        """
        Find which rank has data of a certain theta, and the index of that theta in the rank's local list.

        :return: (target rank, index of data in that rank's local list).
        """
        if n_ranks <= self.n_theta:
            surplus = self.n_theta % n_ranks
            mean_thetas_rank = self.n_theta // n_ranks
            if i_theta < surplus * 2:
                t_rank = i_theta // (mean_thetas_rank + 1)
                t_ind = i_theta % (mean_thetas_rank + 1)
            else:
                t_rank = (i_theta - surplus * (mean_thetas_rank + 1)) // mean_thetas_rank + surplus
                t_ind = (i_theta - surplus * (mean_thetas_rank + 1)) % mean_thetas_rank
        else:
            mean_ranks_theta = np.round(n_ranks / self.n_theta).astype(int)
            t_rank = i_theta * mean_ranks_theta
            t_ind = 0
        return t_rank, t_ind


    def get_batches(self, common_probe_pos=True):
        """
        Batch DPs from the local theta group over local ranks in the same group.

        :param common_probe_pos: Whether probe positions are the same for all angles.
        :return: A list of (i_theta, i_pos) batches.
        """
        n_tot_per_batch = self.minibatch_size * self.n_local_ranks
        for i, i_theta in enumerate(self.theta_ind_ls_local):
            n_pos = len(self.probe_pos) if common_probe_pos else self.n_pos_ls[i_theta]
            spots_ls = range(n_pos)
            if self.randomize_probe_pos:
                spots_ls = np.random.choice(spots_ls, len(spots_ls), replace=False)

            # Append randomly selected diffraction spots if necessary, so that we ensure that all local ranks process
            # the same angle at the same time.
            if n_pos % n_tot_per_batch != 0:
                spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-n_pos % n_tot_per_batch],
                                                                n_tot_per_batch - (n_pos % n_tot_per_batch),
                                                                replace=False))

            # Create task list for the current angle.
            # ind_list_rand is in the format of [((5, 0), (5, 1), ...), ((17, 0), (17, 1), ..., (...))]
            #                                    |___________________|   |_____|
            #                       a batch for all ranks  _|               |_ (i_theta, i_spot)
            #                    (minibatch_size * n_ranks)
            if common_probe_pos:
                # Optimized task distribution for common_probe_pos with lower peak memory.
                if i == 0:
                    ind_list_rand = np.zeros([self.n_theta_local * len(spots_ls), 2], dtype='int32')
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
        return ind_list_rand

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
            wave_real, wave_imag = w.complex_mul(patches[:, :, :, 0], patches[:, :, :, 1], this_probe_mode_real,
                                                 this_probe_mode_imag)
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
        y_ls = self.get_data(this_i_theta, this_ind_batch, theta_downsample=self.theta_downsample)
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

    def get_part2_loss(self, psi=None, w_=None, lambda1=None, g_u=None, lambda2=None):
        return self.get_part2_grad(psi=psi, w_=w_, lambda1=lambda1, g_u=g_u, lambda2=lambda2)[1]

    def get_part2_grad(self, psi=None, w_=None, lambda1=None, g_u=None, lambda2=None):
        if w_ is not None:
            # Assume next subproblem is alignment.
            assert isinstance(self.next_sp, AlignmentSubproblem)
            grad = self.next_sp.rho * (psi - w_ + lambda1 / self.next_sp.rho)
            l_real, l_imag = w.split_channel(grad)
            this_loss = w.sum(l_real ** 2 + l_imag ** 2) / self.next_sp.rho
        elif g_u is not None:
            # Assume next subproblem is backpropagation.
            assert isinstance(self.next_sp, BackpropSubproblem)
            grad = self.next_sp.rho * (psi - g_u + lambda2 / self.next_sp.rho)
            lr, li = w.split_channel(grad)
            this_loss = w.sum(lr ** 2 + li ** 2) / self.next_sp.rho
        else:
            raise ValueError('Check your arguments.')
        return grad, this_loss

    def solve(self, n_iterations=5):
        """Solve subproblem.

        :param n_iterations: Int. Number of inner iterations.
        """
        self.last_iter_part1_loss = 0
        self.last_iter_part2_loss = 0
        grad_psi = w.zeros_like(self.psi_theta_ls[0], requires_grad=False, device=self.device)

        # If next subproblem is backpropagation, prepare the exiting waves beforehand. Rank allocation should
        # follow the pattern of tiled propagation instead of ptychography.
        if isinstance(self.next_sp, BackpropSubproblem):
            print_flush('  PHR: Computing propagation forward pass...', 0, rank, **self.stdout_options)
            my_group, my_theta_ind_ls = self.next_sp.get_my_batch()
            if my_group != -1:
                n_ranks_per_angle = self.next_sp.ranks_per_angle
                my_local_rank = rank % self.next_sp.ranks_per_angle
                szw = self.next_sp.safe_zone_width
                tile_shape = self.next_sp.tile_shape
                tile_shape_padded = np.array(tile_shape) + 2 * szw
                for i_theta in my_theta_ind_ls:
                    print_flush('  PHR: I-theta {} started.'.format(i_theta), 0, rank, same_line=True,
                                **self.stdout_options)
                    u_mmap = self.load_mmap('u_{:04d}'.format(i_theta))
                    u = self.next_sp.prepare_u_tile(u_mmap, my_local_rank)
                    del u_mmap

                    er, ei, psir, psii = self.next_sp.forward(u[None, :], return_intermediate_wavefields=True)
                    ex = w.stack([er, ei], axis=-1)[0]
                    ex = ex[szw:szw + tile_shape[0], szw:szw + tile_shape[1]]

                    # Save intermediate wavefield tiles for later use.
                    psi1 = w.stack([w.stack(psir, axis=-1), w.stack(psii, axis=-1)], axis=-1)[0]
                    psi1 = psi1[szw:szw + tile_shape[0], szw:szw + tile_shape[1], :, :]
                    self.save_variable(psi1, 'psi1_{:05d}_{:04d}'.format(my_local_rank, i_theta))

                    # Group leader rank gathers exiting wave tiles in that angle.
                    if my_local_rank == 0:
                        ex_ls = [ex]
                        for i_tile in range(1, self.next_sp.ranks_per_angle):
                            ex_ls.append(comm.recv(src=my_group * n_ranks_per_angle))
                    else:
                        comm.send(ex, dest=rank // self.next_sp.ranks_per_angle)

                    # Group leader rank assembles exiting wave tiles.
                    if my_local_rank == 0:
                        ex_full = w.zeros([self.whole_object_size[0], self.whole_object_size[1], 2])
                        for i_tile, ex in enumerate(ex_ls):
                            line_st, px_st = self.next_sp.get_tile_position(i_tile)
                            line_end = min([self.whole_object_size[0], line_st + tile_shape[0]])
                            px_end = min([self.whole_object_size[1], px_st + tile_shape[1]])
                            ex_full[line_st:line_end, px_st:px_end, :] = ex[:line_end - line_st, :px_end - px_st, :]
                        self.save_variable(ex_full, 'g_u_{:04d}'.format(i_theta))
            print_flush('  Done.', 0, rank, same_line=True,
                        **self.stdout_options)
            comm.Barrier()

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
        n_tot_per_batch = self.minibatch_size * self.n_local_ranks
        ind_list_rand = self.get_batches(common_probe_pos)
        n_batch = len(ind_list_rand)
        for i_iteration in range(n_iterations):
            # ================================================================================
            # Put diffraction spots from all angles together, and divide into minibatches.
            # ================================================================================
            for i_batch in range(0, n_batch):
                # ================================================================================
                # Initialize batch.
                # ================================================================================
                print_flush('  PHR: Iter {}, batch {} of {} started.'.format(i_iteration, i_batch, n_batch),
                            0, rank, same_line=True, **self.stdout_options)
                starting_batch = 0

                # ================================================================================
                # Get scan position, rotation angle indices, and raw data for current batch.
                # ================================================================================
                t00 = time.time()
                if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                    n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                    ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                this_ind_batch_allranks = ind_list_rand[i_batch]
                this_i_theta = this_ind_batch_allranks[self.local_rank * self.minibatch_size, 0]
                this_local_i_theta = np.where(self.theta_ind_ls_local == this_i_theta)[0][0]
                this_ind_batch = np.sort(
                    this_ind_batch_allranks[self.local_rank * self.minibatch_size:(self.local_rank + 1) * self.minibatch_size, 1])
                probe_pos_int = probe_pos_int if common_probe_pos else probe_pos_int_ls[this_i_theta]
                this_pos_batch = probe_pos_int[this_ind_batch]
                is_last_batch_of_this_theta = i_batch == n_batch - 1 or ind_list_rand[i_batch + 1][0, 0] != this_i_theta
                self.local_comm.Barrier()

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
                patch_ls = self.get_patches(self.psi_theta_ls[this_local_i_theta], this_pos_batch)
                (grad_psi_patch_real_ls, grad_psi_patch_imag_ls), (grad_p_real, grad_p_imag), this_loss = \
                    self.get_part1_grad(patch_ls, this_probe_real, this_probe_imag, this_i_theta, this_pos_batch,
                                        probe_pos_correction, this_ind_batch)
                grad_psi_patch_ls = w.stack([grad_psi_patch_real_ls, grad_psi_patch_imag_ls], axis=-1)
                grad_psi[...] = 0
                grad_psi = self.replace_grad_patches(grad_psi_patch_ls, grad_psi, this_pos_batch, initialize=True)

                # Reduce part-1 gradient within local group for the same angle.
                grad_psi = self.local_comm.allreduce(grad_psi, op=MPI.SUM)
                self.psi_theta_ls[this_local_i_theta] = self.optimizer.apply_gradient(self.psi_theta_ls[this_local_i_theta],
                                                                                      w.cast(grad_psi, 'float32'),
                                                                                      self.total_iter,
                                                                                      params_slicer=[this_i_theta],
                                                                                      **self.optimizer.options_dict)
                if i_iteration == n_iterations - 1:
                    self.last_iter_part1_loss = self.last_iter_part1_loss + this_loss

                if self.optimize_probe and self.total_iter > self.probe_update_delay:
                    p = w.stack([this_probe_real, this_probe_imag], axis=-1)
                    grad_p = w.stack([grad_p_real, grad_p_imag], axis=-1)
                    # Reduce part-1 gradient within local group for the same angle.
                    grad_p = self.local_comm.allreduce(grad_p, op=MPI.SUM) / self.n_local_ranks
                    if self.common_probe:
                        params_slicer = [slice(None)]
                    else:
                        params_slicer = [this_i_theta, this_ind_batch]
                    p = self.probe_optimizer.apply_gradient(p, w.cast(grad_p, 'float32'),
                                                            i_batch + n_batch * i_iteration,
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
                    if isinstance(self.next_sp, AlignmentSubproblem):
                        # TODO: MPI support for this case
                        psi = self.psi_theta_ls[this_local_i_theta]
                        w_ = self.next_sp.w_theta_ls[this_i_theta]
                        lambda1 = self.next_sp.lambda1_theta_ls[this_i_theta]
                        loss_func_args = {'psi': psi, 'w_': w_, 'lambda1': lambda1}
                        grad_psi_2, this_loss = self.get_part2_grad(**loss_func_args)
                    elif isinstance(self.next_sp, BackpropSubproblem):
                        psi = self.psi_theta_ls[this_local_i_theta]
                        g_u = self.load_variable('g_u_{:04d}'.format(this_i_theta))
                        lambda2 = self.load_variable('lambda2_{:04d}'.format(this_i_theta))
                        loss_func_args = {'psi': psi, 'g_u': g_u, 'lambda2': lambda2}
                        grad_psi_2, this_loss = self.get_part2_grad(**loss_func_args)
                    else:
                        raise ValueError('Invalid subproblem dependency.')
                    a = self.optimizer.apply_gradient(psi, w.cast(grad_psi_2, 'float32'),
                                                      i_batch=self.total_iter, **self.optimizer.options_dict,
                                                      params_slicer=[this_local_i_theta])
                    self.psi_theta_ls[this_local_i_theta] = a
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part2_loss = self.last_iter_part2_loss + this_loss
            self.total_iter += 1
        # self.last_iter_part1_loss /= n_batch
        # self.last_iter_part2_loss /= n_theta
        # Dump psi to HDD.
        for i, i_theta in enumerate(self.theta_ind_ls_local):
            self.save_variable(self.psi_theta_ls[i], 'psi_{:04d}'.format(i_theta))
        comm.Barrier()

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
    def __init__(self, whole_object_size, rho=1., optimizer=None, device=None, stdout_options={}):
        """
        Alignment subproblem solver.

        :param whole_object_size: 3D shape of the object to be reconstructed.
        :param device: Device object.
        :param rho: Weight of Lagrangian term.
        """
        super(AlignmentSubproblem, self).__init__(device)
        self.whole_object_size = whole_object_size
        self.rho = rho
        self.optimizer = optimizer
        forward_model = adorym.ForwardModel()
        args = inspect.getfullargspec(self.get_loss).args
        args.pop(0)
        forward_model.argument_ls = args
        forward_model.get_loss_function = lambda: self.get_loss
        self.optimizer.forward_model = forward_model
        self.stdout_options = stdout_options

    def initialize(self, theta_ls=None, output_folder='.'):
        """
        Initialize solver.

        :param theta_ls: List of rotation angles in radians.
        """
        self.setup_temp_folder(output_folder); comm.Barrier()
        self.theta_ls = theta_ls
        self.n_theta = len(theta_ls)
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
                self.next_sp.forward(u_ls, return_intermediate_wavefields=True)
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
                            0, rank, same_line=True, **self.stdout_options)
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
                    grad, (this_part1_loss, this_part2_loss) = self.get_grad(w_ls, u_ls, psi_ls, lambda1_ls, lambda2_ls,
                                                                             recalculate_multislice=False,
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
    def __init__(self, whole_object_size, binning, energy_ev, psize_cm, safe_zone_width=0,
                 rho=1., n_tiles_y=1, n_tiles_x=1, optimizer=None, device=None, stdout_options={}):
        """
        Alignment subproblem solver.

        :param whole_object_size: 3D shape of the object to be reconstructed.
        :param theta_ls: List of rotation angles in radians.
        :param device: Device object.
        :param rho: Weight of Lagrangian term.
        """
        super(BackpropSubproblem, self).__init__(device)
        self.whole_object_size = whole_object_size
        self.binning = binning
        self.energy_ev = energy_ev
        self.psize_cm = psize_cm
        self.safe_zone_width = safe_zone_width
        self.rho = rho
        self.optimizer = optimizer
        forward_model = adorym.ForwardModel()
        args = inspect.getfullargspec(self.get_loss).args
        args.pop(0)
        forward_model.argument_ls = args
        forward_model.get_loss_function = lambda: self.get_loss
        self.optimizer.forward_model = forward_model
        self.stdout_options = stdout_options
        self.n_tiles_y = n_tiles_y
        self.n_tiles_x = n_tiles_x
        self.ranks_per_angle = n_tiles_x * n_tiles_y
        self.tile_shape = [int(np.ceil(self.whole_object_size[0] / n_tiles_y)),
                           int(np.ceil(self.whole_object_size[1] / n_tiles_x))]
        self.tile_shape_padded = [i + 2 * self.safe_zone_width for i in self.tile_shape]

    def initialize(self, theta_ls=None, output_folder=None):
        """
        Initialize solver.

        :param theta_ls: List of rotation angles in radians.
        """
        if self.ranks_per_angle > n_ranks:
            raise ValueError('Number of ranks per angle exceeds total number of ranks. ')
        self.setup_temp_folder(output_folder); comm.Barrier()
        self.theta_ls = theta_ls
        self.n_theta = len(theta_ls)
        theta_ind_ls = list(range(self.n_theta))

        # Divide tasks and create local communicator.
        # Number of ranks in a group is always equal to self.ranks_per_angle = n_tiles_x * n_tiles_y.
        # Surplus ranks are not used.
        # During tiled propagation, each rank processes only one tile.
        self.group_ind, self.theta_ind_ls_local = self.get_my_batch()
        self.n_groups = n_ranks // self.ranks_per_angle
        self.n_theta_local = len(self.theta_ind_ls_local)
        self.local_comm = comm.Split(self.group_ind, rank)
        self.local_rank = self.local_comm.Get_rank()
        self.n_local_ranks = self.ranks_per_angle

        for i_theta in theta_ind_ls[rank::n_ranks]:
            # u arrays are to be stored on HDD, and will get a memmap pointer in self.u_theta_ls.
            u_delta, u_beta = initialize_object_for_dp(self.whole_object_size,
                                                       # random_guess_means_sigmas=[8.7e-7, 5.1e-8, 1e-7, 1e-8],
                                                       random_guess_means_sigmas=[0, 0, 0, 0],
                                                       verbose=False)
            u = np.stack([u_delta, u_beta], axis=-1)
            self.save_variable(u, 'u_{:04d}'.format(i_theta))

            # lambda2 arrays are to be stored both on HDD and in RAM in self.lambda2_theta_ls.
            lmbda2 = np.zeros([*self.whole_object_size[0:2], 2])
            self.save_variable(lmbda2, 'lambda2_{:04d}'.format(i_theta))

        self.optimizer.create_param_arrays([self.n_theta_local, *self.tile_shape, self.whole_object_size[2], 2],
                                           device=self.device)
        self.optimizer.set_index_in_grad_return(0)

    def locate_theta_data(self, i_theta):
        """
        Find which rank has data of a certain theta, and the index of that theta in the rank's local list.

        :return: (target rank, index of data in that rank's local list).
        """
        return ((i_theta % self.n_groups) * self.ranks_per_angle, i_theta // self.n_groups)

    def update_psi_data_mpi(self):
        """
        Update psi data in this subproblem class by gathering psi data in the PHR subproblem classes of other ranks.
        WARNING: This function uses MPI and may raise the Integer Overflow bug.
        """
        assert isinstance(self.prev_sp, PhaseRetrievalSubproblem)
        # I have psi on these i_thetas
        my_psi_itheta_ls = self.prev_sp.theta_ind_ls_local

        dat = [None] * n_ranks
        for i_rank in range(self.ranks_per_angle * self.n_groups):
            that_group = i_rank // self.ranks_per_angle
            dat[i_rank] = [None] * len(self.theta_ls[that_group::self.n_groups])

        for i_local, my_psi_itheta in enumerate(my_psi_itheta_ls):
            # Who need my psi data on this i_theta? Where is this i_theta in their local list?
            t_rank, t_ind = self.locate_theta_data(my_psi_itheta)
            for i_rank in range(t_rank, t_rank + self.ranks_per_angle):
                dat[i_rank][t_ind] = self.prev_sp.psi_theta_ls[i_local]

        dat = comm.alltoall(dat)

        self._psi_theta_ls_local = []
        for i_local, i_theta in enumerate(self.theta_ind_ls_local):
            # Who should send me this i_theta?
            src_rank, _ = self.prev_sp.locate_theta_data(i_theta)
            self._psi_theta_ls_local.append(dat[src_rank][i_local])
        self._psi_theta_ls_local = w.stack(self._psi_theta_ls_local)

    def update_psi_data(self):
        """
        Update psi data in this subproblem class by gathering psi data solved in the PHR subproblem and saved on HDD.
        """
        self._psi_theta_ls_local = []
        for i_local, i_theta in enumerate(self.theta_ind_ls_local):
            psi = self.load_variable('psi_{:04d}'.format(i_theta))
            self._psi_theta_ls_local.append(psi)

    def update_x_data(self):
        """
        Update r(x) tile from distributedly stored x in the TMO subproblem.
        """
        assert isinstance(self.next_sp, TomographySubproblem)
        tile_y, tile_x = self.get_tile_position(self.local_rank)

        self._r_x_ls_local = []
        for i_theta, theta in enumerate(self.theta_ls):
            coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*self.whole_object_size, len(self.theta_ls)),
                                          self.theta_ls[i_theta], reverse=False)
            self.next_sp.x.rotate_array(coord_ls, overwrite_arr=False, override_backend='autograd', override_device='cpu')
            # Create an all-rank batch index array like [(0, 0), (0, 1), (0, 2), ..., (0, n_ranks)].
            this_ind_batch_allranks = np.stack([np.zeros(n_ranks).astype(int), np.arange(n_ranks).astype(int)], axis=-1)
            # Create a probe position array like
            # [(-999, -999), (-999, -999), ..., (-999, -999), (0, 0), (0, 20), (0, 40), ..., (20, 0), ..., (100, 100),
            # (-999, -999), ..., (-999, -999)].
            # Out-of-bound positions are to prevent the read_chunks function from sending data to these ranks.
            # Valid positions correspond to ranks handling the current i_theta.
            probe_pos = np.stack([np.full(n_ranks, fill_value=-self.tile_shape_padded[0] - 1).astype(int),
                                  np.full(n_ranks, fill_value=-self.tile_shape_padded[1] - 1).astype(int)], axis=-1)
            t_rank, _ = self.locate_theta_data(i_theta)
            probe_pos_valid = np.stack(np.mgrid[:self.whole_object_size[0]:self.tile_shape[0],
                                                :self.whole_object_size[1]:self.tile_shape[1]], axis=-1).reshape(-1, 2)
            probe_pos_valid = probe_pos_valid - self.safe_zone_width
            probe_pos[t_rank:t_rank + self.ranks_per_angle] = probe_pos_valid
            x = self.next_sp.x.read_chunks_from_distributed_object(probe_pos, this_ind_batch_allranks, 1,
                                                self.tile_shape_padded, device=self.device, unknown_type='delta_beta',
                                                apply_to_arr_rot=True, dtype='float32',
                                                n_split=self.next_sp.n_all2all_split)
            x.requires_grad = False
            if rank >= t_rank and rank < t_rank + self.ranks_per_angle:
                self._r_x_ls_local.append(x[0])

    def get_my_batch(self):
        """
        Get group index and the list of theta indices to be processed for the current rank.

        :return:
        """
        my_group = rank / self.ranks_per_angle
        if my_group < n_ranks // self.ranks_per_angle - 1e-10:
            my_group = int(my_group)
            n_groups = n_ranks // self.ranks_per_angle
            theta_ind_ls = np.arange(self.n_theta).astype(int)
            my_theta_ind_ls = theta_ind_ls[my_group::n_groups]
        else:
            my_group = -1
            my_theta_ind_ls = []
        return my_group, my_theta_ind_ls

    def get_tile_position(self, i_tile):
        iy = i_tile // self.n_tiles_y
        ix = i_tile % self.n_tiles_x
        return (self.tile_shape[0] * iy, self.tile_shape[1] * ix)

    def forward(self, u_ls, return_intermediate_wavefields=False):
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
        res = multislice_propagate_batch(u_ls, probe_real, probe_imag, self.energy_ev, self.psize_cm,
                                         self.psize_cm, free_prop_cm=0, binning=self.binning,
                                         return_intermediate_wavefields=return_intermediate_wavefields,
                                         device=self.device)
        return res

    def backprop(self, u_ls, probe_real, probe_imag, return_intermediate_wavefields=False):
        """
        Hermitian of operator g.

        :param u_ls: Tensor. A stack of u_theta chunks in shape [n_chunks, chunk_y, chunk_x, chunk_z, 2].
                     All chunks have to be from the same theta.
        :return: Exiting wavefields.
        """
        patch_shape = u_ls[0].shape[1:4]
        res = multislice_backpropagate_batch(u_ls, probe_real, probe_imag, self.energy_ev, self.psize_cm,
                                             self.psize_cm, free_prop_cm=0, binning=self.binning,
                                             return_intermediate_wavefields=return_intermediate_wavefields,
                                             device=self.device)
        return res

    def get_loss(self, u_ls, w_ls, r_x, lambda2_ls, lambda3_ls):
        exit_real, exit_imag = self.forward(u_ls, return_intermediate_wavefields=False)

        grad = -self.rho * (w_ls - w.stack([exit_real, exit_imag], axis=-1) + lambda2_ls / self.rho)
        grad_real, grad_imag = w.split_channel(grad)
        this_part1_loss = w.sum(grad_real ** 2 + grad_imag ** 2) / self.rho

        temp = self.next_sp.rho * (u_ls - r_x + lambda3_ls / self.next_sp.rho)
        temp_delta, temp_beta = w.split_channel(temp)
        this_part2_loss = w.sum(temp_delta ** 2 + temp_beta ** 2) / self.next_sp.rho

        this_loss = this_part1_loss + this_part2_loss
        return this_loss

    def get_grad(self, u_ls, w_ls, r_x, lambda2_ls, lambda3_ls, i_theta, get_multislice_results_from_prevsp=False):
        lmbda_nm = 1240. / self.energy_ev
        delta_nm = self.psize_cm * 1e7
        k = 2. * PI * delta_nm / lmbda_nm
        n_slices = u_ls.shape[-2]

        # Forward pass, store psi'.
        # If this is the first ieration of this subproblem, try to get forward propagation results
        # from the previous subproblem.

        if get_multislice_results_from_prevsp:
            try:
                ex = self.load_variable('g_u_{:04d}'.format(i_theta))
                psi1 = self.load_variable('psi1_{:05d}_{:04d}'.format(self.local_rank, i_theta))
                exit_real, exit_imag = w.split_channel(ex)
                psi_forward_real_ls, psi_forward_imag_ls = w.split_channel(psi1)
            except:
                warnings.warn('Cannot get multislice results from previous subproblem.')
                exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls = \
                    self.forward(u_ls, return_intermediate_wavefields=True)
                psi_forward_real_ls = w.stack(psi_forward_real_ls, axis=3)
                psi_forward_imag_ls = w.stack(psi_forward_imag_ls, axis=3)
        else:
            exit_real, exit_imag, psi_forward_real_ls, psi_forward_imag_ls = \
                self.forward(u_ls, return_intermediate_wavefields=True)
            psi_forward_real_ls = w.stack(psi_forward_real_ls, axis=3)
            psi_forward_imag_ls = w.stack(psi_forward_imag_ls, axis=3)

        # Calculate dL / dg = -rho * [w - g(u) + lambda/rho]
        grad = -self.rho * (w_ls - w.stack([exit_real, exit_imag], axis=-1) + lambda2_ls / self.rho)
        grad_real, grad_imag = w.split_channel(grad)
        this_part1_loss = w.sum(grad_real ** 2 + grad_imag ** 2) / self.rho

        # Back propagate and get psi''.
        _, _, psi_backward_real_ls, psi_backward_imag_ls = \
            self.backprop(u_ls, grad_real, grad_imag, return_intermediate_wavefields=True)
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
        temp = self.next_sp.rho * (u_ls - r_x + lambda3_ls / self.next_sp.rho)
        temp_delta, temp_beta = w.split_channel(temp)
        this_part2_loss = w.sum(temp_delta ** 2 + temp_beta ** 2) / self.next_sp.rho
        grad_delta = grad_delta + temp_delta
        grad_beta = grad_beta + temp_beta

        return w.stack([grad_delta, grad_beta], axis=-1), (this_part1_loss, this_part2_loss)

    def prepare_u_tile(self, u_mmap, my_local_rank):
        """
        Extract u tile from the pointed u array, and pad with safe zone width.

        :param u_mmap: memmap pointer to u array of the current theta.
        :param my_local_rank: local rank, or tile index in the current angle.
        :return: processed u tile.
        """
        tile_y, tile_x = self.get_tile_position(my_local_rank)
        line_st = max([0, tile_y - self.safe_zone_width])
        line_end = min([self.whole_object_size[0], tile_y + self.tile_shape[0] + self.safe_zone_width])
        px_st = max([0, tile_x - self.safe_zone_width])
        px_end = min([self.whole_object_size[1], tile_x + self.tile_shape[1] + self.safe_zone_width])
        u = u_mmap[line_st:line_end, px_st:px_end, :, :]
        u = w.create_variable(u, requires_grad=False, device=self.device)
        u, _ = pad_object_edge(u, self.whole_object_size,
                               np.array([[tile_y - self.safe_zone_width, tile_x - self.safe_zone_width]]),
                               self.tile_shape_padded)
        return u

    def prepare_2d_tile(self, arr, my_local_rank):
        """
        Extract 2D tile from the pointed array, and pad with safe zone width.

        :param arr: array.
        :param my_local_rank: local rank, or tile index in the current angle.
        :return: processed u tile.
        """
        tile_y, tile_x = self.get_tile_position(my_local_rank)
        line_st = max([0, tile_y - self.safe_zone_width])
        line_end = min([self.whole_object_size[0], tile_y + self.tile_shape[0] + self.safe_zone_width])
        px_st = max([0, tile_x - self.safe_zone_width])
        px_end = min([self.whole_object_size[1], tile_x + self.tile_shape[1] + self.safe_zone_width])
        u = arr[line_st:line_end, px_st:px_end, :]
        u = w.create_variable(u, requires_grad=False, device=self.device)
        u, _ = pad_object_edge(u[:, :, None, :], self.whole_object_size,
                               np.array([[tile_y - self.safe_zone_width, tile_x - self.safe_zone_width]]),
                               self.tile_shape_padded)
        u = u[:, :, 0, :]
        return u

    def solve(self, n_iterations=3):
        self.update_psi_data()
        self.update_x_data()
        self.last_iter_part1_loss = 0
        self.last_iter_part2_loss = 0

        tile_y, tile_x = self.get_tile_position(self.local_rank)

        for i_iteration in range(n_iterations):
            for i, i_theta in enumerate(self.theta_ind_ls_local):
                print_flush('  BKP: Iter {}, theta {} started.'.format(i_iteration, i),
                            0, rank, same_line=True, **self.stdout_options)
                u_ls = self.prepare_u_tile(self.load_mmap('u_{:04d}'.format(i_theta)), self.local_rank)[None]
                if isinstance(self.prev_sp, AlignmentSubproblem):
                    x = self.next_sp.x
                    w_ls = self.prev_sp.w_theta_ls[i_theta:i_theta + 1]
                    lambda2_ls = self.load_variable('lambda2_{:04d}'.format(i_theta))[None]
                    lambda3_mmap = self.load_mmap('lambda3_{:04d}'.format(i_theta))
                    lambda3_ls = self.prepare_u_tile(lambda3_mmap, self.local_rank)[None]
                    del lambda3_mmap
                    theta = self.theta_ls[i_theta]
                    loss_func_args = {'u_ls': u_ls, 'w_ls': w_ls, 'x': x, 'lambda2_ls': lambda2_ls,
                                      'lambda3_ls': lambda3_ls, 'theta': theta, 'energy_ev': self.energy_ev,
                                      'psize_cm': self.psize_cm}
                elif isinstance(self.prev_sp, PhaseRetrievalSubproblem):
                    r_x = self._r_x_ls_local[i][None]
                    psi_ls = self.prepare_2d_tile(self._psi_theta_ls_local[i], self.local_rank)[None]
                    lambda2_ls = self.load_variable('lambda2_{:04d}'.format(i_theta))
                    lambda2_ls = self.prepare_2d_tile(lambda2_ls, self.local_rank)[None]
                    lambda3_mmap = self.load_mmap('lambda3_{:04d}'.format(i_theta))
                    lambda3_ls = self.prepare_u_tile(lambda3_mmap, self.local_rank)[None]
                    del lambda3_mmap
                    theta = self.theta_ls[i_theta]
                    loss_func_args = {'u_ls': u_ls, 'w_ls': psi_ls, 'r_x': r_x, 'lambda2_ls': lambda2_ls,
                                      'lambda3_ls': lambda3_ls}
                else:
                    raise ValueError('Invalid dependency of subproblem.')
                self.optimizer.forward_model.update_loss_args(loss_func_args)
                grad_u, (this_part1_loss, this_part2_loss) = \
                    self.get_grad(**loss_func_args, i_theta=i_theta,
                                  get_multislice_results_from_prevsp=(i_iteration == 0))
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
                u_ls = u_ls[0, self.safe_zone_width:self.safe_zone_width + self.tile_shape[0],
                               self.safe_zone_width:self.safe_zone_width + self.tile_shape[1], :, :]
                grad_u = grad_u[0, self.safe_zone_width:self.safe_zone_width + self.tile_shape[0],
                                   self.safe_zone_width:self.safe_zone_width + self.tile_shape[1], :, :]
                u_ls = self.optimizer.apply_gradient(u_ls, w.cast(grad_u, 'float32'), i_batch=self.total_iter,
                                                     params_slicer=i, **self.optimizer.options_dict)
                line_st = tile_y
                line_end = min([tile_y + self.tile_shape[0], self.whole_object_size[0]])
                px_st = tile_x
                px_end = min([tile_x + self.tile_shape[1], self.whole_object_size[1]])
                u_mmap = self.load_mmap('u_{:04d}'.format(i_theta), mode='r+')
                u_ls = w.to_numpy(u_ls[:line_end - line_st, :px_end - px_st, :, :])
                u_mmap[line_st:line_end, px_st:px_end, :, :] = u_ls
                del u_mmap
                if i_iteration == n_iterations - 1:
                    self.last_iter_part1_loss = self.last_iter_part1_loss + this_part1_loss
                    self.last_iter_part2_loss = self.last_iter_part2_loss + this_part2_loss
            self.total_iter += 1
        # self.last_iter_part1_loss = self.last_iter_part1_loss / n_theta
        # self.last_iter_part2_loss = self.last_iter_part2_loss / n_theta

    def update_dual(self):
        r = 0
        self.rsquare = 0
        for i, i_theta in enumerate(self.theta_ind_ls_local):
            u_mmap = self.load_mmap('u_{:04d}'.format(i_theta), mode='r+')
            u_ls = self.prepare_u_tile(u_mmap, self.local_rank)[None]
            del u_mmap
            g_u_real, g_u_imag = self.forward(u_ls, return_intermediate_wavefields=False)
            g_u = w.stack([g_u_real, g_u_imag], axis=-1)[0]
            if isinstance(self.prev_sp, AlignmentSubproblem):
                w_theta_ls = self.prev_sp.w_theta_ls[i]
                this_r = w_theta_ls - g_u
            else:
                psi = self.prepare_2d_tile(self._psi_theta_ls_local[i], self.local_rank)
                this_r = psi - g_u
            this_r = this_r[self.safe_zone_width:self.safe_zone_width + self.tile_shape[0],
                            self.safe_zone_width:self.safe_zone_width + self.tile_shape[0]]
            tile_y, tile_x = self.get_tile_position(self.local_rank)
            line_st = tile_y
            line_end = min([tile_y + self.tile_shape[0], self.whole_object_size[0]])
            px_st = tile_x
            px_end = min([tile_x + self.tile_shape[1], self.whole_object_size[1]])
            lambda2_mmap = self.load_mmap('lambda2_{:04d}'.format(i_theta), mode='r+')
            lambda2 = lambda2_mmap[line_st:line_end, px_st:px_end, :]
            lambda2 = w.create_variable(lambda2, requires_grad=False, device=self.device)
            lambda2 = lambda2 + self.rho * this_r[:line_end - line_st, :px_end - px_st, :]
            lambda2_mmap[line_st:line_end, px_st:px_end, :] = w.to_numpy(lambda2)
            del lambda2_mmap
            rr, ri = w.split_channel(this_r)
            self.rsquare = self.rsquare + w.mean(rr ** 2 + ri ** 2)
        self.rsquare = self.rsquare / self.n_theta


class TomographySubproblem(Subproblem):
    def __init__(self, whole_object_size, rho=1., optimizer=None, device=None, n_all2all_split='auto',
                 stdout_options={}):
        """
        Tomography subproblem solver.

        :param whole_object_size: 3D shape of the object to be reconstructed.
        :param device: Device object.
        :param rho: Weight of Lagrangian term.
        """
        super(TomographySubproblem, self).__init__(device)
        self.whole_object_size = whole_object_size
        self.rho = rho
        self.optimizer = optimizer
        forward_model = adorym.ForwardModel()
        args = inspect.getfullargspec(self.get_loss).args
        args.pop(0)
        forward_model.argument_ls = args
        forward_model.get_loss_function = lambda: self.get_loss
        self.optimizer.forward_model = forward_model
        self.n_all2all_split = n_all2all_split
        self.stdout_options = stdout_options

    def initialize(self, theta_ls=None, output_folder='.'):
        """
        Initialize solver.

        :param theta_ls: List of rotation angles in radians.
        """
        self.setup_temp_folder(output_folder); comm.Barrier()
        self.theta_ls = theta_ls
        self.n_theta = len(theta_ls)

        # x is an adorym.ObjectFunction class in DO mode.
        self.x = adorym.ObjectFunction([*self.whole_object_size, 2], distribution_mode='distributed_object',
                                       output_folder=self.output_folder, device=self.device)
        self.x.initialize_distributed_array_with_zeros()
        self.slice_range_ls = self.x.slice_catalog
        self.slice_range_local = self.slice_range_ls[rank]

        # lambda3 is saved on HDD and have a list of pointers.
        for i_theta, theta in list(enumerate(self.theta_ls))[rank::n_ranks]:
            lmbda3 = np.zeros([*self.whole_object_size, 2])
            self.save_variable(lmbda3, 'lambda3_{:04d}'.format(i_theta))

        self.optimizer.create_distributed_param_arrays(self.x.arr.shape)
        self.optimizer.set_index_in_grad_return(0)

    def forward(self, x, theta):
        return w.rotate(x, theta, axis=0, device=None)

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
        if self.slice_range_local is not None:
            for i_iteration in range(n_iterations):
                np.random.shuffle(theta_ind_ls)
                for i, i_theta in enumerate(theta_ind_ls):
                    print_flush('  TMO: Iter {}, theta {} started.'.format(i_iteration, i),
                                0, rank, same_line=True, **self.stdout_options)
                    u_mmap = self.load_mmap('u_{:04d}'.format(i_theta))
                    u = u_mmap[self.slice_range_local[0]:self.slice_range_local[1]]
                    u = w.create_variable(u, requires_grad=False, device=None)
                    del u_mmap
                    lambda3_mmap = self.load_mmap('lambda3_{:04d}'.format(i_theta), mode='r+')
                    lambda3 = lambda3_mmap[self.slice_range_local[0]:self.slice_range_local[1]]
                    del lambda3_mmap
                    lambda3 = w.create_variable(lambda3, requires_grad=False, device=None)
                    theta = self.theta_ls[i_theta]
                    x = w.create_variable(self.x.arr, requires_grad=False, device=None)
                    loss_func_args = {'x': x, 'u': u, 'lambda3': lambda3, 'theta': theta}
                    self.optimizer.forward_model.update_loss_args(loss_func_args)

                    grad, this_loss = self.get_grad(**loss_func_args)
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
                    x = self.optimizer.apply_gradient(x, w.cast(grad, 'float32'), i_batch=self.total_iter,
                                                      **self.optimizer.options_dict)
                    self.x.arr = w.to_numpy(x)
                    if i_iteration == n_iterations - 1:
                        self.last_iter_part1_loss = self.last_iter_part1_loss + this_loss
                self.total_iter += 1
            # self.last_iter_part1_loss = self.last_iter_part1_loss / n_theta

    def update_dual(self):
        self.rsquare = 0
        if self.slice_range_local is not None:
            for i_theta, theta in enumerate(self.theta_ls):
                u_mmap = self.load_mmap('u_{:04d}'.format(i_theta))
                u = u_mmap[self.slice_range_local[0]:self.slice_range_local[1]]
                u = w.create_variable(u, requires_grad=False, device=None)
                del u_mmap
                x = w.create_variable(self.x.arr, requires_grad=False, device=None)
                this_r = u - self.forward(x, theta)
                lambda3_mmap = self.load_mmap('lambda3_{:04d}'.format(i_theta), mode='r+')
                lambda3 = lambda3_mmap[self.slice_range_local[0]:self.slice_range_local[1]]
                lambda3 = w.create_variable(lambda3, requires_grad=False, device=None)
                lambda3 = lambda3 + self.rho * this_r
                lambda3_mmap[self.slice_range_local[0]:self.slice_range_local[1]] = w.to_numpy(lambda3)
                del lambda3_mmap
                rr, ri = w.split_channel(this_r)
                self.rsquare = self.rsquare + w.mean(rr ** 2 + ri ** 2)
            self.rsquare = self.rsquare / self.n_theta


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
        sp_phr=None, # Phase retrieval subproblem
        sp_aln=None, # Alignment subproblem
        sp_bkp=None, # Backpropagation subproblem
        sp_tmo=None, # Tomography subproblem
        n_iterations_phr=2,
        n_iterations_aln=2,
        n_iterations_bkp=2,
        n_iterations_tmo=2,
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

        sp_phr.initialize([probe_real, probe_imag], prj, theta_ls=theta_ls, probe_pos=probe_pos, output_folder=output_folder)
        if sp_aln is not None: sp_aln.initialize(theta_ls=theta_ls, output_folder=output_folder)
        sp_bkp.initialize(theta_ls=theta_ls, output_folder=output_folder)
        sp_tmo.initialize(theta_ls=theta_ls, output_folder=output_folder)

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

            sp_phr.solve(n_iterations=n_iterations_phr)
            print_flush('PHR done in {} s. Loss: {}/{}.'.format(time.time() - t00, sp_phr.last_iter_part1_loss,
                                                                sp_phr.last_iter_part2_loss), sto_rank, rank)
            for i, i_theta in enumerate(sp_phr.theta_ind_ls_local):
                output_probe(sp_phr.psi_theta_ls[i][:, :, 0], sp_phr.psi_theta_ls[i][:, :, 1],
                             os.path.join(output_folder, 'intermediate', 'ptycho'), custom_name='psi_{}'.format(i_theta),
                             full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)

            t00 = time.time()
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(1, 3)
            # axes[0].imshow(sp_phr.psi_theta_ls[0, :, :, 0])
            # axes[1].imshow(sp_phr.psi_theta_ls[0, :, :, 1])
            # axes[2].imshow(w.arctan2(sp_phr.psi_theta_ls[0, :, :, 1], sp_phr.psi_theta_ls[0, :, :, 0]))
            # plt.show()
            if sp_aln is not None:
                sp_aln.solve(n_iterations=n_iterations_aln)
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
            sp_bkp.solve(n_iterations=n_iterations_bkp)
            print_flush('BKP done in {} s. Loss: {}/{}.'.format(time.time() - t00, sp_bkp.last_iter_part1_loss,
                                                                sp_bkp.last_iter_part2_loss), sto_rank, rank)
            t00 = time.time()
            sp_tmo.solve(n_iterations=n_iterations_tmo)
            print_flush('TMO done in {} s. Loss: {}.'.format(time.time() - t00, sp_tmo.last_iter_part1_loss),
                        sto_rank, rank)

            t00 = time.time()
            if sp_aln is not None:
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
            obj = sp_tmo.x
            output_object(obj, 'distributed_object', os.path.join(output_folder, 'intermediate', 'object'),
                          unknown_type, full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)
            if rank == 0:
                output_probe(sp_phr.probe_real, sp_phr.probe_imag, os.path.join(output_folder, 'intermediate', 'probe'),
                             full_output=False, ds_level=1, i_epoch=i_epoch, save_history=True)

            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                                  time.time() - t0, time.time() - t_zero),
                sto_rank, rank, **stdout_options)
