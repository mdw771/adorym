import autograd.numpy as np
from autograd import grad
from mpi4py import MPI
import dxchange
import time
import os
import h5py
import warnings
import matplotlib.pyplot as plt
from util import *
from misc import *

plt.switch_backend('agg')

PI = 3.1415927


def reconstruct_ptychography(fname, probe_pos, probe_size, obj_size, theta_st=0, theta_end=PI, theta_downsample=None,
                             n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                             alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                             output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                             energy_ev=5000, psize_cm=1e-7, cpu_only=False, save_path='.',
                             phantom_path='phantom', core_parallelization=True, free_prop_cm=None,
                             multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=1,
                             dynamic_rate=True, probe_type='gaussian', probe_initial=None, probe_learning_rate=1e-3,
                             pupil_function=None, probe_circ_mask=0.9, finite_support_mask=None,
                             forward_algorithm='fresnel', dynamic_dropping=False, dropping_threshold=8e-5,
                             n_dp_batch=20, object_type='normal', fresnel_approx=False, **kwargs):

    def calculate_loss(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch):

        obj_stack = np.stack([obj_delta, obj_beta], axis=3)
        obj_rot = apply_rotation(obj_stack, coord_ls[this_i_theta],
                                 'arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta))
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
                                                       probe_imag, energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm='inf',
                                                       obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]])
            exiting_ls.append(exiting)
        exiting_ls = np.concatenate(exiting_ls, 0)
        loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)

        return loss

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()

    # read data
    t0 = time.time()
    print_flush('Reading data...', designate_rank=0, this_rank=rank)
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj = f['exchange/data']
    n_theta = prj.shape[0]
    prj_theta_ind = np.arange(n_theta, dtype=int)
    theta = -np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    if theta_downsample is not None:
        theta = theta[::theta_downsample]
        prj_theta_ind = prj_theta_ind[::theta_downsample]
        n_theta = len(theta)
    original_shape = [n_theta, *prj.shape[1:]]
    print_flush('Data reading: {} s'.format(time.time() - t0), designate_rank=0, this_rank=rank)
    print_flush('Data shape: {}'.format(original_shape), designate_rank=0, this_rank=rank)
    comm.Barrier()

    not_first_level = False

    if output_folder is None:
        output_folder = 'recon_ptycho_minibatch_{}_' \
                        'iter_{}_' \
                        'alphad_{}_' \
                        'alphab_{}_' \
                        'rate_{}_' \
                        'energy_{}_' \
                        'size_{}_' \
                        'ntheta_{}_' \
                        'ms_{}_' \
                        'cpu_{}' \
            .format(minibatch_size,
                    n_epochs, alpha_d, alpha_b,
                    learning_rate, energy_ev,
                    prj.shape[-1], prj.shape[0],
                    multiscale_level, cpu_only)
        if abs(PI - theta_end) < 1e-3:
            output_folder += '_180'
    print_flush('Output folder is {}'.format(output_folder), designate_rank=0, this_rank=rank)

    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

    for ds_level in range(multiscale_level - 1, -1, -1):

        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), designate_rank=0, this_rank=rank)
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
        comm.Barrier()

        # read rotation data
        try:
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)
        except:
            if rank == 0:
                print_flush('Saving rotation coordinates...', designate_rank=0, this_rank=rank)
                save_rotation_lookup(this_obj_size, n_theta)
            comm.Barrier()
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)

        if minibatch_size is None:
            minibatch_size = n_pos

        # unify random seed for all threads
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        if rank == 0:
            if not_first_level == False:
                if initial_guess is None:
                    print_flush('Initializing with Gaussian random.', designate_rank=0, this_rank=rank)
                    obj_delta = np.random.normal(size=this_obj_size, loc=8.7e-7, scale=1e-7)
                    obj_beta = np.random.normal(size=this_obj_size, loc=5.1e-8, scale=1e-8)
                    obj_delta[obj_delta < 0] = 0
                    obj_beta[obj_beta < 0] = 0
                else:
                    print_flush('Using supplied initial guess.', designate_rank=0, this_rank=rank)
                    sys.stdout.flush()
                    obj_delta = initial_guess[0]
                    obj_beta = initial_guess[1]
            else:
                print_flush('Initializing with Gaussian random.', designate_rank=0, this_rank=rank)
                obj_delta = dxchange.read_tiff(os.path.join(output_folder, 'delta_ds_{}.tiff'.format(ds_level * 2)))
                obj_beta = dxchange.read_tiff(os.path.join(output_folder, 'beta_ds_{}.tiff'.format(ds_level * 2)))
                obj_delta = upsample_2x(obj_delta)
                obj_beta = upsample_2x(obj_beta)
                obj_delta += np.random.normal(size=this_obj_size, loc=8.7e-7, scale=1e-7)
                obj_beta += np.random.normal(size=this_obj_size, loc=5.1e-8, scale=1e-8)
                obj_delta[obj_delta < 0] = 0
                obj_beta[obj_beta < 0] = 0
            if object_type == 'phase_only':
                obj_beta[...] = 0
            elif object_type == 'absorption_only':
                obj_delta[...] = 0
            np.save('init_delta_temp.npy', obj_delta)
            np.save('init_beta_temp.npy', obj_beta)
        comm.Barrier()
        if rank != 0:
            obj_delta = np.zeros(this_obj_size)
            obj_beta = np.zeros(this_obj_size)
            obj_delta[:, :, :] = np.load('init_delta_temp.npy')
            obj_beta[:, :, :] = np.load('init_beta_temp.npy')
        comm.Barrier()
        if rank == 0:
            os.remove('init_delta_temp.npy')
            os.remove('init_beta_temp.npy')
        comm.Barrier()

        print_flush('Initialzing probe...', designate_rank=0)
        if probe_type == 'gaussian':
            probe_mag_sigma = kwargs['probe_mag_sigma']
            probe_phase_sigma = kwargs['probe_phase_sigma']
            probe_phase_max = kwargs['probe_phase_max']
            py = np.arange(probe_size[0]) - (probe_size[0] - 1.) / 2
            px = np.arange(probe_size[1]) - (probe_size[1] - 1.) / 2
            pxx, pyy = np.meshgrid(px, py)
            probe_mag = np.exp(-(pxx ** 2 + pyy ** 2) / (2 * probe_mag_sigma ** 2))
            probe_phase = probe_phase_max * np.exp(
                -(pxx ** 2 + pyy ** 2) / (2 * probe_phase_sigma ** 2))
            probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
        elif probe_type == 'optimizable':
            if probe_initial is not None:
                probe_mag, probe_phase = probe_initial
                probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
            else:
                back_prop_cm = (free_prop_cm + (psize_cm * obj_delta.shape[2])) if free_prop_cm is not None else (
                psize_cm * obj_delta.shape[2])
                probe_init = create_probe_initial_guess(os.path.join(save_path, fname), back_prop_cm * 1.e7, energy_ev,
                                                        psize_cm * 1.e7)
                probe_real = probe_init.real
                probe_imag = probe_init.imag
            if pupil_function is not None:
                probe_real = probe_real * pupil_function
                probe_imag = probe_imag * pupil_function
        elif probe_type == 'fixed':
            probe_mag, probe_phase = probe_initial
            probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
        else:
            raise ValueError('Invalid wavefront type. Choose from \'plane\', \'fixed\', \'optimizable\'.')

        # generate Fresnel kernel
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)

        loss_grad = grad(calculate_loss, [0, 1])

        print_flush('Optimizer started.', designate_rank=0, this_rank=rank)
        if rank == 0:
            create_summary(output_folder, locals(), preset='ptycho')

        cont = True
        i_epoch = 0
        while cont:
            n_pos = len(probe_pos)
            n_spots = n_theta * n_pos
            n_tot_per_batch = minibatch_size * size
            n_batch = int(np.ceil(float(n_spots) / n_tot_per_batch))

            m, v = (None, None)
            t0 = time.time()
            spots_ls = range(n_spots)
            ind_list_rand = []

            t00 = time.time()
            print_flush('Allocating jobs over threads...', designate_rank=0, this_rank=rank)
            # Make a list of all thetas and spot positions
            theta_ls = np.arange(n_theta)
            np.random.shuffle(theta_ls)
            for i, i_theta in enumerate(theta_ls):
                spots_ls = range(n_pos)
                if n_pos % minibatch_size != 0:
                    # Append randomly selected diffraction spots if necessary, so that a rank won't be given
                    # spots from different angles in one batch.
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-(n_pos % minibatch_size)],
                                                                    minibatch_size - (n_pos % minibatch_size),
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
            print_flush('Allocation done in {} s.'.format(time.time() - t00), designate_rank=0, this_rank=rank)

            for i_batch in range(n_batch):

                t00 = time.time()
                if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                    n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                    ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                this_ind_batch = ind_list_rand[i_batch]
                this_i_theta = this_ind_batch[rank * minibatch_size, 0]
                this_ind_rank = np.sort(this_ind_batch[rank * minibatch_size:(rank + 1) * minibatch_size, 1])

                this_prj_batch = prj[this_i_theta, this_ind_rank]
                this_pos_batch = probe_pos[this_ind_rank]
                if ds_level > 1:
                    this_prj_batch = this_prj_batch[:, :, ::ds_level, ::ds_level]
                comm.Barrier()
                grads = loss_grad(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch)
                this_grads = np.array(grads)
                grads = np.zeros_like(this_grads)
                comm.Barrier()
                comm.Allreduce(this_grads, grads)
                grads = grads / size
                (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                  grads, i_batch, m, v, step_size=learning_rate)
                obj_delta = np.clip(obj_delta, 0, None)
                obj_beta = np.clip(obj_beta, 0, None)
                if rank == 0:
                    dxchange.write_tiff(obj_delta,
                                        fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                comm.Barrier()
                print_flush('Minibatch done in {} s (rank {})'.format(time.time() - t00, rank))

            if n_epochs == 'auto':
                pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            if dynamic_dropping:
                print_flush('Dynamic dropping...', 0, rank)
                this_loss_table = np.zeros(n_pos)
                loss_table = np.zeros(n_pos)
                fill_start = 0
                ind_list = np.arange(n_pos)
                if n_pos % size != 0:
                    ind_list = np.append(ind_list, np.zeros(size - (n_pos % size)))
                while fill_start < n_pos:
                    this_ind_rank = ind_list[fill_start + rank:fill_start + rank + 1]
                    this_prj_batch = prj[0, this_ind_rank]
                    this_pos_batch = probe_pos[this_ind_rank]
                    this_loss = calculate_loss(obj_delta, obj_beta, 0, this_pos_batch, this_prj_batch)
                    loss_table[fill_start + rank] = this_loss
                    fill_start += size
                comm.Allreduce(this_loss_table, loss_table)
                loss_table = loss_table[:n_pos]
                drop_ls = np.where(loss_table < dropping_threshold)[0]
                np.delete(probe_pos, drop_ls, axis=0)
                print_flush('Dropped {} spot positions.'.format(len(drop_ls)), 0, rank)

            i_epoch = i_epoch + 1

            this_loss = calculate_loss(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch)
            average_loss = 0
            print_flush(
                'Epoch {} (rank {}); loss = {}; Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    this_loss,
                                                                    time.time() - t0, time.time() - t_zero))
            #print_flush(    
            #'Average loss = {}.'.format(comm.Allreduce(this_loss, average_loss)))
            if rank == 0:
                dxchange.write_tiff(obj_delta, fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                    dtype='float32', overwrite=True)
                dxchange.write_tiff(obj_beta, fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)), dtype='float32',
                                overwrite=True)
            print_flush('Current iteration finished.', designate_rank=0, this_rank=rank)
        comm.Barrier()


def reconstruct_ptychography_hdf5(fname, probe_pos, probe_size, obj_size, hdf5_fname='intermediate_obj.h5', theta_st=0, theta_end=PI, theta_downsample=None,
                                  n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                                  alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                                  output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                                  energy_ev=5000, psize_cm=1e-7, cpu_only=False, save_path='.',
                                  phantom_path='phantom', core_parallelization=True, free_prop_cm=None,
                                  multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=1,
                                  dynamic_rate=True, probe_type='gaussian', probe_initial=None, probe_learning_rate=1e-3,
                                  pupil_function=None, probe_circ_mask=0.9, finite_support_mask=None,
                                  forward_algorithm='fresnel', dynamic_dropping=True, dropping_threshold=8e-5,
                                  n_dp_batch=20, object_type='normal', fresnel_approx=False, **kwargs):

    def calculate_loss(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch):

        obj_stack = np.stack([obj_delta, obj_beta], axis=3)
        obj_rot = apply_rotation(obj_stack, coord_ls[this_i_theta],
                                 'arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta))
        probe_pos_batch_ls = []
        exiting_ls = []
        i_dp = 0
        while i_dp < minibatch_size:
            probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
            i_dp += n_dp_batch

        # pad if needed
        pad_arr = np.array([[0, 0], [0, 0]])
        if probe_pos[:, 0].min() - probe_size_half[0] < 0:
            pad_len = probe_size_half[0] - probe_pos[:, 0].min()
            obj_rot = np.pad(obj_rot, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[0, 0] = pad_len
        if probe_pos[:, 0].max() + probe_size_half[0] > this_obj_size[0]:
            pad_len = probe_pos[:, 0].max() + probe_size_half[0] - this_obj_size[0]
            obj_rot = np.pad(obj_rot, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[0, 1] = pad_len
        if probe_pos[:, 1].min() - probe_size_half[1] < 0:
            pad_len = probe_size_half[1] - probe_pos[:, 1].min()
            obj_rot = np.pad(obj_rot, ((0, 0), (pad_len, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[1, 0] = pad_len
        if probe_pos[:, 1].max() + probe_size_half[1] > this_obj_size[1]:
            pad_len = probe_pos[:, 1].max() + probe_size_half[0] - this_obj_size[1]
            obj_rot = np.pad(obj_rot, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode='constant')
            pad_arr[1, 1] = pad_len

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
                                                       probe_imag, energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm='inf',
                                                       obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]])
            exiting_ls.append(exiting)
        exiting_ls = np.concatenate(exiting_ls, 0)
        loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)

        return loss

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()

    # read data
    t0 = time.time()
    print_flush('Reading data...', designate_rank=0, this_rank=rank)
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj = f['exchange/data']
    n_theta = prj.shape[0]
    prj_theta_ind = np.arange(n_theta, dtype=int)
    theta = -np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    if theta_downsample is not None:
        theta = theta[::theta_downsample]
        prj_theta_ind = prj_theta_ind[::theta_downsample]
        n_theta = len(theta)
    original_shape = [n_theta, *prj.shape[1:]]
    print_flush('Data reading: {} s'.format(time.time() - t0), designate_rank=0, this_rank=rank)
    print_flush('Data shape: {}'.format(original_shape), designate_rank=0, this_rank=rank)
    comm.Barrier()

    not_first_level = False

    if output_folder is None:
        output_folder = 'recon_ptycho_minibatch_{}_' \
                        'iter_{}_' \
                        'alphad_{}_' \
                        'alphab_{}_' \
                        'rate_{}_' \
                        'energy_{}_' \
                        'size_{}_' \
                        'ntheta_{}_' \
                        'ms_{}_' \
                        'cpu_{}' \
            .format(minibatch_size,
                    n_epochs, alpha_d, alpha_b,
                    learning_rate, energy_ev,
                    prj.shape[-1], prj.shape[0],
                    multiscale_level, cpu_only)
        if abs(PI - theta_end) < 1e-3:
            output_folder += '_180'
    print_flush('Output folder is {}'.format(output_folder), designate_rank=0, this_rank=rank)

    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

    for ds_level in range(multiscale_level - 1, -1, -1):

        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), designate_rank=0, this_rank=rank)
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
        comm.Barrier()

        # Create parallel HDF5
        f_obj = h5py.File(os.path.join(output_folder, hdf5_fname), 'w', driver='mpio', comm=comm)
        dset = f_obj.create_dataset('obj', shape=(*this_obj_size, 2), dtype='float32')

        # read rotation data
        try:
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)
        except:
            if rank == 0:
                print_flush('Saving rotation coordinates...', designate_rank=0, this_rank=rank)
                save_rotation_lookup(this_obj_size, n_theta)
            comm.Barrier()
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                              n_theta)

        if minibatch_size is None:
            minibatch_size = n_pos

        # unify random seed for all threads
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        if rank == 0:
            if not_first_level == False:
                if initial_guess is None:
                    print_flush('Initializing with Gaussian random.', designate_rank=0, this_rank=rank)
                    obj_delta_init = np.random.normal(size=this_obj_size, loc=8.7e-7, scale=1e-7)
                    obj_beta_init = np.random.normal(size=this_obj_size, loc=5.1e-8, scale=1e-8)
                    obj_delta_init[obj_delta_init < 0] = 0
                    obj_beta_init[obj_beta_init < 0] = 0
                else:
                    print_flush('Using supplied initial guess.', designate_rank=0, this_rank=rank)
                    sys.stdout.flush()
                    obj_delta_init = initial_guess[0]
                    obj_beta_init = initial_guess[1]
            else:
                print_flush('Initializing with Gaussian random.', designate_rank=0, this_rank=rank)
                obj_delta_init = dxchange.read_tiff(os.path.join(output_folder, 'delta_ds_{}.tiff'.format(ds_level * 2)))
                obj_beta_init = dxchange.read_tiff(os.path.join(output_folder, 'beta_ds_{}.tiff'.format(ds_level * 2)))
                obj_delta_init = upsample_2x(obj_delta_init)
                obj_beta_init = upsample_2x(obj_beta_init)
                obj_delta_init += np.random.normal(size=this_obj_size, loc=8.7e-7, scale=1e-7)
                obj_beta_init += np.random.normal(size=this_obj_size, loc=5.1e-8, scale=1e-8)
                obj_delta_init[obj_delta_init < 0] = 0
                obj_beta_init[obj_beta_init < 0] = 0
            if object_type == 'phase_only':
                obj_beta_init[...] = 0
            elif object_type == 'absorption_only':
                obj_delta_init[...] = 0
            dset[:, :, :, 0] = obj_delta_init
            dset[:, :, :, 1] = obj_beta_init
        comm.Barrier()

        print_flush('Initialzing probe...', designate_rank=0)
        if probe_type == 'gaussian':
            probe_mag_sigma = kwargs['probe_mag_sigma']
            probe_phase_sigma = kwargs['probe_phase_sigma']
            probe_phase_max = kwargs['probe_phase_max']
            py = np.arange(probe_size[0]) - (probe_size[0] - 1.) / 2
            px = np.arange(probe_size[1]) - (probe_size[1] - 1.) / 2
            pxx, pyy = np.meshgrid(px, py)
            probe_mag = np.exp(-(pxx ** 2 + pyy ** 2) / (2 * probe_mag_sigma ** 2))
            probe_phase = probe_phase_max * np.exp(
                -(pxx ** 2 + pyy ** 2) / (2 * probe_phase_sigma ** 2))
            probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
        elif probe_type == 'optimizable':
            if probe_initial is not None:
                probe_mag, probe_phase = probe_initial
                probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
            else:
                back_prop_cm = (free_prop_cm + (psize_cm * obj_delta.shape[2])) if free_prop_cm is not None else (
                psize_cm * obj_delta.shape[2])
                probe_init = create_probe_initial_guess(os.path.join(save_path, fname), back_prop_cm * 1.e7, energy_ev,
                                                        psize_cm * 1.e7)
                probe_real = probe_init.real
                probe_imag = probe_init.imag
            if pupil_function is not None:
                probe_real = probe_real * pupil_function
                probe_imag = probe_imag * pupil_function
        elif probe_type == 'fixed':
            probe_mag, probe_phase = probe_initial
            probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
        else:
            raise ValueError('Invalid wavefront type. Choose from \'plane\', \'fixed\', \'optimizable\'.')

        # generate Fresnel kernel
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)

        loss_grad = grad(calculate_loss, [0, 1])

        print_flush('Optimizer started.', designate_rank=0, this_rank=rank)
        if rank == 0:
            create_summary(output_folder, locals(), preset='ptycho')

        cont = True
        i_epoch = 0
        while cont:
            n_pos = len(probe_pos)
            n_spots = n_theta * n_pos
            n_tot_per_batch = minibatch_size * size
            n_batch = int(np.ceil(float(n_spots) / n_tot_per_batch))

            m, v = (None, None)
            t0 = time.time()
            spots_ls = range(n_spots)
            ind_list_rand = []

            t00 = time.time()
            print_flush('Allocating jobs over threads...', designate_rank=0, this_rank=rank)
            # Make a list of all thetas and spot positions
            theta_ls = np.arange(n_theta)
            np.random.shuffle(theta_ls)
            for i, i_theta in enumerate(theta_ls):
                spots_ls = range(n_pos)
                if n_pos % minibatch_size != 0:
                    # Append randomly selected diffraction spots if necessary, so that a rank won't be given
                    # spots from different angles in one batch.
                    spots_ls = np.append(spots_ls, np.random.choice(spots_ls[:-(n_pos % minibatch_size)],
                                                                    minibatch_size - (n_pos % minibatch_size),
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
            print_flush('Allocation done in {} s.'.format(time.time() - t00), designate_rank=0, this_rank=rank)

            for i_batch in range(n_batch):

                t00 = time.time()
                if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                    n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                    ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                this_ind_batch = ind_list_rand[i_batch]
                this_i_theta = this_ind_batch[rank * minibatch_size, 0]
                this_theta = theta_ls[this_i_theta]
                this_ind_rank = np.sort(this_ind_batch[rank * minibatch_size:(rank + 1) * minibatch_size, 1])

                this_prj_batch = prj[this_i_theta, this_ind_rank]
                this_pos_batch = probe_pos[this_ind_rank]
                if ds_level > 1:
                    this_prj_batch = this_prj_batch[:, :, ::ds_level, ::ds_level]
                comm.Barrier()

                # Get values for local chunks of object_delta and beta; interpolate and read directly from HDF5
                obj_delta, obj_beta = get_rotated_subblocks(dset, this_pos_batch, coord_ls, probe_size_half)

                grads = loss_grad(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch)
                this_grads = np.array(grads)
                grads = np.zeros_like(this_grads)
                comm.Barrier()
                comm.Allreduce(this_grads, grads)
                grads = grads / size
                (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                  grads, i_batch, m, v, step_size=learning_rate)
                obj_delta = np.clip(obj_delta, 0, None)
                obj_beta = np.clip(obj_beta, 0, None)
                if rank == 0:
                    dxchange.write_tiff(obj_delta,
                                        fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                comm.Barrier()
                print_flush('Minibatch done in {} s (rank {})'.format(time.time() - t00, rank))

            if n_epochs == 'auto':
                pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            if dynamic_dropping:
                print_flush('Dynamic dropping...', 0, rank)
                this_loss_table = np.zeros(n_pos)
                loss_table = np.zeros(n_pos)
                fill_start = 0
                ind_list = np.arange(n_pos)
                if n_pos % size != 0:
                    ind_list = np.append(ind_list, np.zeros(size - (n_pos % size)))
                while fill_start < n_pos:
                    this_ind_rank = ind_list[fill_start + rank:fill_start + rank + 1]
                    this_prj_batch = prj[0, this_ind_rank]
                    this_pos_batch = probe_pos[this_ind_rank]
                    this_loss = calculate_loss(obj_delta, obj_beta, 0, this_pos_batch, this_prj_batch)
                    loss_table[fill_start + rank] = this_loss
                    fill_start += size
                comm.Allreduce(this_loss_table, loss_table)
                loss_table = loss_table[:n_pos]
                drop_ls = np.where(loss_table < dropping_threshold)[0]
                np.delete(probe_pos, drop_ls, axis=0)
                print_flush('Dropped {} spot positions.'.format(len(drop_ls)), 0, rank)

            i_epoch = i_epoch + 1

            this_loss = calculate_loss(obj_delta, obj_beta, this_i_theta, this_pos_batch, this_prj_batch)
            average_loss = 0
            print_flush(
                'Epoch {} (rank {}); loss = {}; Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    this_loss,
                                                                    time.time() - t0, time.time() - t_zero))
            #print_flush(
            #'Average loss = {}.'.format(comm.Allreduce(this_loss, average_loss)))
            if rank == 0:
                dxchange.write_tiff(obj_delta, fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                    dtype='float32', overwrite=True)
                dxchange.write_tiff(obj_beta, fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)), dtype='float32',
                                overwrite=True)
            print_flush('Current iteration finished.', designate_rank=0, this_rank=rank)
        comm.Barrier()
