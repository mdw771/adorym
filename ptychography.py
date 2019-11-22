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


def reconstruct_ptychography(fname, probe_pos, probe_size, obj_size, theta_st=0, theta_end=PI, n_theta=None, theta_downsample=None,
                             n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                             alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                             output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                             energy_ev=5000, psize_cm=1e-7, cpu_only=False, save_path='.',
                             core_parallelization=True, free_prop_cm=None, optimize_probe_defocusing=False,
                             probe_defocusing_learning_rate=1e-5,
                             multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=1,
                             dynamic_rate=True, probe_type='gaussian', probe_initial=None, probe_learning_rate=1e-3,
                             pupil_function=None, probe_circ_mask=0.9, finite_support_mask=None,
                             forward_algorithm='fresnel', dynamic_dropping=False, dropping_threshold=8e-5,
                             n_dp_batch=20, object_type='normal', fresnel_approx=False, pure_projection=False, two_d_mode=False,
                             shared_file_object=True, **kwargs):

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
                                                           probe_imag, energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm='inf',
                                                           obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                                           fresnel_approx=fresnel_approx, pure_projection=pure_projection)
                exiting_ls.append(exiting)
            exiting_ls = np.concatenate(exiting_ls, 0)
            loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)
            print(loss._value)

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
                                                           free_prop_cm='inf',
                                                           obj_batch_shape=[len(pos_batch), *probe_size,
                                                                            this_obj_size[-1]],
                                                           fresnel_approx=fresnel_approx,
                                                           pure_projection=pure_projection)
                exiting_ls.append(exiting)
                pos_ind += len(pos_batch)
            exiting_ls = np.concatenate(exiting_ls, 0)
            loss = np.mean((np.abs(exiting_ls) - np.abs(this_prj_batch)) ** 2)
            print('loss', loss._value)

        # Write convergence data
        f_conv.write('{},{},{}\n'.format(i_epoch, i_batch, loss._value))
        f_conv.flush()

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

        if shared_file_object:
            # Create parallel npy
            if rank == 0:
                try:
                    os.makedirs(os.path.join(output_folder))
                except:
                    print('Target folder {} exists.'.format(output_folder))
                np.save(os.path.join(output_folder, 'intermediate_obj.npy'), np.zeros([*this_obj_size, 2]))
            comm.Barrier()

            # Create memmap pointer on each rank
            dset = np.load(os.path.join(output_folder, 'intermediate_obj.npy'), mmap_mode='r+', allow_pickle=True)

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
            if not shared_file_object:
                np.save('init_delta_temp.npy', obj_delta)
                np.save('init_beta_temp.npy', obj_beta)
            else:
                dset[:, :, :, 0] = obj_delta
                dset[:, :, :, 1] = obj_beta
        comm.Barrier()

        if not shared_file_object:
            if rank != 0:
                obj_delta = np.zeros(this_obj_size)
                obj_beta = np.zeros(this_obj_size)
                obj_delta[:, :, :] = np.load('init_delta_temp.npy', allow_pickle=True)
                obj_beta[:, :, :] = np.load('init_beta_temp.npy', allow_pickle=True)
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
                print_flush('Estimating probe from measured data...', 0, rank)
                probe_init = create_probe_initial_guess_ptycho(os.path.join(save_path, fname))
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

        opt_arg_ls = [0, 1]
        if probe_type == 'optimizable':
            opt_arg_ls = opt_arg_ls + [2, 3]
        if optimize_probe_defocusing:
            opt_arg_ls.append(4)
        probe_defocus_mm = np.array(0.0)

        loss_grad = grad(calculate_loss, opt_arg_ls)

        # Save convergence data
        try:
            os.makedirs(os.path.join(output_folder, 'convergence'))
        except:
            pass
        f_conv = open(os.path.join(output_folder, 'convergence', 'loss_rank_{}.txt'.format(rank)), 'w')
        f_conv.write('i_epoch,i_batch,loss\n')

        print_flush('Optimizer started.', designate_rank=0, this_rank=rank)
        if rank == 0:
            create_summary(output_folder, locals(), preset='ptycho')

        cont = True
        i_epoch = 0
        m, v, m_p, v_p, m_pd, v_pd = (None, None, None, None, None, None)
        while cont:
            n_pos = len(probe_pos)
            n_spots = n_theta * n_pos
            n_tot_per_batch = minibatch_size * size
            n_batch = int(np.ceil(float(n_spots) / n_tot_per_batch))

            t0 = time.time()
            spots_ls = range(n_spots)
            ind_list_rand = []

            t00 = time.time()
            print_flush('Allocating jobs over threads...', designate_rank=0, this_rank=rank)
            # Make a list of all thetas and spot positions'
            if not two_d_mode:
                theta_ls = np.arange(n_theta)
                np.random.shuffle(theta_ls)
            else:
                theta_ls = np.linspace(0, 2 * PI, prj.shape[0])
                theta_ls = abs(theta_ls - theta_st) < 1e-5
                i_theta = np.nonzero(theta_ls)[0][0]
                theta_ls = np.array([i_theta])
                print(theta_ls)
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

                if shared_file_object:
                    # Get values for local chunks of object_delta and beta; interpolate and read directly from HDF5
                    obj = get_rotated_subblocks(dset, this_pos_batch, coord_ls[this_i_theta], probe_size_half)
                    obj_delta = np.array(obj[:, :, :, :, 0])
                    obj_beta = np.array(obj[:, :, :, :, 1])

                grads = loss_grad(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm, this_i_theta, this_pos_batch, this_prj_batch)
                grads = list(grads)

                if shared_file_object:
                    obj_grads = np.array(grads[:2])
                else:
                    this_obj_grads = np.array(grads[:2])
                    obj_grads = np.zeros_like(this_obj_grads)
                    comm.Barrier()
                    comm.Allreduce(this_obj_grads, obj_grads)
                obj_grads = obj_grads / size
                if not shared_file_object:
                    (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                      obj_grads, i_batch, m, v, step_size=learning_rate)
                else:
                    (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                      obj_grads, 0, None, None, step_size=learning_rate)
                obj_delta = np.clip(obj_delta, 0, None)
                obj_beta = np.clip(obj_beta, 0, None)
                if object_type == 'absorption_only': obj_delta[...] = 0
                if object_type == 'phase_only': obj_beta[...] = 0

                if probe_type == 'optimizable':
                    this_probe_grads = np.array(grads[2:4])
                    probe_grads = np.zeros_like(this_probe_grads)
                    comm.Barrier()
                    comm.Allreduce(this_probe_grads, probe_grads)
                    probe_grads = probe_grads / size
                    (probe_real, probe_imag), m_p, v_p = apply_gradient_adam(np.array([probe_real, probe_imag]),
                                                                          probe_grads, i_epoch, m_p, v_p, step_size=probe_learning_rate)

                if optimize_probe_defocusing:
                    this_pd_grad = np.array(grads[len(opt_arg_ls) - 1])
                    print(this_pd_grad)
                    pd_grads = np.array(0.0)
                    comm.Barrier()
                    comm.Allreduce(this_pd_grad, pd_grads)
                    pd_grads = pd_grads / size
                    print(this_pd_grad, pd_grads)
                    probe_defocus_mm, m_pd, v_pd = apply_gradient_adam(probe_defocus_mm,
                                                                       pd_grads, 0, None, None,
                                                                       step_size=probe_defocusing_learning_rate)
                    # probe_defocus_mm = probe_defocus_mm - probe_defocusing_learning_rate * pd_grads
                    print_flush('Probe defocus is {} mm.'.format(probe_defocus_mm), 0, rank)

                if shared_file_object:
                    obj_delta = obj_delta - obj[:, :, :, :, 0]
                    obj_beta = obj_beta - obj[:, :, :, :, 1]
                    write_subblocks_to_file(dset, this_pos_batch, obj_delta, obj_beta, coord_ls[this_i_theta],
                                            probe_size_half)
                    comm.Barrier()

                if rank == 0:
                    if shared_file_object:
                        dxchange.write_tiff(dset[:, :, :, 0],
                                            fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                            dtype='float32', overwrite=True)
                        dxchange.write_tiff(dset[:, :, :, 0],
                                            fname=os.path.join(output_folder, 'current/delta_{}'.format(i_batch)),
                                            dtype='float32', overwrite=True)
                    else:
                        dxchange.write_tiff(obj_delta,
                                            fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                            dtype='float32', overwrite=True)
                comm.Barrier()
                print_flush('Minibatch done in {} s (rank {})'.format(time.time() - t00, rank))

            if n_epochs == 'auto':
                pass
            else:
                if i_epoch == n_epochs - 1: cont = False

            # if dynamic_dropping:
            #     print_flush('Dynamic dropping...', 0, rank)
            #     this_loss_table = np.zeros(n_pos)
            #     loss_table = np.zeros(n_pos)
            #     fill_start = 0
            #     ind_list = np.arange(n_pos)
            #     if n_pos % size != 0:
            #         ind_list = np.append(ind_list, np.zeros(size - (n_pos % size)))
            #     while fill_start < n_pos:
            #         this_ind_rank = ind_list[fill_start + rank:fill_start + rank + 1]
            #         this_prj_batch = prj[0, this_ind_rank]
            #         this_pos_batch = probe_pos[this_ind_rank]
            #         this_loss = calculate_loss(obj_delta, obj_beta, 0, this_pos_batch, this_prj_batch)
            #         loss_table[fill_start + rank] = this_loss
            #         fill_start += size
            #     comm.Allreduce(this_loss_table, loss_table)
            #     loss_table = loss_table[:n_pos]
            #     drop_ls = np.where(loss_table < dropping_threshold)[0]
            #     np.delete(probe_pos, drop_ls, axis=0)
            #     print_flush('Dropped {} spot positions.'.format(len(drop_ls)), 0, rank)

            i_epoch = i_epoch + 1

            # this_loss = calculate_loss(obj_delta, obj_beta, probe_real, probe_imag, this_i_theta, this_pos_batch, this_prj_batch)
            average_loss = 0
            print_flush(
                'Epoch {} (rank {}); Delta-t = {} s; current time = {} s,'.format(i_epoch, rank,
                                                                    time.time() - t0, time.time() - t_zero))
            #print_flush(    
            #'Average loss = {}.'.format(comm.Allreduce(this_loss, average_loss)))
            if rank == 0:
                if shared_file_object:
                    dxchange.write_tiff(dset[:, :, :, 0],
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(dset[:, :, :, 1],
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.sqrt(probe_real ** 2 + probe_imag ** 2),
                                        fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.arctan2(probe_imag, probe_real),
                                        fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                else:
                    dxchange.write_tiff(obj_delta,
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(obj_beta,
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.sqrt(probe_real ** 2 + probe_imag ** 2),
                                        fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(np.arctan2(probe_imag, probe_real),
                                        fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
            print_flush('Current iteration finished.', designate_rank=0, this_rank=rank)
        comm.Barrier()
