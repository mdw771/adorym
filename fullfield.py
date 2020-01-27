import numpy
print('Numpy is {}'.format(numpy.__file__))
import autograd.numpy as np
from autograd import grad
from mpi4py import MPI
import dxchange
import time
import os
import h5py
import warnings
from util import *
from misc import *
from optimizers import *

PI = 3.1415927


def reconstruct_fullfield(fname, theta_st=0, theta_end=PI, n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                          alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                          output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                          energy_ev=5000, psize_cm=1e-7, n_epochs_mask_release=None, cpu_only=False, save_path='.',
                          shrink_cycle=10, core_parallelization=True, free_prop_cm=None,
                          multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=5,
                          dynamic_rate=True, probe_type='plane', probe_initial=None, probe_learning_rate=1e-3,
                          pupil_function=None, theta_downsample=None, forward_algorithm='fresnel', random_theta=True,
                          object_type='normal', fresnel_approx=True, shared_file_object=False, reweighted_l1=False,
                          **kwargs):
    """
    Reconstruct a beyond depth-of-focus object.
    :param fname: Filename and path of raw data file. Must be in HDF5 format.
    :param theta_st: Starting rotation angle.
    :param theta_end: Ending rotation angle.
    :param n_epochs: Number of epochs to be executed. If given 'auto', optimizer will stop
                     when reduction rate of loss function goes below crit_conv_rate.
    :param crit_conv_rate: Reduction rate of loss function below which the optimizer should
                           stop.
    :param max_nepochs: The maximum number of epochs to be executed if n_epochs is 'auto'.
    :param alpha: Weighting coefficient for both delta and beta regularizer. Should be None
                  if alpha_d and alpha_b are specified.
    :param alpha_d: Weighting coefficient for delta regularizer.
    :param alpha_b: Weighting coefficient for beta regularizer.
    :param gamma: Weighting coefficient for TV regularizer.
    :param learning_rate: Learning rate of ADAM.
    :param output_folder: Name of output folder. Put None for auto-generated pattern.
    :param downsample: Downsampling (not implemented yet).
    :param minibatch_size: Size of minibatch.
    :param save_intermediate: Whether to save the object after each epoch.
    :param energy_ev: Beam energy in eV.
    :param psize_cm: Pixel size in cm.
    :param n_epochs_mask_release: The number of epochs after which the finite support mask
                                  is released. Put None to disable this feature.
    :param cpu_only: Whether to disable GPU.
    :param save_path: The location of finite support mask, the prefix of output_folder and
                      other metadata.
    :param shrink_cycle: Shrink-wrap is executed per every this number of epochs.
    :param core_parallelization: Whether to use Horovod for parallelized computation within
                                 this function.
    :param free_prop_cm: The distance to propagate the wavefront in free space after exiting
                         the sample, in cm.
    :param multiscale_level: The level of multiscale processing. When this number is m and
                             m > 1, m - 1 low-resolution reconstructions will be performed
                             before reconstructing with the original resolution. The downsampling
                             factor for these coarse reconstructions will be [2^(m - 1),
                             2^(m - 2), ..., 2^1].
    :param n_epoch_final_pass: specify a number of iterations for the final pass if multiscale
                               is activated. If None, it will be the same as n_epoch.
    :param initial_guess: supply an initial guess. If None, object will be initialized with noises.
    :param n_batch_per_update: number of minibatches during which gradients are accumulated, after
                               which obj is updated.
    :param dynamic_rate: when n_batch_per_update > 1, adjust learning rate dynamically to allow it
                         to decrease with epoch number
    :param probe_type: type of wavefront. Can be 'plane', '  fixed', or 'optimizable'. If 'optimizable',
                           the probe function will be optimized along with the object.
    :param probe_initial: can be provided for 'optimizable' probe_type, and must be provided for
                              'fixed'.
    """

    def calculate_loss(obj_delta, obj_beta, this_ind_batch, this_prj_batch):

        if not shared_file_object:
            obj_stack = np.stack([obj_delta, obj_beta], axis=3)
            obj_rot_batch = []
            for i in range(minibatch_size):
                obj_rot_batch.append(apply_rotation(obj_stack, coord_ls[this_ind_batch[i]],
                                                    'arrsize_{}_{}_{}_ntheta_{}'.format(dim_y, dim_x, dim_x, n_theta)))
            obj_rot_batch = np.stack(obj_rot_batch)

            exiting_batch = multislice_propagate_batch_numpy(obj_rot_batch[:, :, :, :, 0], obj_rot_batch[:, :, :, :, 1],
                                                             probe_real, probe_imag, energy_ev,
                                                             psize_cm * ds_level, free_prop_cm=free_prop_cm,
                                                             obj_batch_shape=[minibatch_size, *this_obj_size],
                                                             kernel=h, fresnel_approx=fresnel_approx)
            loss = np.mean((np.abs(exiting_batch) - np.abs(this_prj_batch)) ** 2)

        else:
            exiting_batch = multislice_propagate_batch_numpy(obj_delta, obj_beta,
                                                             probe_real, probe_imag, energy_ev,
                                                             psize_cm * ds_level, free_prop_cm=free_prop_cm,
                                                             obj_batch_shape=obj_delta.shape,
                                                             kernel=h, fresnel_approx=fresnel_approx)
            exiting_batch = exiting_batch[:,
                                          safe_zone_width:exiting_batch.shape[1] - safe_zone_width,
                                          safe_zone_width:exiting_batch.shape[2] - safe_zone_width]
            loss = np.mean((np.abs(exiting_batch) - np.abs(this_prj_batch)) ** 2)

        dxchange.write_tiff(np.squeeze(abs(exiting_batch._value)), 'cone_256_foam/test_shared_file_object/current/exit_{}'.format(rank),
                            dtype='float32', overwrite=True)
        dxchange.write_tiff(np.squeeze(abs(this_prj_batch)), 'cone_256_foam/test_shared_file_object/current/prj_{}'.format(rank),
                            dtype='float32', overwrite=True)

        reg_term = 0
        if reweighted_l1:
            if alpha_d not in [None, 0]:
                reg_term = reg_term + alpha_d * np.mean(weight_l1 * np.abs(obj_delta))
                loss = loss + reg_term
            if alpha_b not in [None, 0]:
                reg_term = reg_term + alpha_b * np.mean(weight_l1 * np.abs(obj_beta))
                loss = loss + reg_term
        else:
            if alpha_d not in [None, 0]:
                reg_term = reg_term + alpha_d * np.mean(np.abs(obj_delta))
                loss = loss + reg_term
            if alpha_b not in [None, 0]:
                reg_term = reg_term + alpha_b * np.mean(np.abs(obj_beta))
                loss = loss + reg_term
        if gamma not in [None, 0]:
            if shared_file_object:
                reg_term = reg_term + gamma * total_variation_3d(obj_delta, axis_offset=1)
            else:
                reg_term = reg_term + gamma * total_variation_3d(obj_delta, axis_offset=0)
            loss = loss + reg_term

        print('Loss:', loss._value, 'Regularization term:', reg_term._value if reg_term != 0 else 0)

        # if alpha_d is None:
        #     reg_term = alpha * (np.sum(np.abs(obj_delta)) + np.sum(np.abs(obj_delta))) + gamma * total_variation_3d(
        #         obj_delta)
        # else:
        #     if gamma == 0:
        #         reg_term = alpha_d * np.sum(np.abs(obj_delta)) + alpha_b * np.sum(np.abs(obj_beta))
        #     else:
        #         reg_term = alpha_d * np.sum(np.abs(obj_delta)) + alpha_b * np.sum(
        #             np.abs(obj_beta)) + gamma * total_variation_3d(obj_delta)
        # loss = loss + reg_term

        # Write convergence data
        f_conv.write('{},{},{},'.format(i_epoch, i_batch, loss._value))
        f_conv.flush()

        return loss

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()

    # read data
    t0 = time.time()
    print_flush('Reading data...', 0, rank)
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj_0 = f['exchange/data']
    theta = -np.linspace(theta_st, theta_end, prj_0.shape[0], dtype='float32')
    n_theta = len(theta)
    prj_theta_ind = np.arange(n_theta, dtype=int)
    if theta_downsample is not None:
        prj_0 = prj_0[::theta_downsample]
        theta = theta[::theta_downsample]
        prj_theta_ind = prj_theta_ind[::theta_downsample]
        n_theta = len(theta)
    original_shape = prj_0.shape
    comm.Barrier()
    print_flush('Data reading: {} s'.format(time.time() - t0), 0, rank)
    print_flush('Data shape: {}'.format(original_shape), 0, rank)
    comm.Barrier()

    if output_folder is None:
        output_folder = 'recon_360_minibatch_{}_' \
                        'mskrls_{}_' \
                        'shrink_{}_' \
                        'iter_{}_' \
                        'alphad_{}_' \
                        'alphab_{}_' \
                        'gamma_{}_' \
                        'rate_{}_' \
                        'energy_{}_' \
                        'size_{}_' \
                        'ntheta_{}_' \
                        'prop_{}_' \
                        'ms_{}_' \
                        'cpu_{}' \
            .format(minibatch_size, n_epochs_mask_release, shrink_cycle,
                    n_epochs, alpha_d, alpha_b,
                    gamma, learning_rate, energy_ev,
                    prj_0.shape[-1], prj_0.shape[0], free_prop_cm,
                    multiscale_level, cpu_only)
        if abs(PI - theta_end) < 1e-3:
            output_folder += '_180'

    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

    for ds_level in range(multiscale_level - 1, -1, -1):

        initializer_flag = False if ds_level == range(multiscale_level - 1, -1, -1)[0] else True

        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), 0, rank)
        comm.Barrier()

        # Physical metadata
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]

        # downsample data
        prj = prj_0
        # prj = np.copy(prj_0)
        # if ds_level > 1:
        #     prj = prj[:, ::ds_level, ::ds_level]
        #     prj = prj.astype('complex64')
        # comm.Barrier()

        dim_y, dim_x = prj.shape[-2] // ds_level, prj.shape[-1] // ds_level
        this_obj_size = [dim_y, dim_x, dim_x]
        comm.Barrier()

        if shared_file_object:
            # Create parallel npy
            if rank == 0:
                try:
                    os.makedirs(os.path.join(output_folder))
                except:
                    print('Target folder {} exists.'.format(output_folder))
                np.save(os.path.join(output_folder, 'intermediate_obj.npy'), np.zeros([*this_obj_size, 2]))
                np.save(os.path.join(output_folder, 'intermediate_m.npy'), np.zeros([*this_obj_size, 2]))
                np.save(os.path.join(output_folder, 'intermediate_v.npy'), np.zeros([*this_obj_size, 2]))
            comm.Barrier()

            # Create memmap pointer on each rank
            dset = np.load(os.path.join(output_folder, 'intermediate_obj.npy'), mmap_mode='r+', allow_pickle=True)
            dset_m = np.load(os.path.join(output_folder, 'intermediate_m.npy'), mmap_mode='r+', allow_pickle=True)
            dset_v = np.load(os.path.join(output_folder, 'intermediate_v.npy'), mmap_mode='r+', allow_pickle=True)

            # Get block allocation
            n_blocks_y, n_blocks_x, n_blocks, block_size = get_block_division(this_obj_size, n_ranks)
            print_flush('Number of blocks in y: {}'.format(n_blocks_y), 0, rank)
            print_flush('Number of blocks in x: {}'.format(n_blocks_x), 0, rank)
            print_flush('Block size: {}'.format(block_size), 0, rank)
            probe_pos = []
            # probe_pos is a list of tuples of (line_st, line_end, px_st, ps_end).
            for i_pos in range(n_blocks):
                probe_pos.append(get_block_range(i_pos, n_blocks_x, block_size)[:4])
            probe_pos = np.array(probe_pos)
            if free_prop_cm not in [0, None]:
                safe_zone_width = ceil(4.0 * np.sqrt((delta_nm * dim_x + free_prop_cm * 1e7) * lmbda_nm) / (voxel_nm[0]))
            else:
                safe_zone_width = ceil(4.0 * np.sqrt((delta_nm * dim_x) * lmbda_nm) / (voxel_nm[0]))
            print_flush('safe zone: {}'.format(safe_zone_width), 0, rank)

        # read rotation data
        try:
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(dim_y, dim_x, dim_x, n_theta),
                                              n_theta)
        except:
            save_rotation_lookup([dim_y, dim_x, dim_x], n_theta)
            coord_ls = read_all_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(dim_y, dim_x, dim_x, n_theta),
                                              n_theta)

        if minibatch_size is None:
            minibatch_size = n_theta

        if n_epochs_mask_release is None:
            n_epochs_mask_release = np.inf

        if (not shared_file_object) or (shared_file_object and rank == 0):
            try:
                mask = dxchange.read_tiff_stack(os.path.join(save_path, 'fin_sup_mask', 'mask_00000.tiff'),
                                                range(prj_0.shape[1]))
            except:
                try:
                    mask = dxchange.read_tiff(os.path.join(save_path, 'fin_sup_mask', 'mask.tiff'))
                except:
                    obj_pr = dxchange.read_tiff_stack(os.path.join(save_path, 'paganin_obj/recon_00000.tiff'),
                                                      range(prj_0.shape[1]), 5)
                    obj_pr = gaussian_filter(np.abs(obj_pr), sigma=3, mode='constant')
                    mask = np.zeros_like(obj_pr)
                    mask[obj_pr > 1e-5] = 1
                    dxchange.write_tiff_stack(mask, os.path.join(save_path, 'fin_sup_mask/mask'), dtype='float32',
                                              overwrite=True)
            if ds_level > 1:
                mask = mask[::ds_level, ::ds_level, ::ds_level]
            if shared_file_object:
                np.save(os.path.join(output_folder, 'intermediate_mask.npy'), mask)
        comm.Barrier()

        if shared_file_object:
            dset_mask = np.load(os.path.join(output_folder, 'intermediate_mask.npy'), mmap_mode='r+', allow_pickle=True)

        # unify random seed for all threads
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        if rank == 0:
            if initializer_flag == False:
                if initial_guess is None:
                    print_flush('Initializing with Gaussian random.', 0, rank)
                    obj_delta = np.random.normal(size=[dim_y, dim_x, dim_x], loc=8.7e-7, scale=1e-7) * mask
                    obj_beta = np.random.normal(size=[dim_y, dim_x, dim_x], loc=5.1e-8, scale=1e-8) * mask
                    obj_delta[obj_delta < 0] = 0
                    obj_beta[obj_beta < 0] = 0
                else:
                    print_flush('Using supplied initial guess.', 0, rank)
                    sys.stdout.flush()
                    obj_delta = initial_guess[0]
                    obj_beta = initial_guess[1]
            else:
                print_flush('Initializing previous pass outcomes.', 0, rank)
                obj_delta = dxchange.read_tiff(os.path.join(output_folder, 'delta_ds_{}.tiff'.format(ds_level * 2)))
                obj_beta = dxchange.read_tiff(os.path.join(output_folder, 'beta_ds_{}.tiff'.format(ds_level * 2)))
                obj_delta = upsample_2x(obj_delta)
                obj_beta = upsample_2x(obj_beta)
                obj_delta += np.random.normal(size=[dim_y, dim_x, dim_x], loc=8.7e-7, scale=1e-7) * mask
                obj_beta += np.random.normal(size=[dim_y, dim_x, dim_x], loc=5.1e-8, scale=1e-8) * mask
                obj_delta[obj_delta < 0] = 0
                obj_beta[obj_beta < 0] = 0
            obj_size = obj_delta.shape
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
                dset_m[...] = 0
                dset_v[...] = 0
        comm.Barrier()

        if not shared_file_object:
            obj_delta = np.zeros(this_obj_size)
            obj_beta = np.zeros(this_obj_size)
            obj_delta[:, :, :] = np.load('init_delta_temp.npy', allow_pickle=True)
            obj_beta[:, :, :] = np.load('init_beta_temp.npy', allow_pickle=True)
            comm.Barrier()
            if rank == 0:
                os.remove('init_delta_temp.npy')
                os.remove('init_beta_temp.npy')
            comm.Barrier()

        print_flush('Initialzing probe...', 0, rank)
        if not shared_file_object:
            if probe_type == 'plane':
                probe_real = np.ones([dim_y, dim_x])
                probe_imag = np.zeros([dim_y, dim_x])
            elif probe_type == 'optimizable':
                if probe_initial is not None:
                    probe_mag, probe_phase = probe_initial
                    probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
                else:
                    # probe_mag = np.ones([dim_y, dim_x])
                    # probe_phase = np.zeros([dim_y, dim_x])
                    back_prop_cm = (free_prop_cm + (psize_cm * obj_size[2])) if free_prop_cm is not None else (
                    psize_cm * obj_size[2])
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
            elif probe_type == 'point':
                # this should be in spherical coordinates
                probe_real = np.ones([dim_y, dim_x])
                probe_imag = np.zeros([dim_y, dim_x])
            elif probe_type == 'gaussian':
                probe_mag_sigma = kwargs['probe_mag_sigma']
                probe_phase_sigma = kwargs['probe_phase_sigma']
                probe_phase_max = kwargs['probe_phase_max']
                py = np.arange(obj_size[0]) - (obj_size[0] - 1.) / 2
                px = np.arange(obj_size[1]) - (obj_size[1] - 1.) / 2
                pxx, pyy = np.meshgrid(px, py)
                probe_mag = np.exp(-(pxx ** 2 + pyy ** 2) / (2 * probe_mag_sigma ** 2))
                probe_phase = probe_phase_max * np.exp(
                    -(pxx ** 2 + pyy ** 2) / (2 * probe_phase_sigma ** 2))
                probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
            else:
                raise ValueError('Invalid wavefront type. Choose from \'plane\', \'fixed\', \'optimizable\'.')
        else:
            if probe_type == 'plane':
                probe_real = np.ones([block_size + 2 * safe_zone_width] * 2)
                probe_imag = np.zeros([block_size + 2 * safe_zone_width] * 2)
            else:
                raise ValueError('probe_type other than plane is not yet supported with shared file object.')

        # =============finite support===================
        if not shared_file_object:
            obj_delta = obj_delta * mask
            obj_beta = obj_beta * mask
            obj_delta = np.clip(obj_delta, 0, None)
            obj_beta = np.clip(obj_beta, 0, None)
        # ==============================================

        # generate Fresnel kernel
        if not shared_file_object:
            h = get_kernel(delta_nm, lmbda_nm, voxel_nm, [dim_y, dim_y, dim_x], fresnel_approx=fresnel_approx)
        else:
            h = get_kernel(delta_nm, lmbda_nm, voxel_nm, [block_size + safe_zone_width * 2,
                                                          block_size + safe_zone_width * 2,
                                                          dim_x], fresnel_approx=fresnel_approx)

        loss_grad = grad(calculate_loss, [0, 1])

        # Save convergence data
        try:
            os.makedirs(os.path.join(output_folder, 'convergence'))
        except:
            pass
        f_conv = open(os.path.join(output_folder, 'convergence', 'loss_rank_{}.txt'.format(rank)), 'w')
        f_conv.write('i_epoch,i_batch,loss,time\n')

        print_flush('Optimizer started.', 0, rank)
        if rank == 0:
            create_summary(output_folder, locals(), preset='fullfield')

        cont = True
        i_epoch = 0
        while cont:
            if shared_file_object:
                # Do a ptychography-like allocation.
                n_pos = len(probe_pos)
                n_spots = n_theta * n_pos
                n_tot_per_batch = minibatch_size * n_ranks
                n_batch = int(np.ceil(float(n_spots) / n_tot_per_batch))
                spots_ls = range(n_spots)
                ind_list_rand = []

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
                            [ind_list_rand, np.vstack([np.array([i_theta] * len(spots_ls)), spots_ls]).transpose()],
                            axis=0)
                ind_list_rand = split_tasks(ind_list_rand, n_tot_per_batch)
                probe_size_half = block_size // 2 + safe_zone_width
            else:
                ind_list_rand = np.arange(n_theta)
                np.random.shuffle(ind_list_rand)
                n_tot_per_batch = n_ranks * minibatch_size
                if n_theta % n_tot_per_batch > 0:
                    ind_list_rand = np.concatenate([ind_list_rand, ind_list_rand[:n_tot_per_batch - n_theta % n_tot_per_batch]])
                ind_list_rand = split_tasks(ind_list_rand, n_tot_per_batch)
                ind_list_rand = [np.sort(x) for x in ind_list_rand]

            m, v = (None, None)
            t0 = time.time()
            for i_batch in range(len(ind_list_rand)):

                t00 = time.time()
                if not shared_file_object:
                    this_ind_batch = ind_list_rand[i_batch][rank * minibatch_size:(rank + 1) * minibatch_size]
                    this_prj_batch = prj[this_ind_batch, ::ds_level, ::ds_level]
                else:
                    if len(ind_list_rand[i_batch]) < n_tot_per_batch:
                        n_supp = n_tot_per_batch - len(ind_list_rand[i_batch])
                        ind_list_rand[i_batch] = np.concatenate([ind_list_rand[i_batch], ind_list_rand[0][:n_supp]])

                    this_ind_batch = ind_list_rand[i_batch]
                    this_i_theta = this_ind_batch[rank * minibatch_size, 0]
                    this_ind_rank = this_ind_batch[rank * minibatch_size:(rank + 1) * minibatch_size, 1]

                    this_prj_batch = []
                    for i_pos in this_ind_rank:
                        line_st, line_end, px_st, px_end = probe_pos[i_pos]
                        line_st_0 = max([0, line_st])
                        line_end_0 = min([dim_y, line_end])
                        px_st_0 = max([0, px_st])
                        px_end_0 = min([dim_x, px_end])
                        patch = prj[this_i_theta, ::ds_level, ::ds_level][line_st_0:line_end_0, px_st_0:px_end_0]
                        if line_st < 0:
                            patch = np.pad(patch, [[-line_st, 0], [0, 0]], mode='constant')
                        if line_end > dim_y:
                            patch = np.pad(patch, [[0, line_end - dim_y], [0, 0]], mode='constant')
                        if px_st < 0:
                            patch = np.pad(patch, [[0, 0], [-px_st, 0]], mode='constant')
                        if px_end > dim_x:
                            patch = np.pad(patch, [[0, 0], [0, px_end - dim_x]], mode='constant')
                        this_prj_batch.append(patch)
                    this_prj_batch = np.array(this_prj_batch)
                    this_pos_batch = probe_pos[this_ind_rank]
                    this_pos_batch_safe = this_pos_batch + np.array([-safe_zone_width, safe_zone_width, -safe_zone_width, safe_zone_width])
                    # if ds_level > 1:
                    #     this_prj_batch = this_prj_batch[:, :, ::ds_level, ::ds_level]
                    comm.Barrier()

                    # Get values for local chunks of object_delta and beta; interpolate and read directly from HDF5
                    obj = get_rotated_subblocks(dset, this_pos_batch_safe, coord_ls[this_i_theta], None)
                    obj_delta = np.array(obj[:, :, :, :, 0])
                    obj_beta = np.array(obj[:, :, :, :, 1])
                    m = get_rotated_subblocks(dset_m, this_pos_batch, coord_ls[this_i_theta], None)
                    m = np.array([m[:, :, :, :, 0], m[:, :, :, :, 1]])
                    m_0 = np.copy(m)
                    v = get_rotated_subblocks(dset_v, this_pos_batch, coord_ls[this_i_theta], None)
                    v = np.array([v[:, :, :, :, 0], v[:, :, :, :, 1]])
                    v_0 = np.copy(v)
                    mask = get_rotated_subblocks(dset_mask, this_pos_batch, coord_ls[this_i_theta], None, monochannel=True)

                    mask_0 = np.copy(mask)

                # Update weight for reweighted L1
                if i_batch % 10 == 0 and i_epoch >= 1:
                    weight_l1 = np.max(obj_delta) / (abs(obj_delta) + 1e-8)
                else:
                    weight_l1 = np.ones_like(obj_delta)

                grads = loss_grad(obj_delta, obj_beta, this_ind_batch, this_prj_batch)
                if not shared_file_object:
                    this_grads = np.array(grads)
                    grads = np.zeros_like(this_grads)
                    comm.Allreduce(this_grads, grads)
                # grads = comm.allreduce(this_grads)
                grads = np.array(grads)
                grads = grads / n_ranks

                if shared_file_object:
                    grads = grads[:, :, safe_zone_width:safe_zone_width+block_size, safe_zone_width:safe_zone_width+block_size, :]
                    obj_delta = obj_delta[:, safe_zone_width:obj_delta.shape[1] - safe_zone_width,
                                             safe_zone_width:obj_delta.shape[2] - safe_zone_width, :]
                    obj_beta = obj_beta[:, safe_zone_width:obj_beta.shape[1] - safe_zone_width,
                                           safe_zone_width:obj_beta.shape[2] - safe_zone_width, :]

                (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                  grads, i_batch, m, v, step_size=learning_rate)

                # finite support
                obj_delta = obj_delta * mask
                obj_beta = obj_beta * mask
                obj_delta = np.clip(obj_delta, 0, None)
                obj_beta = np.clip(obj_beta, 0, None)

                # shrink wrap

                if shrink_cycle is not None:
                    if i_batch % shrink_cycle == 0 and i_batch > 0:
                        boolean = obj_delta > 1e-12; boolean = boolean.astype('float')
                        if not shared_file_object:
                            mask = mask * boolean.astype('float')
                        if shared_file_object:
                            write_subblocks_to_file(dset_mask, this_pos_batch, boolean, None, coord_ls[this_i_theta],
                                                    probe_size_half, mask=True)

                if shared_file_object:
                    obj = obj[:, safe_zone_width:obj.shape[1] - safe_zone_width,
                                 safe_zone_width:obj.shape[2] - safe_zone_width, :, :]
                    obj_delta = obj_delta - obj[:, :, :, :, 0]
                    obj_beta = obj_beta - obj[:, :, :, :, 1]
                    obj_delta = obj_delta / n_ranks
                    obj_beta = obj_beta / n_ranks
                    write_subblocks_to_file(dset, this_pos_batch, obj_delta, obj_beta, coord_ls[this_i_theta],
                                            probe_size_half)
                    m = m - m_0
                    m /= n_ranks
                    write_subblocks_to_file(dset_m, this_pos_batch, m[0], m[1], coord_ls[this_i_theta],
                                            probe_size_half)
                    v = v - v_0
                    v /= n_ranks
                    write_subblocks_to_file(dset_v, this_pos_batch, v[0], v[1], coord_ls[this_i_theta],
                                            probe_size_half)

                if rank == 0:
                    if shared_file_object:
                        # dxchange.write_tiff(dset[:, :, :, 0],
                        #                     fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                        #                     dtype='float32', overwrite=True)
                        dxchange.write_tiff(dset[:, :, :, 0],
                                            fname=os.path.join(output_folder, 'current/delta_{}'.format(i_batch)),
                                            dtype='float32', overwrite=True)
                    else:
                        dxchange.write_tiff(obj_delta,
                                            fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                            dtype='float32', overwrite=True)

                print_flush('Minibatch done in {} s (rank {})'.format(time.time() - t00, rank))

                f_conv.write('{}\n'.format(time.time() - t_zero))
                f_conv.flush()

            if n_epochs == 'auto':
                pass
            else:
                if i_epoch == n_epochs - 1: cont = False
            i_epoch = i_epoch + 1

            # print_flush(
            #     'Epoch {} (rank {}); loss = {}; Delta-t = {} s; current time = {}.'.format(i_epoch, rank,
            #                                                         calculate_loss(obj_delta, obj_beta, this_ind_batch,
            #                                                                        this_prj_batch),
            #                                                         time.time() - t0, time.time() - t_zero))
            if rank == 0:
                if shared_file_object:
                    dxchange.write_tiff(dset[:, :, :, 0],
                                        fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(dset[:, :, :, 1],
                                        fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                else:
                    dxchange.write_tiff(obj_delta, fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                                        dtype='float32', overwrite=True)
                    dxchange.write_tiff(obj_beta, fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)), dtype='float32',
                                        overwrite=True)

        print_flush('Current iteration finished.', 0, rank)
