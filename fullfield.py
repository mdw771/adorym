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
import matplotlib.pyplot as plt

plt.switch_backend('agg')

PI = 3.1415927


def reconstruct_fullfield(fname, theta_st=0, theta_end=PI, n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                          alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                          output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                          energy_ev=5000, psize_cm=1e-7, n_epochs_mask_release=None, cpu_only=False, save_path='.',
                          phantom_path='phantom', shrink_cycle=20, core_parallelization=True, free_prop_cm=None,
                          multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=5,
                          dynamic_rate=True, probe_type='plane', probe_initial=None, probe_learning_rate=1e-3,
                          pupil_function=None, theta_downsample=None, forward_algorithm='fresnel', random_theta=True,
                          object_type='normal', fresnel_approx=True, **kwargs):
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
    :param phantom_path: The location of phantom objects (for test version only).
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

        obj_stack = np.stack([obj_delta, obj_beta], axis=3)
        obj_rot_batch = []
        for i in range(minibatch_size):
            obj_rot_batch.append(apply_rotation(obj_stack, coord_ls[this_ind_batch[i]],
                                                'arrsize_{}_{}_{}_ntheta_{}'.format(dim_y, dim_x, dim_x, n_theta)))
        obj_rot_batch = np.stack(obj_rot_batch)

        exiting_batch = multislice_propagate_batch_numpy(obj_rot_batch[:, :, :, :, 0], obj_rot_batch[:, :, :, :, 1],
                                                         probe_real, probe_imag, energy_ev,
                                                         psize_cm * ds_level, free_prop_cm=free_prop_cm,
                                                         obj_batch_shape=[minibatch_size, *obj_size],
                                                         kernel=h, fresnel_approx=fresnel_approx)
        loss = np.mean((np.abs(exiting_batch) - np.abs(this_prj_batch)) ** 2)

        if alpha_d is None:
            reg_term = alpha * (np.sum(np.abs(obj_delta)) + np.sum(np.abs(obj_delta))) + gamma * total_variation_3d(
                obj_delta)
        else:
            if gamma == 0:
                reg_term = alpha_d * np.sum(np.abs(obj_delta)) + alpha_b * np.sum(np.abs(obj_beta))
            else:
                reg_term = alpha_d * np.sum(np.abs(obj_delta)) + alpha_b * np.sum(
                    np.abs(obj_beta)) + gamma * total_variation_3d(obj_delta)
        loss = loss + reg_term

        return loss

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    t_zero = time.time()

    # read data
    t0 = time.time()
    print_flush('Reading data...', 0, rank)
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj_0 = f['exchange/data'][...].astype('complex64')
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

    initializer_flag = False

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

        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level), 0, rank)
        comm.Barrier()

        # downsample data
        prj = np.copy(prj_0)
        if ds_level > 1:
            prj = prj[:, ::ds_level, ::ds_level]
            prj = prj.astype('complex64')
        comm.Barrier()

        ind_ls = np.arange(n_theta)
        np.random.shuffle(ind_ls)
        n_tot_per_batch = size * minibatch_size
        if n_theta % n_tot_per_batch > 0:
            ind_ls = np.concatenate(ind_ls, ind_ls[:n_tot_per_batch - n_theta % n_tot_per_batch])
        ind_ls = split_tasks(ind_ls, n_tot_per_batch)
        ind_ls = [np.sort(x) for x in ind_ls]

        dim_y, dim_x = prj.shape[-2:]
        comm.Barrier()

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

        try:
            mask = dxchange.read_tiff_stack(os.path.join(save_path, 'fin_sup_mask', 'mask_00000.tiff'),
                                            range(prj_0.shape[1]), 5)
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
        dim_z = mask.shape[-1]

        # unify random seed for all threads
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        if initializer_flag == False:
            if initial_guess is None:
                print_flush('Initializing with Gaussian random.', 0, rank)
                obj_delta = np.random.normal(size=[dim_y, dim_x, dim_z], loc=8.7e-7, scale=1e-7) * mask
                obj_beta = np.random.normal(size=[dim_y, dim_x, dim_z], loc=5.1e-8, scale=1e-8) * mask
                obj_delta[obj_delta < 0] = 0
                obj_beta[obj_beta < 0] = 0
            else:
                print_flush('Using supplied initial guess.', 0, rank)
                sys.stdout.flush()
                obj_delta = initial_guess[0]
                obj_beta = initial_guess[1]
        else:
            print_flush('Initializing with Gaussian random.', 0, rank)
            obj_delta = dxchange.read_tiff(os.path.join(output_folder, 'delta_ds_{}.tiff'.format(ds_level * 2)))
            obj_beta = dxchange.read_tiff(os.path.join(output_folder, 'beta_ds_{}.tiff'.format(ds_level * 2)))
            obj_delta = upsample_2x(obj_delta)
            obj_beta = upsample_2x(obj_beta)
            obj_delta += np.random.normal(size=[dim_y, dim_x, dim_z], loc=8.7e-7, scale=1e-7) * mask
            obj_beta += np.random.normal(size=[dim_y, dim_x, dim_z], loc=5.1e-8, scale=1e-8) * mask
            obj_delta[obj_delta < 0] = 0
            obj_beta[obj_beta < 0] = 0
        obj_size = obj_delta.shape
        if object_type == 'phase_only':
            obj_beta[...] = 0
        elif object_type == 'absorption_only':
            obj_delta[...] = 0
        # ====================================================

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

        # =============finite support===================
        obj_delta = obj_delta * mask
        obj_beta = obj_beta * mask
        obj_delta = np.clip(obj_delta, 0, None)
        obj_beta = np.clip(obj_beta, 0, None)
        # ==============================================

        # generate Fresnel kernel
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        h = get_kernel(delta_nm, lmbda_nm, voxel_nm, [dim_y, dim_y, dim_x], fresnel_approx=fresnel_approx)

        loss_grad = grad(calculate_loss, [0, 1])

        print_flush('Optimizer started.', 0, rank)
        if rank == 0:
            create_summary(output_folder, locals(), preset='fullfield')

        cont = True
        i_epoch = 0
        while cont:
            m, v = (None, None)
            t0 = time.time()
            for i_batch in range(len(ind_ls)):

                t00 = time.time()
                this_ind_batch = ind_ls[i_batch][rank * minibatch_size:(rank + 1) * minibatch_size]
                this_prj_batch = prj[this_ind_batch]
                grads = loss_grad(obj_delta, obj_beta, this_ind_batch, this_prj_batch)
                this_grads = np.array(grads)
                grads = np.zeros_like(this_grads)
                comm.Allreduce(this_grads, grads)
                grads = grads / size
                (obj_delta, obj_beta), m, v = apply_gradient_adam(np.array([obj_delta, obj_beta]),
                                                                  grads, i_batch, m, v, step_size=learning_rate)

                dxchange.write_tiff(obj_delta,
                                    fname=os.path.join(output_folder, 'intermediate', 'current'.format(ds_level)),
                                    dtype='float32', overwrite=True)
                # finite support
                obj_delta = obj_delta * mask
                obj_beta = obj_beta * mask
                obj_delta = np.clip(obj_delta, 0, None)
                obj_beta = np.clip(obj_beta, 0, None)

                # shrink wrap
                if shrink_cycle is not None:
                    if i_epoch >= shrink_cycle:
                        boolean = obj_delta > 1e-15
                    mask = mask * boolean

                print_flush('Minibatch done in {} s (rank {})'.format(time.time() - t00, rank))

            if n_epochs == 'auto':
                pass
            else:
                if i_epoch == n_epochs - 1: cont = False
            i_epoch = i_epoch + 1

            print_flush(
                'Epoch {} (rank {}); loss = {}; Delta-t = {} s; current time = {}.'.format(i_epoch, rank,
                                                                    calculate_loss(obj_delta, obj_beta, this_ind_batch,
                                                                                   this_prj_batch),
                                                                    time.time() - t0, time.time() - t_zero))
        dxchange.write_tiff(obj_delta, fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)),
                            dtype='float32', overwrite=True)
        dxchange.write_tiff(obj_beta, fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)), dtype='float32',
                            overwrite=True)

        print_flush('Current iteration finished.', 0, rank)
