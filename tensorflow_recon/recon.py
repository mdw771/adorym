from tensorflow.contrib.image import rotate as tf_rotate
from tensorflow.python.client import timeline
import tensorflow as tf
import dxchange
import time
import os
import h5py
import warnings
from util import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')


PI = 3.1415927


def reconstruct_diff(fname, theta_st=0, theta_end=PI, n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200,
                     alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                     output_folder=None, minibatch_size=None, save_intermediate=False, full_intermediate=False,
                     energy_ev=5000, psize_cm=1e-7, n_epochs_mask_release=None, cpu_only=False, save_path='.',
                     phantom_path='phantom', shrink_cycle=20, core_parallelization=True, free_prop_cm=None,
                     multiscale_level=1, n_epoch_final_pass=None, initial_guess=None, n_batch_per_update=5,
                     dynamic_rate=True, wavefront_type='plane', wavefront_initial=None, probe_learning_rate=1e-3,
                     pupil_function=None):
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
    :param wavefront_type: type of wavefront. Can be 'plane', 'fixed', or 'optimizable'. If 'optimizable',
                           the probe function will be optimized along with the object.
    :param wavefront_initial: can be provided for 'optimizable' wavefront_type, and must be provided for
                              'fixed'.
    """

    # TODO: rewrite minibatching to ensure going through the entire dataset

    def rotate_and_project(i, loss, obj):

        # obj_rot = apply_rotation(obj, coord_ls[rand_proj], 'arrsize_64_64_64_ntheta_500')
        obj_rot = tf_rotate(obj, this_theta_batch[i], interpolation='BILINEAR')
        if not cpu_only:
            with tf.device('/gpu:0'):
                exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], probe_real, probe_imag, energy_ev, psize_cm * ds_level, h=h, free_prop_cm=free_prop_cm)
        else:
            exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], probe_real, probe_imag, energy_ev, psize_cm * ds_level, h=h, free_prop_cm=free_prop_cm)
        loss += tf.reduce_mean(tf.squared_difference(tf.abs(exiting), tf.abs(this_prj_batch[i])))
        i = tf.add(i, 1)
        return (i, loss, obj)

    def rotate_and_project_batch(loss, obj):

        obj_rot_batch = []
        for i in range(minibatch_size):
            obj_rot_batch.append(tf_rotate(obj, this_theta_batch[i], interpolation='BILINEAR'))
        # obj_rot = apply_rotation(obj, coord_ls[rand_proj], 'arrsize_64_64_64_ntheta_500')
        obj_rot_batch = tf.stack(obj_rot_batch)
        exiting_batch = multislice_propagate_batch(obj_rot_batch[:, :, :, :, 0], obj_rot_batch[:, :, :, :, 1], energy_ev, psize_cm)
        loss += tf.reduce_mean(tf.squared_difference(tf.abs(exiting_batch), tf.abs(this_prj_batch)))
        return loss

    # import Horovod or its fake shell
    if core_parallelization is False:
        warnings.warn('Parallelization is disabled in the reconstruction routine. ')
        from pseudo import hvd
    else:
        try:
            import horovod.tensorflow as hvd
            hvd.init()
        except:
            from pseudo import Hvd
            hvd = Hvd()
            warnings.warn('Unable to import Horovod.')
        try:
            assert hvd.mpi_threads_supported()
        except:
            warnings.warn('MPI multithreading is not supported.')
        try:
            import mpi4py.rc
            mpi4py.rc.initialize = False
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            mpi4py_is_ok = True
            assert hvd.size() == comm.Get_size()
        except:
            warnings.warn('Unable to import mpi4py. Using multiple threads with n_epoch set to "auto" may lead to undefined behaviors.')
            from pseudo import Mpi
            comm = Mpi()
            mpi4py_is_ok = False

    # global_step = tf.Variable(0, trainable=False, name='global_step')

    t0 = time.time()

    # read data
    print_flush('Reading data...')
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj_0 = f['exchange/data'][...].astype('complex64')
    original_shape = prj_0.shape
    print_flush('Data reading: {} s'.format(time.time() - t0))
    print_flush('Data shape: {}'.format(original_shape))
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

        graph = tf.Graph()
        graph.as_default()

        ds_level = 2 ** ds_level
        print_flush('Multiscale downsampling level: {}'.format(ds_level))
        comm.Barrier()

        # downsample data
        prj = prj_0
        if ds_level > 1:
            prj = prj[::ds_level, ::ds_level, ::ds_level]
            prj = prj.astype('complex64')
        comm.Barrier()

        dim_y, dim_x = prj.shape[-2:]
        n_theta = prj.shape[0]
        theta = -np.linspace(theta_st, theta_end, n_theta, dtype='float32')
        comm.Barrier()
        prj_dataset = tf.data.Dataset.from_tensor_slices((theta, prj)).shard(hvd.size(), hvd.rank()).shuffle(
            buffer_size=100).repeat().batch(minibatch_size)
        prj_iter = prj_dataset.make_one_shot_iterator()
        this_theta_batch, this_prj_batch = prj_iter.get_next()
        comm.Barrier()

        # # read rotation data
        # try:
        #     coord_ls = read_all_origin_coords('arrsize_64_64_64_ntheta_500', n_theta)
        # except:
        #     save_rotation_lookup([dim_y, dim_x, dim_x], n_theta)
        #     coord_ls = read_all_origin_coords('arrsize_64_64_64_ntheta_500', n_theta)

        if minibatch_size is None:
            minibatch_size = n_theta

        if n_epochs_mask_release is None:
            n_epochs_mask_release = np.inf

        # =============== finite support mask ==============
        try:
            mask = dxchange.read_tiff_stack(os.path.join(save_path, 'fin_sup_mask', 'mask_00000.tiff'), range(prj_0.shape[1]), 5)
        except:
            obj_pr = dxchange.read_tiff_stack(os.path.join(save_path, 'paganin_obj/recon_00000.tiff'), range(prj_0.shape[1]), 5)
            obj_pr = gaussian_filter(np.abs(obj_pr), sigma=3, mode='constant')
            mask = np.zeros_like(obj_pr)
            mask[obj_pr > 1e-5] = 1
            dxchange.write_tiff_stack(mask, os.path.join(save_path, 'fin_sup_mask/mask'), dtype='float32', overwrite=True)
        if ds_level > 1:
            mask = mask[::ds_level, ::ds_level, ::ds_level]
        mask_add = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], 2])
        mask_add[:, :, :, 0] = mask
        mask_add[:, :, :, 1] = mask
        mask_add = tf.convert_to_tensor(mask_add, dtype=tf.float32)

        # unify random seed for all threads
        comm.Barrier()
        seed = int(time.time() / 60)
        np.random.seed(seed)
        comm.Barrier()

        # initializer_flag = True
        if initializer_flag == False:
            if initial_guess is None:
                print_flush('Initializing with Gaussian random.')
                grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
                grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
                obj_init = np.zeros([dim_y, dim_x, dim_x, 2])
                obj_init[:, :, :, 0] = np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_delta.mean(), scale=grid_delta.mean() * 0.5) * mask
                obj_init[:, :, :, 1] = np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_beta.mean(), scale=grid_beta.mean() * 0.5) * mask
                obj_init[obj_init < 0] = 0
            else:
                print_flush('Using supplied initial guess.')
                sys.stdout.flush()
                obj_init = np.zeros([dim_y, dim_x, dim_x, 2])
                obj_init[:, :, :, 0] = initial_guess[0]
                obj_init[:, :, :, 1] = initial_guess[1]
        else:
            print_flush('Initializing with Gaussian random.')
            grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
            grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
            delta_init = dxchange.read_tiff(os.path.join(output_folder, 'delta_ds_{}.tiff'.format(ds_level * 2)))
            beta_init = dxchange.read_tiff(os.path.join(output_folder, 'beta_ds_{}.tiff'.format(ds_level * 2)))
            obj_init = np.zeros([delta_init.shape[0], delta_init.shape[1], delta_init.shape[2], 2])
            obj_init[:, :, :, 0] = delta_init
            obj_init[:, :, :, 1] = beta_init
            # obj_init = res
            obj_init = upsample_2x(obj_init)
            obj_init[:, :, :, 0] += np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_delta.mean(), scale=grid_delta.mean() * 0.5) * mask
            obj_init[:, :, :, 1] += np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_beta.mean(), scale=grid_beta.mean() * 0.5) * mask
            obj_init[obj_init < 0] = 0
        # dxchange.write_tiff(obj_init[:, :, :, 0], 'cone_256_filled/dump/obj_init', dtype='float32')
        obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)
        # ====================================================

        if wavefront_type == 'plane':
            probe_real = tf.constant(np.ones([dim_y, dim_x]), dtype=tf.float32)
            probe_imag = tf.constant(np.zeros([dim_y, dim_x]), dtype=tf.float32)
        elif wavefront_type == 'optimizable':
            if wavefront_initial is not None:
                probe_mag, probe_phase = wavefront_initial
                probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
            else:
                # probe_mag = np.ones([dim_y, dim_x])
                # probe_phase = np.zeros([dim_y, dim_x])
                back_prop_cm = (free_prop_cm + (psize_cm * obj_init.shape[2])) if free_prop_cm is not None else (psize_cm * obj_init.shape[2])
                probe_init = create_probe_initial_guess(os.path.join(save_path, fname), back_prop_cm * 1.e7, energy_ev, psize_cm * 1.e7)
                probe_real = probe_init.real
                probe_imag = probe_init.imag
            if pupil_function is not None:
                probe_real = probe_real * pupil_function
                probe_imag = probe_imag * pupil_function
                pupil_function = tf.convert_to_tensor(pupil_function, dtype=tf.float32)
            probe_real = tf.Variable(probe_real, dtype=tf.float32, trainable=True)
            probe_imag = tf.Variable(probe_imag, dtype=tf.float32, trainable=True)
        elif wavefront_type == 'fixed':
            probe_mag, probe_phase = wavefront_initial
            probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
            probe_real = tf.constant(probe_real, dtype=tf.float32)
            probe_imag = tf.constant(probe_imag, dtype=tf.float32)
        else:
            raise ValueError('Invalid wavefront type. Choose from \'plane\', \'fixed\', \'optimizable\'.')

        loss = tf.constant(0.0)

        # generate Fresnel kernel
        voxel_nm = np.array([psize_cm] * 3) * 1.e7 * ds_level
        lmbda_nm = 1240. / energy_ev
        delta_nm = voxel_nm[-1]
        kernel = get_kernel(delta_nm, lmbda_nm, voxel_nm, [dim_y, dim_y, dim_x])
        h = tf.convert_to_tensor(kernel, dtype=tf.complex64, name='kernel')

        if cpu_only:
            i = tf.constant(0)
            # c = lambda i, loss, obj: tf.less(i, minibatch_size)
            # _, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])
            for j in range(minibatch_size):
                i, loss, obj = rotate_and_project(i, loss, obj)
        else:
            loss = rotate_and_project_batch(loss, obj)

        # loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))
        # loss = loss / n_theta + gamma * energy_leak(obj, mask_add)
        if alpha_d is None:
            reg_term = alpha * tf.norm(obj, ord=1) + gamma * tf.image.total_variation(obj[:, :, :, 0])
        else:
            if gamma == 0:
                reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1)
            else:
                # reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1) + gamma * total_variation_3d(obj[:, :, :, 0:1])
                reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1) + gamma * tf.norm(obj[:, :, :, 0], ord=2)
            # reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1)

        loss = loss + reg_term
        if wavefront_type == 'optimizable':
            probe_reg = 1.e-10 * (tf.image.total_variation(tf.reshape(probe_real, [dim_y, dim_x, -1])) +
                                   tf.image.total_variation(tf.reshape(probe_real, [dim_y, dim_x, -1])))
            loss = loss + probe_reg
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('regularizer', reg_term)
        tf.summary.scalar('error', loss - reg_term)

        # if initializer_flag == False:
        i_epoch = tf.Variable(0, trainable=False, dtype='float32')
        accum_grad = tf.Variable(tf.zeros_like(obj.initialized_value()), trainable=False)
        if dynamic_rate and n_batch_per_update > 1:
            # modifier =  1. / n_batch_per_update
            modifier = tf.exp(-i_epoch) * (n_batch_per_update - 1) + 1
            optimizer = tf.train.AdamOptimizer(learning_rate=float(learning_rate) * hvd.size() * modifier)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size())
        optimizer = hvd.DistributedOptimizer(optimizer, name='distopt_{}'.format(ds_level))
        if n_batch_per_update > 1:
            this_grad = optimizer.compute_gradients(loss, obj)
            this_grad = this_grad[0]
            initialize_grad = accum_grad.assign(tf.zeros_like(accum_grad))
            accum_op = accum_grad.assign_add(this_grad[0])
            update_obj = optimizer.apply_gradients([(accum_grad / n_batch_per_update, this_grad[1])])
        else:
            optimizer = optimizer.minimize(loss, var_list=[obj])
        if minibatch_size >= n_theta:
            optimizer = optimizer.minimize(loss, var_list=[obj])
        # hooks = [hvd.BroadcastGlobalVariablesHook(0)]

        if wavefront_type == 'optimizable':
            optimizer_probe = tf.train.AdamOptimizer(learning_rate=probe_learning_rate * hvd.size())
            optimizer_probe = hvd.DistributedOptimizer(optimizer_probe, name='distopt_probe_{}'.format(ds_level))
            if n_batch_per_update > 1:
                accum_grad_probe = [tf.Variable(tf.zeros_like(probe_real.initialized_value()), trainable=False),
                                    tf.Variable(tf.zeros_like(probe_imag.initialized_value()), trainable=False)]
                this_grad_probe = optimizer_probe.compute_gradients(loss, [probe_real, probe_imag])
                initialize_grad_probe = [accum_grad_probe[i].assign(tf.zeros_like(accum_grad_probe[i])) for i in range(2)]
                accum_op_probe = [accum_grad_probe[i].assign_add(this_grad_probe[i][0]) for i in range(2)]
                update_probe = [optimizer_probe.apply_gradients([(accum_grad_probe[i] / n_batch_per_update, this_grad_probe[i][1])]) for i in range(2)]
            else:
                optimizer_probe = optimizer_probe.minimize(loss, var_list=[probe_real, probe_imag])
            if minibatch_size >= n_theta:
                optimizer_probe = optimizer_probe.minimize(loss, var_list=[probe_real, probe_imag])

        loss_ls = []
        reg_ls = []

        merged_summary_op = tf.summary.merge_all()

        # create benchmarking metadata
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        if cpu_only:
            sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True))
        else:
            config = tf.ConfigProto(log_device_placement=False)
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = str(hvd.local_rank())
            sess = tf.Session(config=config)

        sess.run(tf.global_variables_initializer())
        hvd.broadcast_global_variables(0)

        summary_writer = tf.summary.FileWriter(os.path.join(output_folder, 'tb'))

        t0 = time.time()

        print_flush('Optimizer started.')

        n_loop = n_epochs if n_epochs != 'auto' else max_nepochs
        if ds_level == 1 and n_epoch_final_pass is not None:
            n_loop = n_epoch_final_pass
        n_batch = int(np.ceil(float(n_theta) / minibatch_size) / hvd.size())
        t00 = time.time()
        for epoch in range(n_loop):
            if mpi4py_is_ok:
                stop_iteration = False
            else:
                stop_iteration = open('.stop_itertion', 'w')
                stop_iteration.write('False')
                stop_iteration_file.close()
            i_epoch = i_epoch + 1
            if minibatch_size < n_theta:
                batch_counter = 0
                for i_batch in range(n_batch):
                    try:
                        if n_batch_per_update > 1:
                            t0_batch = time.time()
                            if wavefront_type == 'optimizable':
                                _, _, current_loss, current_reg, current_probe_reg, summary_str = sess.run(
                                    [accum_op, accum_op_probe, loss, reg_term, probe_reg, merged_summary_op], options=run_options,
                                    run_metadata=run_metadata)
                            else:
                                _, current_loss, current_reg, summary_str = sess.run(
                                    [accum_op, loss, reg_term, merged_summary_op], options=run_options,
                                    run_metadata=run_metadata)
                            print_flush('Minibatch done in {} s (rank {}); current loss = {}, probe reg. = {}.'.format(time.time() - t0_batch, hvd.rank(), current_loss, current_probe_reg))
                            batch_counter += 1
                            if batch_counter == n_batch_per_update or i_batch == n_batch - 1:
                                sess.run(update_obj)
                                sess.run(initialize_grad)
                                if wavefront_type == 'optimizable':
                                    sess.run(update_probe)
                                    sess.run(initialize_grad_probe)
                                batch_counter = 0
                                print_flush('Gradient applied.')
                        else:
                            t0_batch = time.time()
                            if wavefront_type == 'optimizable':
                                _, _, current_loss, current_reg, current_probe_reg, summary_str = sess.run([optimizer, optimizer_probe, loss, reg_term, probe_reg, merged_summary_op], options=run_options, run_metadata=run_metadata)
                                print_flush(
                                    'Minibatch done in {} s (rank {}); current loss = {}, probe reg. = {}.'.format(
                                        time.time() - t0_batch, hvd.rank(), current_loss, current_probe_reg))

                            else:
                                _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op], options=run_options, run_metadata=run_metadata)
                                print_flush(
                                    'Minibatch done in {} s (rank {}); current loss = {}.'.format(
                                        time.time() - t0_batch, hvd.rank(), current_loss))

                        # enforce pupil function
                        if wavefront_type == 'optimizable' and pupil_function is not None:
                            probe_real = probe_real * pupil_function
                            probe_imag = probe_imag * pupil_function

                    except tf.errors.OutOfRangeError:
                        break
            else:
                if wavefront_type == 'optimizable':
                    _, _, current_loss, current_reg, summary_str = sess.run([optimizer, optimizer_probe, loss, reg_term, merged_summary_op], options=run_options, run_metadata=run_metadata)
                else:
                    _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op], options=run_options, run_metadata=run_metadata)

            # timeline for benchmarking
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            try:
                os.makedirs(os.path.join(output_folder, 'profiling'))
            except:
                pass
            with open(os.path.join(output_folder, 'profiling', 'time_{}.json'.format(epoch)), 'w') as f:
                f.write(ctf)

            # non negative hard
            obj = tf.nn.relu(obj)

            # check stopping criterion
            if n_epochs == 'auto':
                if len(loss_ls) > 0:
                    print_flush('Reduction rate of loss is {}.'.format((current_loss - loss_ls[-1]) / loss_ls[-1]))
                    sys.stdout.flush()
                if len(loss_ls) > 0 and -crit_conv_rate < (current_loss - loss_ls[-1]) / loss_ls[-1] < 0 and hvd.rank() == 0:
                    loss_ls.append(current_loss)
                    reg_ls.append(current_reg)
                    summary_writer.add_summary(summary_str, epoch)
                    if mpi4py_is_ok:
                        stop_iteration = True
                    else:
                        stop_iteration = open('.stop_iteration', 'w')
                        stop_iteration.write('True')
                        stop_iteration.close()
                comm.Barrier()
                if mpi4py_is_ok:
                    stop_iteration = comm.bcast(stop_iteration, root=0)
                else:
                    stop_iteration_file = open('.stop_iteration', 'r')
                    stop_iteration = stop_iteration_file.read()
                    stop_iteration_file.close()
                    stop_iteration = True if stop_iteration else False
                if stop_iteration:
                    break
            if epoch < n_epochs_mask_release:
                # =============finite support===================
                if n_epochs == 'auto' or epoch != n_epochs - 1:
                    obj = obj * mask_add
                # ==============================================
                # ================shrink wrap===================
                if epoch % shrink_cycle == 0 and epoch > 0:
                    mask_temp = sess.run(obj[:, :, :, 0] > 1e-8)
                    boolean = np.zeros_like(obj_init)
                    boolean[:, :, :, 0] = mask_temp
                    boolean[:, :, :, 1] = mask_temp
                    boolean = tf.convert_to_tensor(boolean, dtype=tf.float32)
                    mask_add = mask_add * boolean
                    if hvd.rank() == 0 and hvd.local_rank() == 0:
                        dxchange.write_tiff_stack(sess.run(mask_add[:, :, :, 0]),
                                                  os.path.join(save_path, 'fin_sup_mask/epoch_{}/mask'.format(epoch)), dtype='float32', overwrite=True)
                # ==============================================
            if hvd.rank() == 0:
                loss_ls.append(current_loss)
                reg_ls.append(current_reg)
                summary_writer.add_summary(summary_str, epoch)
            if save_intermediate and hvd.rank() == 0:
                temp_obj = sess.run(obj)
                temp_obj = np.abs(temp_obj)
                if full_intermediate:
                    dxchange.write_tiff(temp_obj[:, :, :, 0],
                                        fname=os.path.join(output_folder, 'intermediate', 'ds_{}_iter_{:03d}'.format(ds_level, epoch)),
                                        dtype='float32',
                                        overwrite=True)
                else:
                    dxchange.write_tiff(temp_obj[int(temp_obj.shape[0] / 2), :, :, 0],
                                        fname=os.path.join(output_folder, 'intermediate', 'ds_{}_iter_{:03d}'.format(ds_level, epoch)),
                                        dtype='float32',
                                        overwrite=True)
                    probe_current_real, probe_current_imag = sess.run([probe_real, probe_imag])
                    probe_current_mag, probe_current_phase = real_imag_to_mag_phase(probe_current_real, probe_current_imag)
                    dxchange.write_tiff(probe_current_mag,
                                        fname=os.path.join(output_folder, 'intermediate', 'probe',
                                                           'mag_ds_{}_iter_{:03d}'.format(ds_level, epoch)),
                                        dtype='float32',
                                        overwrite=True)
                    dxchange.write_tiff(probe_current_phase,
                                        fname=os.path.join(output_folder, 'intermediate', 'probe',
                                                           'phase_ds_{}_iter_{:03d}'.format(ds_level, epoch)),
                                        dtype='float32',
                                        overwrite=True)
                dxchange.write_tiff(temp_obj[:, :, :, 0], os.path.join(output_folder, 'current', 'delta'), dtype='float32', overwrite=True)
                print_flush('Iteration {} (rank {}); loss = {}; time = {} s'.format(epoch, hvd.rank(), current_loss, time.time() - t00))
            sys.stdout.flush()
            # except:
            #     # if one thread breaks out after meeting stopping criterion, intercept Horovod error and break others
            #     break

            print_flush('Total time: {}'.format(time.time() - t0))
        sys.stdout.flush()

        if hvd.rank() == 0:

            res = sess.run(obj)
            dxchange.write_tiff(res[:, :, :, 0], fname=os.path.join(output_folder, 'delta_ds_{}'.format(ds_level)), dtype='float32', overwrite=True)
            dxchange.write_tiff(res[:, :, :, 1], fname=os.path.join(output_folder, 'beta_ds_{}'.format(ds_level)), dtype='float32', overwrite=True)

            probe_final_real, probe_final_imag = sess.run([probe_real, probe_imag])
            probe_final_mag, probe_final_phase = real_imag_to_mag_phase(probe_final_real, probe_final_imag)
            dxchange.write_tiff(probe_final_mag, fname=os.path.join(output_folder, 'probe_mag_ds_{}'.format(ds_level)), dtype='float32', overwrite=True)
            dxchange.write_tiff(probe_final_phase, fname=os.path.join(output_folder, 'probe_phase_ds_{}'.format(ds_level)), dtype='float32', overwrite=True)

            error_ls = np.array(loss_ls) - np.array(reg_ls)

            x = len(loss_ls)
            plt.figure()
            plt.semilogy(range(x), loss_ls, label='Total loss')
            plt.semilogy(range(x), reg_ls, label='Regularizer')
            plt.semilogy(range(x), error_ls, label='Error term')
            plt.legend()
            try:
                os.makedirs(os.path.join(output_folder, 'convergence'))
            except:
                pass
            plt.savefig(os.path.join(output_folder, 'convergence', 'converge_ds_{}.png'.format(ds_level)), format='png')
            np.save(os.path.join(output_folder, 'convergence', 'total_loss_ds_{}'.format(ds_level)), loss_ls)
            np.save(os.path.join(output_folder, 'convergence', 'reg_ds_{}'.format(ds_level)), reg_ls)
            np.save(os.path.join(output_folder, 'convergence', 'error_ds_{}'.format(ds_level)), error_ls)

            print_flush('Clearing current graph...')
        sess.run(tf.global_variables_initializer())
        sess.close()
        tf.reset_default_graph()
        initializer_flag = True
        print_flush('Current iteration finished.')


def reconstruct_pureproj(fname, sino_range, theta_st=0, theta_end=PI, n_epochs=200, alpha=1e-4, learning_rate=1.0,
               output_folder=None, output_name='recon', downsample=None,
               save_intermediate=False, initial_guess=None, center=None):

    def rotate_and_project(i, loss, obj):

        loss += tf.reduce_mean(tf.squared_difference(
            tf.reduce_sum(tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR'), 1)[:, :, 0], prj[i]))
        i = tf.add(i, 1)
        return (i, loss, obj)

    f = open('loss.txt', 'a')

    # with tf.device('/:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

    if output_folder is None:
        output_folder = 'uni_0p5_init_{}_alpha{}_rate_{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)

    t0 = time.time()
    print('Reading data...')
    prj, flt, drk, _ = dxchange.read_aps_32id(fname, sino=sino_range)
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))
    prj = tomopy.normalize(prj, flt, drk)
    prj = preprocess(prj)
    # scale up to prevent precision issue
    prj *= 1.e2

    if center is None:
        center = prj.shape[-1] / 2

    # correct for center
    offset = int(prj.shape[-1] / 2) - center
    if offset != 0:
        for i in range(prj.shape[0]):
            prj[i, :, :] = realign_image(prj[i, :, :], [0, offset])

    if downsample is not None:
        prj = tomopy.downsample(prj, level=downsample[0], axis=0)
        prj = tomopy.downsample(prj, level=downsample[1], axis=1)
        prj = tomopy.downsample(prj, level=downsample[2], axis=2)
        if downsample[2] != 0:
            center /= (2 ** downsample[2])
        print('Downsampled shape: {}'.format(prj.shape))

    dxchange.write_tiff(prj, 'prj', dtype='float32', overwrite=True)

    dim_y, dim_x = prj.shape[-2:]
    n_theta = prj.shape[0]

    # reference recon by gridrec
    rec = tomopy.recon(prj, tomopy.angles(n_theta), algorithm='gridrec', center=int(prj.shape[-1] / 2))
    dxchange.write_tiff_stack(rec, 'ref_results/recon', dtype='float32', overwrite=True)

    # convert data
    prj = tf.convert_to_tensor(prj)

    theta = -np.linspace(theta_st, theta_end, n_theta)

    theta_ls_tensor = tf.constant(theta, dtype='float32')

    if initial_guess is None:
        obj = tf.Variable(initial_value=tf.zeros([dim_y, dim_x, dim_x, 1]))
        obj += 0.5
    else:
        init = dxchange.read_tiff(initial_guess)
        if init.ndim == 3:
            init = init[:, :, :, np.newaxis]
        else:
            init = init[np.newaxis, :, :, np.newaxis]
        obj = tf.Variable(initial_value=init)

    loss = tf.constant(0.0)

    i = tf.constant(0)
    c = lambda i, loss, obj: tf.less(i, n_theta)

    _, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])
    loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    loss_ls = []

    sess.run(tf.global_variables_initializer())

    t0 = time.time()

    print('Optimizer started.')

    # create benchmarking metadata
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for epoch in range(n_epochs):

        t00 = time.time()
        _, current_loss = sess.run([optimizer, loss], options=run_options, run_metadata=run_metadata)
        loss_ls.append(current_loss)
        if save_intermediate:
            temp_obj = sess.run(obj)

            # timeline for benchmarking
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            try:
                os.makedirs(os.path.join(output_folder, 'json'))
            except:
                pass
            with open(os.path.join(output_folder, 'json', 'time_{}.json'.format(epoch)), 'w') as f:
                f.write(ctf)

            dxchange.write_tiff(np.squeeze(temp_obj[0, :, :, 0]),
                                      fname=os.path.join(output_folder, 'intermediate', 'iter_{:03d}'.format(epoch)),
                                      dtype='float32',
                                      overwrite=True)
        # print(sess.run(tf.reduce_sum(tf.image.total_variation(obj))))
        print('Iteration {}; loss = {}; time = {} s'.format(epoch, current_loss, time.time() - t00))

    print('Total time: {}'.format(time.time() - t0))

    res = sess.run(obj)
    dxchange.write_tiff_stack(res[:, :, :, 0], fname=os.path.join(output_folder, output_name), dtype='float32', overwrite=True)

    final_loss, final_tv = sess.run([loss, tf.reduce_sum(tf.image.total_variation(obj))])
    final_tv *= alpha
    f.write('{} {} {} {}\n'.format(alpha, final_loss, (final_loss-final_tv), final_tv))
    f.close()
    np.save(os.path.join(output_folder, 'converge'), loss_ls)
