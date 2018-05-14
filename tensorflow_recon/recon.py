from tensorflow.contrib.image import rotate as tf_rotate
from tensorflow.python.client import timeline
import tensorflow as tf
import dxchange
import time
import os
import h5py
import warnings
from util import *
try:
    import horovod.tensorflow as hvd
except:
    from pseudo import hvd
    warnings.warn('Unable to import Horovod.')


PI = 3.1415927


def reconstruct_diff(fname, theta_st=0, theta_end=PI, n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200, alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-6, learning_rate=1.0,
                     output_folder=None, downsample=None, minibatch_size=None, save_intermediate=False,
                     energy_ev=5000, psize_cm=1e-7, n_epochs_mask_release=None, cpu_only=False, save_path='.',
                     phantom_path='phantom', shrink_cycle=20, free_prop_cm=None):

    # TODO: rewrite minibatching to ensure going through the entire dataset

    def rotate_and_project(i, loss, obj):

        # obj_rot = apply_rotation(obj, coord_ls[rand_proj], 'arrsize_64_64_64_ntheta_500')
        obj_rot = tf_rotate(obj, this_theta_batch[i], interpolation='BILINEAR')
        if not cpu_only:
            with tf.device('/gpu:0'):
                exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], energy_ev, psize_cm, h=h, free_prop_cm=free_prop_cm)
        else:
            exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], energy_ev, psize_cm, h=h, free_prop_cm=free_prop_cm)
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

    hvd.init()
    global_step = tf.Variable(0, trainable=False, name='global_step')

    t0 = time.time()

    # read data
    print('Reading data...')
    f = h5py.File(os.path.join(save_path, fname), 'r')
    prj = f['exchange/data'][...].astype('complex64')
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))
    sys.stdout.flush()

    dim_y, dim_x = prj.shape[-2:]
    n_theta = prj.shape[0]
    theta = -np.linspace(theta_st, theta_end, n_theta, dtype='float32')
    prj_dataset = tf.data.Dataset.from_tensor_slices((theta, prj)).shard(hvd.size(), hvd.rank()).shuffle(
        buffer_size=100).repeat().batch(minibatch_size)
    prj_iter = prj_dataset.make_one_shot_iterator()
    this_theta_batch, this_prj_batch = prj_iter.get_next()

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
                        'cpu_{}'\
            .format(minibatch_size, n_epochs_mask_release, shrink_cycle,
                    n_epochs, alpha_d, alpha_b,
                    gamma, learning_rate, energy_ev,
                    dim_x, n_theta, free_prop_cm,
                    cpu_only)
        # output_folder = 'rot_bi_nn_360_stoch_{}_mskrl_{}_iter_{}_alphad_{}_alphab_{}_rate{}_ds_{}_{}_{}'.format(minibatch_size, n_epochs_mask_release, n_epochs, alpha_d, alpha_b, learning_rate, *downsample)
        # output_folder = 'rot_bi_bl_180_stoch_{}_mskrl_{}_iter_{}_alphad_{}_alphab_{}_rate{}_ds_{}_{}_{}'.format(minibatch_size, n_epochs_mask_release, n_epochs, alpha_d, alpha_b, learning_rate, *downsample)
        if abs(PI - theta_end) < 1e-3:
            output_folder += '_180'

    if save_path != '.':
        output_folder = os.path.join(save_path, output_folder)

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

    # initialize
    # 2 channels are for real and imaginary parts respectively

    # ====================================================
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
    obj_init = np.zeros([dim_y, dim_x, dim_x, 2])
    obj_init[:, :, :, 0] = np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_delta.mean(), scale=grid_delta.mean() * 0.5)
    obj_init[:, :, :, 1] = np.random.normal(size=[dim_y, dim_y, dim_x], loc=grid_beta.mean(), scale=grid_beta.mean() * 0.5)
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)
    # ====================================================

    # =============== finite support mask ==============
    try:
        mask = dxchange.read_tiff_stack(os.path.join(save_path, 'fin_sup_mask', 'mask_00000.tiff'), range(dim_y), 5)
    except:
        obj_pr = dxchange.read_tiff_stack(os.path.join(save_path, 'paganin_obj/recon_00000.tiff'), range(dim_y), 5)
        obj_pr = gaussian_filter(np.abs(obj_pr), sigma=3, mode='constant')
        mask = np.zeros_like(obj_pr)
        mask[obj_pr > 1e-5] = 1
        dxchange.write_tiff_stack(mask, os.path.join(save_path, 'fin_sup_mask/mask'), dtype='float32', overwrite=True)
    mask_add = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], 2])
    mask_add[:, :, :, 0] = mask
    mask_add[:, :, :, 1] = mask
    mask_add = tf.convert_to_tensor(mask_add, dtype=tf.float32)
    # ==================================================

    loss = tf.constant(0.0)

    # generate Fresnel kernel
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    lmbda_nm = 1240. / energy_ev
    delta_nm = voxel_nm[-1]
    kernel = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_delta.shape)
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
            reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1) + gamma * total_variation_3d(obj[:, :, :, 0])
        # reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1)

    loss = loss + reg_term
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularizer', reg_term)
    tf.summary.scalar('error', loss - reg_term)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)
    optimizer = optimizer.minimize(loss, global_step=global_step)
    # hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    loss_ls = []
    reg_ls = []

    merged_summary_op = tf.summary.merge_all()

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

    print('Optimizer started.')
    sys.stdout.flush()

    n_loop = n_epochs if n_epochs != 'auto' else max_nepochs
    n_batch = int(np.ceil(float(n_theta) / minibatch_size) / hvd.size())
    t00 = time.time()
    for epoch in range(n_loop):
        if minibatch_size < n_theta:
            for i_batch in range(n_batch):
                try:
                    t0_batch = time.time()
                    _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op])
                    print('Minibatch done in {} s (rank {}); current loss = {}.'.format(time.time() - t0_batch, hvd.rank(), current_loss))
                    sys.stdout.flush()
                except tf.errors.OutOfRangeError:
                    break
        else:
            _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op])

        # =============non negative hard================
        obj = tf.nn.relu(obj)
        # ==============================================
        if n_epochs == 'auto':
            if len(loss_ls) > 0:
                print('Reduction rate of loss is {}.'.format((current_loss - loss_ls[-1]) / loss_ls[-1]))
                sys.stdout.flush()
            if len(loss_ls) > 0 and -crit_conv_rate < (current_loss - loss_ls[-1]) / loss_ls[-1] < 0:
                loss_ls.append(current_loss)
                reg_ls.append(current_reg)
                summary_writer.add_summary(summary_str, epoch)
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
                boolean = tf.convert_to_tensor(boolean)
                mask_add = mask_add * tf.cast(boolean, tf.float32)
                dxchange.write_tiff_stack(sess.run(mask_add[:, :, :, 0]),
                                          os.path.join(save_path, 'fin_sup_mask/epoch_{}/mask'.format(epoch)), dtype='float32', overwrite=True)
            # ==============================================
        loss_ls.append(current_loss)
        reg_ls.append(current_reg)
        summary_writer.add_summary(summary_str, epoch)
        if save_intermediate:
            temp_obj = sess.run(obj)
            temp_obj = np.abs(temp_obj)
            dxchange.write_tiff(temp_obj[26, :, :, 0],
                                fname=os.path.join(output_folder, 'intermediate', 'iter_{:03d}'.format(epoch)),
                                dtype='float32',
                                overwrite=True)
            dxchange.write_tiff(temp_obj[:, :, :, 0], os.path.join(output_folder, 'current', 'delta'), dtype='float32', overwrite=True)
        print('Iteration {} (rank {}); loss = {}; time = {} s'.format(epoch, hvd.rank(), current_loss, time.time() - t00))
        sys.stdout.flush()

    print('Total time: {}'.format(time.time() - t0))
    sys.stdout.flush()

    res = sess.run(obj)
    dxchange.write_tiff(res[:, :, :, 0], fname=os.path.join(output_folder, 'delta'), dtype='float32', overwrite=True)
    dxchange.write_tiff(res[:, :, :, 1], fname=os.path.join(output_folder, 'beta'), dtype='float32', overwrite=True)

    error_ls = np.array(loss_ls) - np.array(reg_ls)

    n_epochs = len(loss_ls)
    plt.figure()
    plt.semilogy(range(n_epochs), loss_ls, label='Total loss')
    plt.semilogy(range(n_epochs), reg_ls, label='Regularizer')
    plt.semilogy(range(n_epochs), error_ls, label='Error term')
    plt.legend()
    try:
        os.makedirs(os.path.join(output_folder, 'convergence'))
    except:
        pass
    plt.savefig(os.path.join(output_folder, 'convergence', 'converge.png'), format='png')
    np.save(os.path.join(output_folder, 'convergence', 'total_loss'), loss_ls)
    np.save(os.path.join(output_folder, 'convergence', 'reg'), reg_ls)
    np.save(os.path.join(output_folder, 'convergence', 'error'), error_ls)


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
