from tensorflow.contrib.image import rotate as tf_rotate
from tensorflow.python.client import timeline
import tensorflow as tf
import dxchange
import time
import os
import h5py
from util import *


PI = 3.1415927

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


def reconstruct_diff(fname, theta_st=0, theta_end=PI, n_epochs='auto', crit_conv_rate=0.03, max_nepochs=200, alpha=1e-7, alpha_d=None, alpha_b=None, gamma=1e-2, learning_rate=1.0,
                     output_folder=None, downsample=None, minibatch_size=None, save_intermediate=False,
                     energy_ev=5000, psize_cm=1e-7, n_epochs_mask_release=None, cpu_only=False):

    # TODO: rewrite minibatching to ensure going through the entire dataset

    def rotate_and_project(i, loss, obj):

        rand_proj = batch_inds[i]
        # obj_rot = apply_rotation(obj, coord_ls[rand_proj], 'arrsize_64_64_64_ntheta_500')
        obj_rot = tf_rotate(obj, theta_ls_tensor[rand_proj], interpolation='BILINEAR')
        # with tf.device('cpu:0'):
        exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], energy_ev, psize_cm)
        loss += tf.reduce_mean(tf.squared_difference(tf.abs(exiting), tf.abs(prj[rand_proj])))
        i = tf.add(i, 1)
        return (i, loss, obj)

    def energy_leak(obj, support_mask):

        leak = tf.reduce_sum(tf.pow(obj * (1 - support_mask), 2))
        non_leak = tf.reduce_sum(tf.pow(obj * support_mask, 2))
        return leak / non_leak

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    global_step = tf.Variable(0, trainable=False, name='global_step')

    if cpu_only:
        config = tf.ConfigProto(
            device_count = {'GPU': 0},
            # log_device_placement=True
        )
        sess = tf.Session(config=config)
    else:
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    if output_folder is None:
        # output_folder = 'uni_diff_tf_proj_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)
        # output_folder = 'fin_sup_leak_uni_diff_{}_gamma{}_rate{}_ds_{}_{}_{}'.format(n_epochs, gamma, learning_rate, *downsample)
        # output_folder = 'fin_sup_pos_l1_uni_diff_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)
        # output_folder = 'fin_sup_360_stoch_{}_mskrl_{}_iter_{}_alphad_{}_alphab_{}_rate{}_ds_{}_{}_{}'.format(minibatch_size, n_epochs_mask_release, n_epochs, alpha_d, alpha_b, learning_rate, *downsample)
        output_folder = 'rot_bi_nn_360_stoch_{}_mskrl_{}_iter_{}_alphad_{}_alphab_{}_rate{}_ds_{}_{}_{}'.format(minibatch_size, n_epochs_mask_release, n_epochs, alpha_d, alpha_b, learning_rate, *downsample)
        # output_folder = 'rot_bi_bl_180_stoch_{}_mskrl_{}_iter_{}_alphad_{}_alphab_{}_rate{}_ds_{}_{}_{}'.format(minibatch_size, n_epochs_mask_release, n_epochs, alpha_d, alpha_b, learning_rate, *downsample)

    t0 = time.time()

    # read data
    print('Reading data...')
    f = h5py.File(fname, 'r')
    prj = f['exchange/data'][...]
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))

    dim_y, dim_x = prj.shape[-2:]
    n_theta = prj.shape[0]

    # read rotation data
    coord_ls = read_all_origin_coords('arrsize_64_64_64_ntheta_500', n_theta)
    print(coord_ls.get_shape().as_list())

    if minibatch_size is None:
        minibatch_size = n_theta

    if n_epochs_mask_release is None:
        n_epochs_mask_release = np.inf

    # convert data
    prj = tf.convert_to_tensor(prj, dtype=np.complex64)
    theta = -np.linspace(theta_st, theta_end, n_theta)
    theta_ls_tensor = tf.constant(theta, dtype='float32')

    # initialize
    # 2 channels are for real and imaginary parts respectively

    # ====================================================
    grid_delta = np.load('phantom/grid_delta.npy')
    grid_beta = np.load('phantom/grid_beta.npy')
    obj_init = np.zeros([dim_y, dim_x, dim_x, 2])
    obj_init[:, :, :, 0] = grid_delta.mean()
    obj_init[:, :, :, 1] = grid_delta.mean()
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)
    # ====================================================

    # =============== finite support mask ==============
    obj_pr = dxchange.read_tiff_stack('paganin_obj/recon_00000.tiff', range(64), 5)
    obj_pr = gaussian_filter(np.abs(obj_pr), sigma=1, mode='constant')
    mask = np.zeros_like(obj_pr)
    mask[obj_pr > 3e-5] = 1
    dxchange.write_tiff_stack(mask, 'fin_sup_mask/mask', dtype='float32', overwrite=True)
    mask_add = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], 2])
    mask_add[:, :, :, 0] = mask
    mask_add[:, :, :, 1] = mask
    mask_add = tf.convert_to_tensor(mask_add, dtype=tf.float32)
    # ==================================================

    loss = tf.constant(0.0)

    batch_inds = tf.placeholder(dtype=tf.int64)
    i = tf.constant(0)
    c = lambda i, loss, obj: tf.less(i, minibatch_size)

    _, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])

    # loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))
    # loss = loss / n_theta + gamma * energy_leak(obj, mask_add)
    if alpha_d is None:
        reg_term = alpha * tf.norm(obj, ord=1)
    else:
        reg_term = alpha_d * tf.norm(obj[:, :, :, 0], ord=1) + alpha_b * tf.norm(obj[:, :, :, 1], ord=1)

    loss = loss / minibatch_size + reg_term
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularizer', reg_term)
    tf.summary.scalar('error', loss - reg_term)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    loss_ls = []
    reg_ls = []

    sess.run(tf.global_variables_initializer())

    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(os.path.join(output_folder, 'tb'), graph_def=sess.graph_def)

    t0 = time.time()

    print('Optimizer started.')

    n_loop = n_epochs if n_epochs != 'auto' else max_nepochs
    for epoch in range(n_loop):
        t00 = time.time()
        if minibatch_size < n_theta:
            shuffled_inds = range(n_theta)
            np.random.shuffle(shuffled_inds)
            batches = create_batches(shuffled_inds, minibatch_size)
            for i_batch in range(len(batches)):
                this_batch = batches[i_batch]
                if len(this_batch) < minibatch_size:
                    this_batch = np.pad(this_batch, [0, minibatch_size-len(this_batch)], 'constant')
                _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op], feed_dict={batch_inds: this_batch})
                print('Minibatch done.')
        else:
            _, current_loss, current_reg, summary_str = sess.run([optimizer, loss, reg_term, merged_summary_op], feed_dict={batch_inds: np.arange(n_theta, dtype=int)})
        # =============non negative hard================
        obj = tf.nn.relu(obj)
        # ==============================================
        if n_epochs == 'auto':
            if len(loss_ls) > 0:
                print((current_loss - loss_ls[-1]) / loss_ls[-1])
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
            if epoch % 20 == 0 and epoch > 0:
                mask_temp = sess.run(obj[:, :, :, 0] > 1e-8)
                boolean = np.zeros_like(obj_init)
                boolean[:, :, :, 0] = mask_temp
                boolean[:, :, :, 1] = mask_temp
                boolean = tf.convert_to_tensor(boolean)
                mask_add = mask_add * tf.cast(boolean, tf.float32)
                dxchange.write_tiff_stack(sess.run(mask_add[:, :, :, 0]),
                                          'fin_sup_mask/epoch_{}/mask'.format(epoch), dtype='float32', overwrite=True)
            # ==============================================
        loss_ls.append(current_loss)
        reg_ls.append(current_reg)
        summary_writer.add_summary(summary_str, epoch)
        if save_intermediate:
            temp_obj = sess.run(obj)
            temp_obj = np.abs(temp_obj)
            dxchange.write_tiff(temp_obj[32, :, :, 0],
                                fname=os.path.join(output_folder, 'intermediate', 'iter_{:03d}'.format(epoch)),
                                dtype='float32',
                                overwrite=True)
        print('Iteration {}; loss = {}; time = {} s'.format(epoch, current_loss, time.time() - t00))

    print('Total time: {}'.format(time.time() - t0))

    res = sess.run(obj)
    dxchange.write_tiff_stack(res[:, :, :, 0], fname=os.path.join(output_folder, 'delta'), dtype='float32', overwrite=True)
    dxchange.write_tiff_stack(res[:, :, :, 1], fname=os.path.join(output_folder, 'beta'), dtype='float32', overwrite=True)

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