from tensorflow.contrib.image import rotate as tf_rotate
from tensorflow.python.client import timeline
import dxchange
import time
import os
from util import *


PI = 3.1415927

def reconstrct(fname, sino_range, theta_st=0, theta_end=PI, n_epochs=200, alpha=1e-4, learning_rate=1.0,
               output_folder=None, output_name='recon', downsample=None,
               save_intermediate=False, initial_guess=None, center=None):

    def rotate_and_project(i, loss, obj):

        loss += tf.reduce_mean(tf.squared_difference(
            tf.reduce_sum(tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR'), 1)[:, :, 0], prj[i]))
        i = tf.add(i, 1)
        return (i, loss, obj)

    # def rotate_and_project_2(obj, theta):
    #
    #     obj_tensor = tf.convert_to_tensor(obj)
    #     obj_tensor = tf.reshape(obj_tensor, shape=[dim_y, dim_x, dim_x, 1])
    #     prjobj = sess.run(tf.reduce_sum(tf_rotate(obj_tensor, theta, interpolation='BILINEAR'), 1)[:, :, 0])
    #     return prjobj

    f = open('loss.txt', 'a')

    # with tf.device('/gpu:0'):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

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
        _, current_loss = sess.run([optimizer, loss])
        loss_ls.append(current_loss)
        if save_intermediate:
            temp_obj = sess.run(obj, options=run_options, run_metadata=run_metadata)

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