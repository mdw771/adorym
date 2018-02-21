import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
import dxchange
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import rotate
from scipy.misc import imrotate
import matplotlib.pyplot as plt
import tomopy
import time
import os
from util import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_epochs = 200
# alpha_ls = np.arange(1e-5, 1e-4, 1e-5)
alpha_ls = [1e-7]
gamma_ls = [0]
# learning_rate_ls = [1]
learning_rate_ls = [5e-6]
center = 32
energy_ev = 5000
psize_cm = 1e-7
batch_size = 100
# output_folder = 'recon_h5_{}_alpha{}'.format(n_epochs, alpha)
# ============================================


def reconstruct(fname, theta_st=0, theta_end=PI, n_epochs=200, alpha=1e-4, gamma=1e-2, learning_rate=1.0, output_folder=None, downsample=None,
                save_intermediate=False):

    def rotate_and_project(i, loss, obj):

        rand_proj = batch_inds[i]
        obj_rot = tf_rotate(obj, theta_ls_tensor[rand_proj], interpolation='BILINEAR')
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
    sess = tf.Session()

    if output_folder is None:
        # output_folder = 'uni_diff_tf_proj_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)
        # output_folder = 'fin_sup_leak_uni_diff_{}_gamma{}_rate{}_ds_{}_{}_{}'.format(n_epochs, gamma, learning_rate, *downsample)
        # output_folder = 'fin_sup_pos_l1_uni_diff_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)
        output_folder = 'fin_sup_360_stoch_{}_pos_l1_uni_diff_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(batch_size, n_epochs, alpha, learning_rate, *downsample)
        # output_folder = 'dual_sphere_diff_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)

    t0 = time.time()

    # read data
    print('Reading data...')
    f = h5py.File(fname, 'r')
    prj = f['exchange/data'][...]
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))

    dim_y, dim_x = prj.shape[-2:]
    n_theta = prj.shape[0]

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

    # obj = tf.Variable(initial_value=tf.zeros([dim_y, dim_x, dim_x, 2]), dtype=tf.float32)
    # obj += 0.5

    batches = []
    for _ in range(n_epochs):
        batches.append(np.random.choice(range(n_theta), batch_size, replace=False))
    batches = np.array(batches)

    loss = tf.constant(0.0)

    i = tf.constant(0)
    c = lambda i, loss, obj: tf.less(i, batch_size)

    batch_inds = tf.placeholder(dtype=tf.int64, shape=(batch_size))

    _, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])

    # loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))
    # loss = loss / n_theta + gamma * energy_leak(obj, mask_add)
    loss = loss / n_theta + alpha * tf.norm(obj, ord=1)


    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    loss_ls = []

    sess.run(tf.global_variables_initializer())

    # ===========================================================
    # exiting = multislice_propagate(obj[:, :, :, 0], obj[:, :, :, 1], energy_ev, psize_cm)
    # exiting = sess.run(exiting)
    # dxchange.write_tiff(np.abs(exiting), 'diffraction_dat_tf/mag', dtype='float32', overwrite=True)
    # dxchange.write_tiff(np.angle(exiting), 'diffraction_dat_tf/phase', dtype='float32', overwrite=True)
    # ===========================================================

    t0 = time.time()

    print('Optimizer started.')

    for epoch in range(n_epochs):

        t00 = time.time()
        _, current_loss = sess.run([optimizer, loss], feed_dict={batch_inds: batches[epoch]})
        # =============finite support===================
        if epoch != n_epochs - 1:
            obj = obj * mask_add
        # ==============================================
        # =============non negative hard================
        if epoch != n_epochs - 1:
            obj = tf.nn.relu(obj)
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
        if save_intermediate:
            temp_obj = sess.run(obj)
            temp_obj = np.abs(temp_obj)
            dxchange.write_tiff(temp_obj[32, :, :, 0],
                                      fname=os.path.join(output_folder, 'intermediate', 'iter_{:03d}'.format(epoch)),
                                      dtype='float32',
                                      overwrite=True)
            # ===============================================
            # obj_rot = obj
            # exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], energy_ev, psize_cm)
            # exiting = sess.run(exiting)
            # dxchange.write_tiff(np.abs(exiting),
            #                     os.path.join(output_folder, 'intermediate', 'wave_{:03d}'.format(epoch)),
            #                     dtype='float32',
            #                     overwrite=True)
            # ===============================================
        # print(sess.run(tf.reduce_sum(tf.image.total_variation(obj))))
        print('Iteration {}; loss = {}; time = {} s'.format(epoch, current_loss, time.time() - t00))

    print('Total time: {}'.format(time.time() - t0))

    res = sess.run(obj)
    dxchange.write_tiff_stack(res[:, :, :, 0], fname=os.path.join(output_folder, 'delta'), dtype='float32', overwrite=True)
    dxchange.write_tiff_stack(res[:, :, :, 1], fname=os.path.join(output_folder, 'beta'), dtype='float32', overwrite=True)

    plt.figure()
    plt.semilogy(range(n_epochs), loss_ls)
    # plt.show()
    try:
        os.makedirs(os.path.join(output_folder, 'convergence'))
    except:
        pass
    plt.savefig(os.path.join(output_folder, 'convergence', 'converge.png'), format='png')
    np.save(os.path.join(output_folder, 'convergence', 'converge'), loss_ls)


if __name__ == '__main__':

    for alpha in alpha_ls:
        for gamma in gamma_ls:
            for learning_rate in learning_rate_ls:
                print('Rate: {}; gamma: {}'.format(learning_rate, gamma))
                reconstruct(fname='data_diff_tf_360.h5',
                            n_epochs=n_epochs,
                            theta_st=theta_st,
                            theta_end=theta_end,
                            gamma=gamma,
                            alpha=alpha,
                            learning_rate=learning_rate,
                            downsample=(0, 0, 0),
                            save_intermediate=True)
