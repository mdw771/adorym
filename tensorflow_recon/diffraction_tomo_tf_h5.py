import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
import dxchange
import numpy as np
import h5py
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
theta_end = PI
n_epochs = 600
sino_range = (600, 601, 1)
# alpha_ls = np.arange(1e-5, 1e-4, 1e-5)
alpha_ls = [0]
# learning_rate_ls = [1]
learning_rate_ls = [5e-6]
center = 32
energy_ev = 5000
psize_cm = 1e-7
# output_folder = 'recon_h5_{}_alpha{}'.format(n_epochs, alpha)
# ============================================


def reconstrct(fname, theta_st=0, theta_end=PI, n_epochs=200, alpha=1e-4, learning_rate=1.0, output_folder=None, downsample=None,
               save_intermediate=False):

    def rotate_and_project(i, loss, obj):

        obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
        exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], energy_ev, psize_cm)
        exiting = tf.pow(tf.abs(exiting), 2)
        loss += tf.reduce_mean(tf.squared_difference(exiting, tf.pow(tf.abs(prj[i]), 2)))
        i = tf.add(i, 1)
        return (i, loss, obj)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess = tf.Session()

    if output_folder is None:
        # output_folder = 'uni_diff_tf_proj_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)
        output_folder = 'fin_sup_uni_diff_{}_alpha{}_rate{}_ds_{}_{}_{}'.format(n_epochs, alpha, learning_rate, *downsample)

    t0 = time.time()

    # read data
    print('Reading data...')
    f = h5py.File(fname, 'r')
    prj = f['exchange/data'][...]
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))

    # convert to intensity and drop phase
    prj = np.abs(prj) ** 2

    # correct for center
    # offset = int(prj.shape[-1] / 2) - center
    # if offset != 0:
    #     for i in range(prj.shape[0]):
    #         prj[i, :, :] = realign_image(prj[i, :, :], [0, offset])

    # downsample
    if downsample is not None:
        prj = tomopy.downsample(prj, level=downsample[0], axis=0)
        prj = tomopy.downsample(prj, level=downsample[1], axis=1)
        prj = tomopy.downsample(prj, level=downsample[2], axis=2)
        print('Downsampled shape: {}'.format(prj.shape))

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
    obj_init[:, :, :, 1] = grid_beta.mean()
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)
    # ====================================================

    # obj = tf.Variable(initial_value=tf.zeros([dim_y, dim_x, dim_x, 2]), dtype=tf.float32)
    # obj += 0.5

    loss = tf.constant(0.0)

    i = tf.constant(0)
    c = lambda i, loss, obj: tf.less(i, n_theta)


    _, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])

    loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    loss_ls = []

    sess.run(tf.global_variables_initializer())

    # ===========================================================
    # exiting = multislice_propagate(obj[:, :, :, 0], obj[:, :, :, 1], energy_ev, psize_cm)
    # exiting = sess.run(exiting)
    # dxchange.write_tiff(np.abs(exiting), 'diffraction_dat_tf/mag', dtype='float32', overwrite=True)
    # dxchange.write_tiff(np.angle(exiting), 'diffraction_dat_tf/phase', dtype='float32', overwrite=True)
    # ===========================================================

    # =============== finite support mask ==============
    from scipy.signal import convolve
    kernel = np.ones([10, 10, 10])
    mask = (grid_delta > 1e-10).astype('float')
    mask = convolve(mask, kernel, mode='same')
    mask[mask < 1e-10] = 0
    mask[mask > 1e-10] = 1
    dxchange.write_tiff_stack(mask, 'temp/mask', overwrite=True, dtype='float32')
    mask_add = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], 2])
    mask_add[:, :, :, 0] = mask
    mask_add[:, :, :, 1] = mask
    mask_add = tf.convert_to_tensor(mask_add, dtype=tf.float32)
    # ==================================================




    t0 = time.time()

    print('Optimizer started.')

    for epoch in range(n_epochs):

        t00 = time.time()
        _, current_loss = sess.run([optimizer, loss])
        # =============finite support===================
        obj = obj * mask_add
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
    dxchange.write_tiff_stack(res[:, :, :, 0], fname=os.path.join(output_folder, 'recon'), dtype='float32', overwrite=True)

    plt.figure()
    plt.semilogy(range(n_epochs), loss_ls)
    # plt.show()
    plt.savefig(os.path.join(output_folder, 'converge.png'), format='png')


if __name__ == '__main__':

    for alpha in alpha_ls:
        for learning_rate in learning_rate_ls:
            print('Rate: {}; alpha: {}'.format(learning_rate, alpha))
            reconstrct(fname='data_diff_tf.h5',
                       n_epochs=n_epochs,
                       alpha=alpha,
                       learning_rate=learning_rate,
                       downsample=(0, 0, 0),
                       save_intermediate=True)
