import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
import dxchange
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.misc import imrotate
import matplotlib.pyplot as plt
import tomopy
import time
from util import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = PI
n_epochs = 200
sino_range = (0, 50, 1)
# ============================================


def rotate_and_project(i, loss, obj):

    loss += tf.reduce_mean(tf.squared_difference(tf.reduce_sum(tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR'), 1)[:, :, 0], prj[i]))
    i = tf.add(i, 1)
    return (i, loss, obj)


def rotate_and_project_2(obj, theta):

    obj_tensor = tf.convert_to_tensor(obj)
    obj_tensor = tf.reshape(obj_tensor, shape=[dim_y, dim_x, dim_x, 1])
    prjobj = sess.run(tf.reduce_sum(tf_rotate(obj_tensor, theta, interpolation='BILINEAR'), 1)[:, :, 0])
    return prjobj


sess = tf.Session()

t0 = time.time()
print('Reading data...')
prj, flt, drk, theta = dxchange.read_aps_32id('data.h5', sino=sino_range)
print('Data reading: {} s'.format(time.time() - t0))
print('Data shape: {}'.format(prj.shape))
prj = tomopy.normalize(prj, flt, drk)
prj = preprocess(prj)

prj = tomopy.downsample(prj, level=2, axis=2)
print('Downsampled shape: {}'.format(prj.shape))

dim_y, dim_x = prj.shape[-2:]

n_theta = prj.shape[0]

prj = tf.convert_to_tensor(prj)

# theta_ls_tensor = tf.constant(np.linspace(theta_st, theta_end, n_theta), dtype='float32')
theta_ls_tensor = tf.constant(theta, dtype='float32')

# rec = tomopy.recon(prj, tomopy.angles(n_theta), algorithm='gridrec')
# dxchange.write_tiff(np.squeeze(prj[:, 32, :]), 'sinogram', dtype='float32', overwrite=True)
# dxchange.write_tiff_stack(rec, 'gridrec_results/grid', dtype='float32', overwrite=True)
#
# raise Exception

obj = tf.Variable(initial_value=tf.random_normal([dim_y, dim_x, dim_x, 1]))
# obj = tf.Variable(obj_true, dtype='float32')
# obj = tf.reshape(obj, shape=[img_dim, img_dim, img_dim, 1])


loss = tf.constant(0.0)

# d_theta = (theta_end - theta_st) / (n_theta - 1)
# theta_ls = np.linspace(theta_st, theta_end, n_theta)
i = tf.constant(0)
c = lambda i, loss, obj: tf.less(i, n_theta)

# obj = tf_rotate(obj, -d_theta, interpolation='BILINEAR')

_, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])

loss = loss / n_theta + 1.e-4 * tf.reduce_sum(tf.image.total_variation(obj))

optimizer = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)

loss_ls = []

sess.run(tf.global_variables_initializer())

t0 = time.time()

print('Optimizer started.')

for epoch in range(n_epochs):

    t00 = time.time()
    _, current_loss = sess.run([optimizer, loss])
    loss_ls.append(current_loss)
    print(sess.run(tf.reduce_sum(tf.image.total_variation(obj))))
    print('Iteration {}; loss = {}; time = {} s'.format(epoch, current_loss, time.time() - t00))

print('Total time: {}'.format(time.time() - t0))

res = sess.run(obj)
dxchange.write_tiff_stack(res[:, :, :, 0], fname='recon_h5/recon', dtype='float32', overwrite=True)

plt.plot(range(n_epochs), loss_ls)
plt.show()

