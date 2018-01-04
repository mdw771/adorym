import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
import dxchange
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.misc import imrotate
import matplotlib.pyplot as plt
import tomopy


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = PI
n_theta = 500
img_dim = 64
n_epochs = 200
alpha = 1.e-4
# ============================================


def rotate_and_project(i, loss, obj):

    loss += tf.reduce_mean(tf.squared_difference(tf.reduce_sum(tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR'), 1)[:, :, 0], prj[i]))
    i = tf.add(i, 1)
    return (i, loss, obj)


def rotate_and_project_2(obj, theta):

    obj_tensor = tf.convert_to_tensor(obj)
    obj_tensor = tf.reshape(obj_tensor, shape=[img_dim_y, img_dim_x, img_dim_x, 1])
    prjobj = sess.run(tf.reduce_sum(tf_rotate(obj_tensor, theta, interpolation='BILINEAR'), 1)[:, :, 0])
    return prjobj


sess = tf.Session()

try:
    prj = dxchange.read_tiff_stack('test_data/00000.tiff', range(0, n_theta), 5)
    obj_true = np.load('grid_delta.npy') * 1e7
    print(prj.shape)
except:
    obj_true = np.load('grid_delta.npy') * 1e7
    print('Creating test data.')
    for i, theta in enumerate(np.linspace(theta_st, theta_end, n_theta)):
        print(i)
        img = rotate_and_project_2(obj_true, theta)
        dxchange.write_tiff(img, 'test_data/{:05d}'.format(i), dtype='float32', overwrite=True)
    prj = dxchange.read_tiff_stack('test_data/00000.tiff', range(0, n_theta), 5)
    print(prj.shape)

img_dim_y = prj.shape[1]
img_dim_x = prj.shape[2]

prj = tf.convert_to_tensor(prj)

theta_ls_tensor = tf.constant(np.linspace(theta_st, theta_end, n_theta), dtype='float32')

# rec = tomopy.recon(prj, tomopy.angles(n_theta), algorithm='gridrec')
# dxchange.write_tiff(np.squeeze(prj[:, 32, :]), 'sinogram', dtype='float32', overwrite=True)
# dxchange.write_tiff_stack(rec, 'gridrec_results/grid', dtype='float32', overwrite=True)
#
# raise Exception

obj = tf.Variable(initial_value=tf.random_normal([img_dim_y, img_dim_x, img_dim_x, 1]))
# obj = tf.Variable(obj_true, dtype='float32')
# obj = tf.reshape(obj, shape=[img_dim, img_dim, img_dim, 1])


loss = tf.constant(0.0)

# d_theta = (theta_end - theta_st) / (n_theta - 1)
# theta_ls = np.linspace(theta_st, theta_end, n_theta)
i = tf.constant(0)
c = lambda i, loss, obj: tf.less(i, n_theta)

# obj = tf_rotate(obj, -d_theta, interpolation='BILINEAR')

_, loss, _ = tf.while_loop(c, rotate_and_project, [i, loss, obj])

loss = loss / n_theta + alpha * tf.reduce_sum(tf.image.total_variation(obj))

optimizer = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)

loss_ls = []

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):

    _, current_loss = sess.run([optimizer, loss])
    loss_ls.append(current_loss)
    print(sess.run(tf.reduce_sum(tf.image.total_variation(obj))))
    print('Iteration {}; loss = {}'.format(epoch, current_loss))

res = sess.run(obj)
dxchange.write_tiff_stack(res[:, :, :, 0], fname='test_results/obj', dtype='float32', overwrite=True)

plt.plot(range(n_epochs), loss_ls)
plt.show()

