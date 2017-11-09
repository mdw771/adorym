import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate
import dxchange
import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt


PI = 3.1415927

# ============================================
theta_st = 0
theta_end = PI
n_theta = 100
img_dim = 64
n_epochs = 20
# ============================================


try:
    prj = dxchange.read_tiff_stack('test_data/00000.tiff', range(0, n_theta), 5)
    obj_true = np.load('grid_delta.npy') * 1e7
    print(prj.shape)
except:
    obj_true = np.load('grid_delta.npy') * 1e7
    print('Creating test data.')
    for i, theta in enumerate(np.linspace(theta_st, theta_end, n_theta)):
        temp = rotate(obj_true, np.rad2deg(theta), axes=(1, 2), reshape=False)
        img = np.sum(temp, axis=1)
        dxchange.write_tiff(img, 'test_data/{:05d}'.format(i), dtype='float32', overwrite=True)
        prj = dxchange.read_tiff_stack('test_data/00000.tiff', range(0, n_theta), 5)
        print(prj.shape)

obj = tf.random_normal([img_dim, img_dim, img_dim])
prj_pred = tf.zeros([n_theta, img_dim, img_dim])
loss = tf.Variable(initial_value=0., dtype='float32')

d_theta = (theta_end - theta_st) / (n_theta - 1)
for i, theta in enumerate(np.linspace(theta_st, theta_end, n_theta)):
    obj = tf_rotate(obj, d_theta)
    loss = tf.add(loss, tf.reduce_sum(tf.squared_difference(tf.reduce_sum(obj, 1), prj[i])))

optimizer = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()

sess.run(tf.global_variables_initializer())
for epoch in range(n_epochs):
    _, current_loss = sess.run([optimizer, loss])
    print('Iteration {}; loss = {}'.format(epoch, current_loss))

res = sess.run(obj)
dxchange.write_tiff_stack(res, fname='test_results/obj', dtype='float32', overwrite=True)
