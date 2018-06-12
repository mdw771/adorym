import xdesign
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator
import h5py
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate

from util import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI ###############################
n_theta = 500
energy_ev = 5000
psize_cm = 1.e-7 ###############################
free_prop_cm = None
# ============================================

def rotate_and_project(i, obj):

    # coord_old = read_origin_coords('arrsize_64_64_64_ntheta_500', i)
    # obj_rot = apply_rotation(obj, coord_old, 'arrsize_64_64_64_ntheta_500')
    obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
    exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], probe_real, probe_imag, energy_ev, psize_cm, free_prop_cm=free_prop_cm)
    # exiting = tf.abs(exiting)
    return exiting

config = tf.ConfigProto(device_count = {'GPU': 0})
sess = tf.Session(config=config)

# read model
# grid_delta = np.load('adhesin/phantom/grid_delta.npy')
grid_delta = np.load('cone_256_filled/phantom/grid_delta.npy')
# grid_beta = np.load('adhesin/phantom/grid_beta.npy')
grid_beta = np.load('cone_256_filled/phantom/grid_beta.npy')
img_dim = grid_delta.shape[0]
obj_init = np.zeros([img_dim, img_dim, img_dim, 2])
obj_init[:, :, :, 0] = grid_delta
obj_init[:, :, :, 1] = grid_beta
obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)

# list of angles
theta_ls = -np.linspace(theta_st, theta_end, n_theta)
theta_ls_tensor = tf.constant(theta_ls, dtype='float32')

# create data file
f = h5py.File('cone_256_filled/data_cone_256_1nm_1um.h5', 'w')
grp = f.create_group('exchange')
dat = grp.create_dataset('data', shape=(n_theta, grid_delta.shape[0], grid_delta.shape[1]), dtype=np.complex64)

# create probe function
probe_mag = np.ones([img_dim, img_dim], dtype=np.float32)
probe_phase = np.zeros([img_dim, img_dim], dtype=np.float32)
probe_phase[int(img_dim / 2), int(img_dim / 2)] = 0.1
probe_phase = gaussian_filter(probe_phase, 3)
# probe_phase = np.repeat(probe_phase[np.newaxis, :], img_dim, axis=0)
probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
print(probe_real)
print(probe_imag)
probe_real = tf.constant(probe_real)
probe_imag = tf.constant(probe_imag)

sess.run(tf.global_variables_initializer())

for i, theta in enumerate(theta_ls):

    print(i)
    wave = rotate_and_project(i, obj)
    wave = sess.run(wave)
    dat[i, :, :] = wave
    dxchange.write_tiff(np.abs(wave), 'diffraction_dat_tf/mag_{:05d}'.format(i), overwrite=True, dtype='float32')

f.close()