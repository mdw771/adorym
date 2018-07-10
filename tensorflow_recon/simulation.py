import os

import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import rotate as tf_rotate

from util import *


def create_fullfield_data(energy_ev, psize_cm, free_prop_cm, n_theta, phantom_path, save_folder, fname,
                          probe_type='plane', wavefront_initial=None, theta_st=0, theta_end=2*PI):

    def rotate_and_project(i, obj):
        obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
        exiting = multislice_propagate(obj_rot[:, :, :, 0], obj_rot[:, :, :, 1], probe_real, probe_imag, energy_ev,
                                       psize_cm, free_prop_cm=free_prop_cm)
        return exiting

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    # read model
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
    img_dim = grid_delta.shape
    obj_init = np.zeros(np.append(img_dim, 2))
    obj_init[:, :, :, 0] = grid_delta
    obj_init[:, :, :, 1] = grid_beta
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)

    # list of angles
    theta_ls = -np.linspace(theta_st, theta_end, n_theta)
    theta_ls_tensor = tf.constant(theta_ls, dtype='float32')

    # create data file
    flag_overwrite = 'y'
    if os.path.exists(os.path.join(save_folder, fname)):
        flag_overwrite = raw_input('File exists. Overwrite? (y/n) ')
    if flag_overwrite in ['y', 'Y']:
        f = h5py.File(os.path.join(save_folder, fname), 'w')
        grp = f.create_group('exchange')
        dat = grp.create_dataset('data', shape=(n_theta, grid_delta.shape[0], grid_delta.shape[1]), dtype=np.complex64)
    else:
        return

    # create probe function
    if probe_type == 'plane':
        probe_real = tf.constant(np.ones([img_dim[0], img_dim[1]]), dtype=tf.float32)
        probe_imag = tf.constant(np.zeros([img_dim[0], img_dim[1]]), dtype=tf.float32)
    elif probe_type == 'fixed':
        assert wavefront_initial.shape == img_dim
        probe_mag, probe_phase = wavefront_initial
        probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
        probe_real = tf.constant(probe_real, dtype=tf.float32)
        probe_imag = tf.constant(probe_imag, dtype=tf.float32)
    else:
        raise ValueError('Invalid wavefront type. Choose from \'plane\' or \'fixed\'.')

    sess.run(tf.global_variables_initializer())

    for i, theta in enumerate(theta_ls):
        print(i)
        wave = rotate_and_project(i, obj)
        wave = sess.run(wave)
        dat[i, :, :] = wave
        dxchange.write_tiff(abs(wave), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(i)), overwrite=True, dtype='float32')
    f.close()
    return


def create_ptychography_data(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                             probe_type='gaussian', probe_size=(72, 72), wavefront_initial=None,
                             theta_st=0, theta_end=2*PI, **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(i, obj):
        obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
        for j, pos in enumerate(probe_pos):
            subobj = tf.slice(obj_rot, [pos[0] - probe_size_half[0], pos[1] - probe_size_half[1], 0],
                              [probe_size[0], probe_size[1], img_dim[2]])
            exiting = multislice_propagate(subobj[:, :, :, 0], subobj[:, :, :, 1], probe_real, probe_imag, energy_ev,
                                           psize_cm, free_prop_cm=None)
            exiting = fftshift(tf.fft2d(exiting))
        return exiting

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    # read model
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
    img_dim = grid_delta.shape
    probe_size_half = (np.array(probe_size) / 2).astype('int')
    obj_init = np.zeros(np.append(img_dim, 2))
    obj_init[:, :, :, 0] = grid_delta
    obj_init[:, :, :, 1] = grid_beta
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)

    # list of angles
    theta_ls = -np.linspace(theta_st, theta_end, n_theta)
    theta_ls_tensor = tf.constant(theta_ls, dtype='float32')

    # create data file
    flag_overwrite = 'y'
    if os.path.exists(os.path.join(save_folder, fname)):
        flag_overwrite = raw_input('File exists. Overwrite? (y/n) ')
    if flag_overwrite in ['y', 'Y']:
        f = h5py.File(os.path.join(save_folder, fname), 'w')
        grp = f.create_group('exchange')
        dat = grp.create_dataset('data', shape=(n_theta, len(probe_pos), probe_size[0], probe_size[1]), dtype=np.complex64)
    else:
        return

    # create probe
    if probe_type == 'gaussian':
        py = np.arange(probe_size[0]) - (probe_size[0] - 1.) / 2
        px = np.arange(probe_size[1]) - (probe_size[1] - 1.) / 2
        pxx, pyy = np.meshgrid(px, py)
        probe_mag = np.exp(-(pxx ** 2 + pyy ** 2) / (2 * kwargs['probe_mag_sigma'] ** 2))
        probe_phase = kwargs['probe_phase_max'] * np.exp(-(pxx ** 2 + pyy ** 2) / (2 * kwargs['probe_phase_sigma'] ** 2))
        probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)

    sess.run(tf.global_variables_initializer())

    for i, theta in enumerate(theta_ls):
        wave = rotate_and_project(i, obj)
        wave = sess.run(wave)
        dat[i, :, :] = wave
        dxchange.write_tiff(abs(wave), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(i)), overwrite=True, dtype='float32')
    f.close()
    return