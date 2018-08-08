import os
from six.moves import input

import numpy as np
import dxchange
import tensorflow as tf
import h5py
from tensorflow.contrib.image import rotate as tf_rotate
from scipy.ndimage import gaussian_filter
import tomopy

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

    i = tf.placeholder(dtype=tf.int32)

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
        flag_overwrite = input('File exists. Overwrite? (y/n) ')
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

    wave = rotate_and_project(i, obj)

    for ii, theta in enumerate(theta_ls):
        print(ii)
        sess.run(tf.global_variables_initializer())
        wave_out = sess.run(wave, feed_dict={i:ii})
        dat[ii, :, :] = wave_out
        dxchange.write_tiff(abs(wave_out), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(ii)), overwrite=True, dtype='float32')
    f.close()
    return


def create_ptychography_data(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                             probe_type='gaussian', probe_size=(72, 72), wavefront_initial=None,
                             theta_st=0, theta_end=2*PI, probe_circ_mask=0.9, **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(i, obj):
        obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
        exiting_ls = []
        for j, pos in enumerate(probe_pos):
            print('Pos: {}'.format(j))
            subobj = tf.slice(obj_rot, [int(pos[0]) - probe_size_half[0], int(pos[1]) - probe_size_half[1], 0, 0],
                              [probe_size[0], probe_size[1], img_dim[2], 2])
            exiting = multislice_propagate(subobj[:, :, :, 0], subobj[:, :, :, 1], probe_real, probe_imag, energy_ev,
                                           psize_cm, free_prop_cm=None)
            if probe_circ_mask is not None:
                exiting = exiting * probe_mask
            exiting = fftshift(tf.fft2d(exiting))
            exiting_ls.append(exiting)
        return exiting_ls

    config = tf.ConfigProto(device_count={'GPU': 0})
    sess = tf.Session(config=config)

    i = tf.placeholder(dtype=tf.int32)

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
        flag_overwrite = input('File exists. Overwrite? (y/n) ')
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
        if probe_circ_mask is not None:
            probe_real, probe_imag = tomopy.circ_mask(np.array([probe_real, probe_imag]), axis=0, ratio=probe_circ_mask)
            probe_mask = tomopy.circ_mask(np.ones_like(probe_real)[np.newaxis, :, :], axis=0, ratio=probe_circ_mask)
            probe_mask = gaussian_filter(np.squeeze(probe_mask), 3)
            probe_mask = tf.constant(probe_mask, dtype=tf.complex64)

    waveset = rotate_and_project(i, obj)

    for ii, theta in enumerate(theta_ls):
        print('Theta: {}'.format(ii))
        sess.run(tf.global_variables_initializer())
        waveset_out = np.array(sess.run(waveset, feed_dict={i: ii}))
        dat[ii, :, :, :] = waveset_out
        dxchange.write_tiff(abs(waveset_out), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(ii)), overwrite=True, dtype='float32')
    f.close()
    return


def create_ptychography_data_batch_numpy(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                                         probe_type='gaussian', probe_size=(72, 72), wavefront_initial=None,
                                         theta_st=0, theta_end=2*PI, probe_circ_mask=0.9, minibatch_size=20, **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(theta, obj):
        obj_rot = sp_rotate(obj, theta, reshape=False)

        for k, pos_batch in tqdm(enumerate(probe_pos_batches)):
            grid_delta_ls = []
            grid_beta_ls = []
            for j, pos in enumerate(pos_batch):
                pos = np.array(pos, dtype=int)
                subobj = obj_rot[pos[0] - probe_size_half[0]:pos[0] - probe_size_half[0] + probe_size[0],
                                 pos[1] - probe_size_half[1]:pos[1] - probe_size_half[1] + probe_size[1],
                                 :, :]
                grid_delta_ls.append(subobj[:, :, :, 0])
                grid_beta_ls.append(subobj[:, :, :, 1])
            grid_delta_ls = np.array(grid_delta_ls)
            grid_beta_ls = np.array(grid_beta_ls)
            exiting = multislice_propagate_batch_numpy(grid_delta_ls, grid_beta_ls, probe_real, probe_imag, energy_ev,
                                                       psize_cm, free_prop_cm='inf',
                                                       obj_batch_shape=[len(pos_batch), probe_size[0], probe_size[1], grid_delta.shape[-1]])
            if probe_circ_mask is not None:
                exiting = exiting * probe_mask
            if k == 0:
                exiting_ls = np.copy(exiting)
            else:
                exiting_ls = np.vstack([exiting_ls, exiting])
        return exiting_ls

    n_pos = len(probe_pos)
    minibatch_size = min([minibatch_size, n_pos])
    n_batch = np.ceil(float(n_pos) / minibatch_size)
    print(n_pos, minibatch_size, n_batch)
    probe_pos_batches = np.array_split(probe_pos, n_batch)

    # read model
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
    img_dim = grid_delta.shape
    probe_size_half = (np.array(probe_size) / 2).astype('int')
    obj = np.zeros(np.append(img_dim, 2))
    obj[:, :, :, 0] = grid_delta
    obj[:, :, :, 1] = grid_beta

    # list of angles
    theta_ls = -np.linspace(theta_st, theta_end, n_theta)
    theta_ls = np.rad2deg(theta_ls)

    # create data file
    flag_overwrite = 'y'
    if os.path.exists(os.path.join(save_folder, fname)):
        flag_overwrite = input('File exists. Overwrite? (y/n) ')
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
        if probe_circ_mask is not None:
            probe_real, probe_imag = tomopy.circ_mask(np.array([probe_real, probe_imag]), axis=0, ratio=probe_circ_mask)
            probe_mask = tomopy.circ_mask(np.ones_like(probe_real)[np.newaxis, :, :], axis=0, ratio=probe_circ_mask)
            probe_mask = gaussian_filter(np.squeeze(probe_mask), 3)

    for ii, theta in enumerate(theta_ls):
        print('Theta: {}'.format(ii))
        waveset_out = rotate_and_project(theta, obj)
        dat[ii, :, :, :] = waveset_out
        dxchange.write_tiff(abs(waveset_out), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(ii)), overwrite=True, dtype='float32')
    f.close()
    return