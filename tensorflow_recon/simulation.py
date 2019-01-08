import os
from six.moves import input

import numpy as np
import dxchange
import tensorflow as tf
import h5py
from tensorflow.contrib.image import rotate as tf_rotate
from scipy.ndimage import rotate as sp_rotate
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

from util import *
from npfuncs import *


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


def create_fullfield_data_numpy(energy_ev, psize_cm, free_prop_cm, n_theta, phantom_path, save_folder, fname, batch_size=1,
                                probe_type='plane', wavefront_initial=None, theta_st=0, theta_end=2*PI, monitor_output=False, **kwargs):

    def rotate_and_project(this_theta_batch, obj):
        obj_rot_batch = []
        for theta in this_theta_batch:
            obj_rot_batch.append(sp_rotate(obj, theta, reshape=False, axes=(1, 2)))
        obj_rot_batch = np.array(obj_rot_batch)
        if probe_type == 'point':
            exiting = multislice_propagate_spherical_numpy(obj_rot_batch[:, :, :, :, 0], obj_rot_batch[:, :, :, :, 1],
                                                           probe_real, probe_imag, energy_ev,
                                                           psize_cm, dist_to_source_cm, det_psize_cm,
                                                           theta_max, phi_max, free_prop_cm,
                                                           obj_batch_shape=obj_rot_batch.shape[:-1])
        else:
            exiting = multislice_propagate_batch_numpy(obj_rot_batch[:, :, :, :, 0], obj_rot_batch[:, :, :, :, 1],
                                                       probe_real, probe_imag, energy_ev,
                                                       psize_cm, free_prop_cm=free_prop_cm, obj_batch_shape=obj_rot_batch.shape[:-1])
        return exiting

    # read model
    grid_delta = np.load(os.path.join(phantom_path, 'grid_delta.npy'))
    grid_beta = np.load(os.path.join(phantom_path, 'grid_beta.npy'))
    img_dim = grid_delta.shape
    obj = np.zeros(np.append(img_dim, 2))
    obj[:, :, :, 0] = grid_delta
    obj[:, :, :, 1] = grid_beta

    if probe_type == 'point':
        dist_to_source_cm = kwargs['dist_to_source_cm']
        det_psize_cm = kwargs['det_psize_cm']
        theta_max = kwargs['theta_max']
        phi_max = kwargs['phi_max']

    # list of angles
    theta_ls = -np.linspace(theta_st, theta_end, n_theta) / np.pi * 180
    n_batch = np.ceil(float(n_theta) / batch_size)
    theta_batch = np.array_split(theta_ls, n_batch)

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
        probe_real = np.ones([img_dim[0], img_dim[1]], dtype='float32')
        probe_imag = np.zeros([img_dim[0], img_dim[1]], dtype='float32')
    elif probe_type == 'fixed':
        assert wavefront_initial.shape == img_dim
        probe_mag, probe_phase = wavefront_initial
        probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
    elif probe_type == 'point':
        probe_real = np.ones([img_dim[0], img_dim[1]], dtype='float32')
        probe_imag = np.zeros([img_dim[0], img_dim[1]], dtype='float32')
    elif probe_type == 'gaussian':
        probe_mag_sigma = kwargs['probe_mag_sigma']
        probe_phase_sigma = kwargs['probe_phase_sigma']
        probe_phase_max = kwargs['probe_phase_max']
        py = np.arange(obj.shape[0]) - (obj.shape[0] - 1.) / 2
        px = np.arange(obj.shape[1]) - (obj.shape[1] - 1.) / 2
        pxx, pyy = np.meshgrid(px, py)
        probe_mag = np.exp(-(pxx ** 2 + pyy ** 2) / (2 * probe_mag_sigma ** 2))
        probe_phase = probe_phase_max * np.exp(
            -(pxx ** 2 + pyy ** 2) / (2 * probe_phase_sigma ** 2))
        probe_real, probe_imag = mag_phase_to_real_imag(probe_mag, probe_phase)
    else:
        raise ValueError('Invalid wavefront type. Choose from \'plane\', \'point\', or \'fixed\'.')

    for i_batch, this_theta_batch in tqdm(enumerate(theta_batch)):
        wave_out = rotate_and_project(this_theta_batch, obj)
        if monitor_output:
            dxchange.write_tiff(np.abs(wave_out), os.path.join(save_folder, 'monitor_output', 'prj_{}'.format(i_batch)), dtype='float32', overwrite=True)
        dat[i_batch * batch_size:i_batch * batch_size + batch_size, :, :] = wave_out
    f.close()
    return


def create_ptychography_data(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                             probe_type='gaussian', probe_size=(72, 72), wavefront_initial=None,
                             theta_st=0, theta_end=2*PI, probe_circ_mask=0.9, n_dp_batch=20, **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(i, obj):

        # obj_rot = apply_rotation(obj, coord_ls[rand_proj], 'arrsize_64_64_64_ntheta_500')
        obj_rot = tf_rotate(obj, theta_ls_tensor[i], interpolation='BILINEAR')
        probe_pos_batch_ls = np.array_split(probe_pos, int(np.ceil(float(n_pos) / n_dp_batch)))
        # probe_pos_batch_ls = np.array_split(probe_pos, 6)
        exiting_ls = []

        # pad if needed
        pad_arr = np.array([[0, 0], [0, 0]])
        if probe_pos[:, 0].min() - probe_size_half[0] < 0:
            pad_len = probe_size_half[0] - probe_pos[:, 0].min()
            obj_rot = tf.pad(obj_rot, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode='CONSTANT')
            pad_arr[0, 0] = pad_len
        if probe_pos[:, 0].max() + probe_size_half[0] > obj_size[0]:
            pad_len = probe_pos[:, 0].max() + probe_size_half[0] - obj_size[0]
            obj_rot = tf.pad(obj_rot, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='CONSTANT')
            pad_arr[0, 1] = pad_len
        if probe_pos[:, 1].min() - probe_size_half[1] < 0:
            pad_len = probe_size_half[1] - probe_pos[:, 1].min()
            obj_rot = tf.pad(obj_rot, ((0, 0), (pad_len, 0), (0, 0), (0, 0)), mode='CONSTANT')
            pad_arr[1, 0] = pad_len
        if probe_pos[:, 1].max() + probe_size_half[1] > obj_size[1]:
            pad_len = probe_pos[:, 1].max() + probe_size_half[0] - obj_size[1]
            obj_rot = tf.pad(obj_rot, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode='CONSTANT')
            pad_arr[1, 1] = pad_len

        for k, pos_batch in enumerate(probe_pos_batch_ls):
            subobj_ls = []
            for j, pos in enumerate(pos_batch):
                pos = [int(x) for x in pos]
                pos[0] = pos[0] + pad_arr[0, 0]
                pos[1] = pos[1] + pad_arr[1, 0]
                subobj = obj_rot[pos[0] - probe_size_half[0]:pos[0] - probe_size_half[0] + probe_size[0],
                                 pos[1] - probe_size_half[1]:pos[1] - probe_size_half[1] + probe_size[1],
                                 :, :]
                subobj_ls.append(subobj)

            subobj_ls = tf.stack(subobj_ls)
            exiting = multislice_propagate_batch(subobj_ls[:, :, :, :, 0], subobj_ls[:, :, :, :, 1], probe_real,
                                                 probe_imag,
                                                 energy_ev, psize_cm, h=h, free_prop_cm='inf',
                                                 obj_batch_shape=[len(pos_batch), *probe_size, obj_size[-1]])
            exiting_ls.append(exiting)
        exiting_ls = tf.concat(exiting_ls, 0)
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
    obj_size = grid_delta.shape
    obj = tf.Variable(initial_value=obj_init, dtype=tf.float32)

    # list of angles
    theta_ls = -np.linspace(theta_st, theta_end, n_theta)
    theta_ls_tensor = tf.constant(theta_ls, dtype='float32')
    n_pos = len(probe_pos)
    probe_pos = np.array(probe_pos)

    # generate Fresnel kernel
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    lmbda_nm = 1240. / energy_ev
    delta_nm = voxel_nm[-1]
    kernel = get_kernel(delta_nm, lmbda_nm, voxel_nm, probe_size)
    h = tf.convert_to_tensor(kernel, dtype=tf.complex64, name='kernel')

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
        obj_rot = sp_rotate(obj, theta, reshape=False, axes=(1, 2))

        # pad if needed
        pad_arr = np.array([[0, 0], [0, 0]])
        if probe_pos[:, 0].min() - probe_size_half[0] < 0:
            pad_len = probe_size_half[0] - probe_pos[:, 0].min()
            obj_rot = np.pad(obj_rot, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[0, 0] = pad_len
        if probe_pos[:, 0].max() + probe_size_half[0] > img_dim[0]:
            pad_len = probe_pos[:, 0].max() + probe_size_half[0] - img_dim[0]
            obj_rot = np.pad(obj_rot, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[0, 1] = pad_len
        if probe_pos[:, 1].min() - probe_size_half[1] < 0:
            pad_len = probe_size_half[1] - probe_pos[:, 1].min()
            obj_rot = np.pad(obj_rot, ((0, 0), (pad_len, 0), (0, 0), (0, 0)), mode='constant')
            pad_arr[1, 0] = pad_len
        if probe_pos[:, 1].max() + probe_size_half[1] > img_dim[1]:
            pad_len = probe_pos[:, 1].max() + probe_size_half[0] - img_dim[1]
            obj_rot = np.pad(obj_rot, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode='constant')
            pad_arr[1, 1] = pad_len

        for k, pos_batch in tqdm(enumerate(probe_pos_batches)):
            grid_delta_ls = []
            grid_beta_ls = []
            for j, pos in enumerate(pos_batch):
                pos = np.array(pos, dtype=int)
                pos[0] = pos[0] + pad_arr[0, 0]
                pos[1] = pos[1] + pad_arr[1, 0]
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

    probe_pos = np.array(probe_pos)
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
