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
from propagate import *


def create_fullfield_data_numpy(energy_ev, psize_cm, free_prop_cm, n_theta, phantom_path, save_folder, fname, batch_size=1,
                                probe_type='plane', wavefront_initial=None, theta_st=0, theta_end=2*PI, monitor_output=False, **kwargs):

    def rotate_and_project(this_theta_batch, obj):
        obj_rot_batch = []
        for theta in this_theta_batch:
            obj_rot_batch.append(sp_rotate(obj, theta, reshape=False, axes=(1, 2)))
        obj_rot_batch = np.array(obj_rot_batch)
        if probe_type == 'point':
            raise DeprecationWarning
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


def create_ptychography_data_batch_numpy(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                                         probe_type='gaussian', probe_size=(72, 72), wavefront_initial=None,
                                         theta_st=0, theta_end=2*PI, probe_circ_mask=0.9, minibatch_size=23, fresnel_approx=True,
                                         free_prop_cm='inf', **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(theta, obj):
        obj_rot = sp_rotate(obj, theta, reshape=False, axes=(1, 2))

        # pad if needed
        obj_rot, pad_arr = pad_object(obj_rot, grid_delta.shape, probe_pos, probe_size)

        for k, pos_batch in tqdm(enumerate(probe_pos_batches)):
            grid_delta_ls = []
            grid_beta_ls = []
            for j, pos in enumerate(pos_batch):
                pos = np.array(pos, dtype=int)
                pos[0] = pos[0] + pad_arr[0, 0]
                pos[1] = pos[1] + pad_arr[1, 0]
                subobj = obj_rot[pos[0]:pos[0] + probe_size[0], pos[1]:pos[1] + probe_size[1], :, :]
                grid_delta_ls.append(subobj[:, :, :, 0])
                grid_beta_ls.append(subobj[:, :, :, 1])
            grid_delta_ls = np.array(grid_delta_ls)
            grid_beta_ls = np.array(grid_beta_ls)
            exiting = multislice_propagate_batch_numpy(grid_delta_ls, grid_beta_ls, probe_real, probe_imag, energy_ev,
                                                       psize_cm, free_prop_cm=free_prop_cm,
                                                       obj_batch_shape=[len(pos_batch), probe_size[0], probe_size[1], grid_delta.shape[-1]], fresnel_approx=fresnel_approx)
            if k == 0:
                exiting_ls = np.copy(exiting)
            else:
                exiting_ls = np.vstack([exiting_ls, exiting])
            dxchange.write_tiff(abs(exiting), 'cone_256_foam_ptycho/test', dtype='float32')
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

    probe_real, probe_imag = initialize_probe(probe_size, probe_type, probe_initial=wavefront_initial)
    probe_mag, probe_phase = real_imag_to_mag_phase(probe_real, probe_imag)

    dxchange.write_tiff(probe_mag, os.path.join(save_folder, 'probe_mag'), dtype='float64', overwrite=True)
    dxchange.write_tiff(probe_phase, os.path.join(save_folder, 'probe_phase'), dtype='float64', overwrite=True)
    dxchange.write_tiff(probe_mag, os.path.join(save_folder, 'probe_mag_f32'), dtype='float32', overwrite=True)
    dxchange.write_tiff(probe_phase, os.path.join(save_folder, 'probe_phase_f32'), dtype='float32', overwrite=True)

    for ii, theta in enumerate(theta_ls):
        print('Theta: {}'.format(ii))
        waveset_out = rotate_and_project(theta, obj)
        dat[ii, :, :, :] = waveset_out
        dxchange.write_tiff(abs(waveset_out), os.path.join(save_folder, 'diffraction_dat', 'mag_{:05d}'.format(ii)), overwrite=True, dtype='float32')
    f.close()
    return
