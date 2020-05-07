import os
from six.moves import input

import numpy as np
import dxchange
import h5py
from scipy.ndimage import rotate as sp_rotate
from tqdm import tqdm

from adorym.util import *
from adorym.propagate import *
import adorym.global_settings as global_settings


def create_ptychography_data_batch_numpy(energy_ev, psize_cm, n_theta, phantom_path, save_path, fname, probe_pos,
                                         probe_type='gaussian', probe_size=(72, 72), probe_initial=None,
                                         theta_st=0, theta_end=2*PI, minibatch_size=23, fresnel_approx=True,
                                         free_prop_cm='inf', normalize_fft=False, **kwargs):
    """
    If probe_type is 'gaussian', supply parameters 'probe_mag_sigma', 'probe_phase_sigma', 'probe_phase_max'.
    """
    def rotate_and_project(theta, obj, probe_pos, i_theta=None):
        obj_rot = sp_rotate(obj, -theta, reshape=False, axes=(1, 2))

        # Add probe_pos offset error if demanded
        if 'pos_offset_vec' in kwargs.keys() and kwargs['pos_offset_vec'] is not None:
            pos_offset = kwargs['pos_offset_vec'][ii]
            probe_pos += pos_offset
            print('Added positional error for theta ID {}: {}.'.format(i_theta, pos_offset))

        # pad if needed
        obj_rot, pad_arr = pad_object(obj_rot, grid_delta.shape, probe_pos, probe_size)

        e_real_ls = []
        e_imag_ls = []
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
            grid_batch = np.stack([grid_delta_ls, grid_beta_ls], -1)
            e_real, e_imag = multislice_propagate_batch(grid_batch, probe_real, probe_imag, energy_ev,
                                                 psize_cm, free_prop_cm=free_prop_cm,
                                                 obj_batch_shape=[len(pos_batch), probe_size[0], probe_size[1], grid_delta.shape[-1]],
                                                 fresnel_approx=fresnel_approx, normalize_fft=normalize_fft)
            e_real_ls.append(e_real)
            e_imag_ls.append(e_imag)
        e_real_ls = np.concatenate(e_real_ls, 0)
        e_imag_ls = np.concatenate(e_imag_ls, 0)
        return e_real_ls, e_imag_ls

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
    theta_ls = np.linspace(theta_st, theta_end, n_theta)
    theta_ls = np.rad2deg(theta_ls)

    # create data file
    flag_overwrite = 'y'
    if os.path.exists(os.path.join(save_path, fname)):
        flag_overwrite = input('File exists. Overwrite? (y/n) ')
    if flag_overwrite in ['y', 'Y']:
        f = h5py.File(os.path.join(save_path, fname), 'w')
        grp = f.create_group('exchange')
        dat = grp.create_dataset('data', shape=(n_theta, len(probe_pos), probe_size[0], probe_size[1]), dtype=np.complex64)
    else:
        return

    probe_real, probe_imag = initialize_probe(probe_size, probe_type, probe_initial=probe_initial, **kwargs)
    probe_mag, probe_phase = real_imag_to_mag_phase(probe_real, probe_imag)

    dxchange.write_tiff(probe_mag, os.path.join(save_path, 'probe_mag'), dtype='float64', overwrite=True)
    dxchange.write_tiff(probe_phase, os.path.join(save_path, 'probe_phase'), dtype='float64', overwrite=True)
    # dxchange.write_tiff(probe_mag, os.path.join(save_path, 'probe_mag_f32'), dtype='float32', overwrite=True)
    # dxchange.write_tiff(probe_phase, os.path.join(save_path, 'probe_phase_f32'), dtype='float32', overwrite=True)

    np.savetxt(os.path.join(save_path, 'probe_pos.txt'), probe_pos, fmt='%d')
    if 'pos_offset_vec' in kwargs.keys():
        if kwargs['pos_offset_vec'] is not None:
            np.savetxt(os.path.join(save_path, 'pos_offset_vec.txt'), kwargs['pos_offset_vec'], fmt='%d')

    for ii, theta in enumerate(theta_ls):
        print('Theta: {}'.format(ii))
        waveset_out_real, waveset_out_imag = rotate_and_project(theta, obj, probe_pos, i_theta=ii)
        waveset_out = waveset_out_real + waveset_out_imag * 1j
        dat[ii, :, :, :] = waveset_out
        dxchange.write_tiff(abs(waveset_out), os.path.join(save_path, 'diffraction_dat', 'mag_{:05d}'.format(ii)), overwrite=True, dtype='float32')
    f.close()
    return
