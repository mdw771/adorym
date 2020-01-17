import autograd.numpy as np
import dxchange
import h5py
import matplotlib.pyplot as plt
import matplotlib
import warnings
from math import ceil, floor
try:
    import sys
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import fourier_shift
except:
    warnings.warn('Some dependencies are screwed up.')
import os
import pickle
import glob
from scipy.special import erf

from constants import *
from interpolation import *


class Simulator(object):
    """Optical simulation based on multislice propagation.

    Attributes
    ----------
    grid : numpy.ndarray or list of numpy.ndarray
        Descretized grid for the phantom object. If type == 'refractive_index',
        it takes a list of [delta_grid, beta_grid].
    energy : float
        Beam energy in eV. Should match the energy used for creating the grids.
    psize : list
        Pixel size in cm.
    type : str
        Value type of input grid.
    """

    def __init__(self, energy, grid=None, psize=None, type='refractive_index'):

        if type == 'refractive_index':
            if grid is not None:
                self.grid_delta, self.grid_beta = grid
            else:
                self.grid_delta = self.grid_beta = None
            self.energy_kev = energy * 1.e-3
            self.voxel_nm = np.array(psize) * 1.e7
            self.mean_voxel_nm = np.prod(self.voxel_nm)**(1. / 3)
            self._ndim = 3
            self.size_nm = np.array(self.grid_delta.get_shape().as_list()) * self.voxel_nm
            self.shape = self.grid_delta.get_shape().as_list()
            self.lmbda_nm = 1.24 / self.energy_kev
            self.mesh = []
            temp = []
            for i in range(self._ndim):
                temp.append(np.arange(self.shape[i]) * self.voxel_nm[i])
            self.mesh = np.meshgrid(*temp, indexing='xy')

            # wavefront in x-y plane or x edge
            self.wavefront = np.zeros(self.grid_delta.shape[:-1], dtype=np.complex64)
        else:
            raise ValueError('Currently only delta and beta grids are supported.')

    def save_grid(self, save_path='data/sav/grid'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(os.path.join(save_path, 'grid_delta'), self.grid_delta)
        np.save(os.path.join(save_path, 'grid_beta'), self.grid_beta)
        grid_pars = [self.shape, self.voxel_nm, self.energy_kev * 1.e3]
        np.save(os.path.join(save_path, 'grid_pars'), grid_pars)

    def read_grid(self, save_path='data/sav/grid'):
        try:
            self.grid_delta = np.load(os.path.join(save_path, 'grid_delta.npy'), allow_pickle=True)
            self.grid_beta = np.load(os.path.join(save_path, 'grid_beta.npy'), allow_pickle=True)
        except:
            raise ValueError('Failed to read grid.')

    def save_slice_images(self, save_path='data/sav/slices'):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        dxchange.write_tiff_stack(self.grid_delta, os.path.join(save_path, 'delta'),
                                  overwrite=True, dtype=np.float32)
        dxchange.write_tiff_stack(self.grid_beta, os.path.join(save_path, 'beta'),
                                  overwrite=True, dtype=np.float32)

    def show_grid(self, part='delta'):
        import tifffile
        if part == 'delta':
            tifffile.imshow(self.grid_delta)
        elif part == 'beta':
            tifffile.imshow(self.grid_beta)
        else:
            warnings.warn('Wrong part specified for show_grid.')

    def initialize_wavefront(self, type, **kwargs):
        """Initialize wavefront.

        Parameters:
        -----------
        type : str
            Type of wavefront to be initialized. Valid options:
            'plane', 'spot', 'point_projection_lens', 'spherical'
        kwargs :
            Options specific to the selection of type.
            'plane': no option
            'spot': 'width'
            'point_projection_lens': 'focal_length', 'lens_sample_dist'
            'spherical': 'dist_to_source'
        """
        wave_shape = np.asarray(self.wavefront.shape)
        if type == 'plane':
            self.wavefront[...] = 1.
        elif type == 'spot':
            wid = kwargs['width']
            radius = int(wid / 2)
            if self._ndim == 2:
                center = int(wave_shape[0] / 2)
                self.wavefront[center-radius:center-radius+wid] = 1.
            elif self._ndim == 3:
                center = np.array(wave_shape / 2, dtype=int)
                self.wavefront[center[0]-radius:center[0]-radius+wid, center[1]-radius:center[1]-radius+wid] = 1.
        elif type == 'spherical':
            z = kwargs['dist_to_source']
            xx = self.mesh[0][:, :, 0]
            yy = self.mesh[1][:, :, 0]
            xx -= xx[0, -1] / 2
            yy -= yy[-1, 0] / 2
            print(xx, yy, z)
            r = np.sqrt(xx ** 2 + yy ** 2 + z ** 2)
            self.wavefront = np.exp(-1j * 2 * np.pi * r / self.lmbda_nm)
        elif type == 'point_projection_lens':
            f = kwargs['focal_length']
            s = kwargs['lens_sample_dist']
            xx = self.mesh[0][:, :, 0]
            yy = self.mesh[1][:, :, 0]
            xx -= xx[0, -1] / 2
            yy -= yy[-1, 0] / 2
            r = np.sqrt(xx ** 2 + yy ** 2)
            theta = np.arctan(r / (s - f))
            path = np.mod(s / np.cos(theta), self.lmbda_nm)
            phase = path * 2 * PI
            wavefront = np.ones(wave_shape).astype('complex64')
            wavefront = wavefront + 1j * np.tan(phase)
            self.wavefront = wavefront / np.abs(wavefront)


def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True):
    """Get Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    k = 2 * PI / lmbda_nm
    u_max = 1. / (2. * voxel_nm[0])
    v_max = 1. / (2. * voxel_nm[1])
    u, v = gen_mesh([v_max, u_max], grid_shape[0:2])
    # H = np.exp(1j * k * dist_nm * np.sqrt(1 - lmbda_nm**2 * (u**2 + v**2)))
    if fresnel_approx:
        H = np.exp(1j * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    else:
        quad = 1 - lmbda_nm ** 2 * (u**2 + v**2)
        quad_inner = np.clip(quad, a_min=0, a_max=None)
        H = np.exp(-1j * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))

    return H


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape):

    """
    Get Fresnel propagation kernel for IR algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    size_nm = np.array(voxel_nm) * np.array(grid_shape)
    k = 2 * PI / lmbda_nm
    ymin, xmin = np.array(size_nm)[:2] / -2.
    dy, dx = voxel_nm[0:2]
    x = np.arange(xmin, xmin + size_nm[1], dx)
    y = np.arange(ymin, ymin + size_nm[0], dy)
    x, y = np.meshgrid(x, y)
    try:
        h = np.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
        H = np.fft.fftshift(fft2(h)) * voxel_nm[0] * voxel_nm[1]
        dxchange.write_tiff(x, '2d_512/monitor_output/x', dtype='float32', overwrite=True)
    except:
        h = tf.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * tf.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
        # h = tf.convert_to_tensor(h, dtype='complex64')
        H = np.fft.fftshift(np.fft.fft2(h)) * voxel_nm[0] * voxel_nm[1]

    return H


def preprocess(dat, blur=None, normalize_bg=False):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    # if normalize_bg:
    #     dat = tomopy.normalize_bg(dat)
    dat = -np.log(dat)
    dat[np.where(np.isnan(dat) == True)] = 0
    if blur is not None:
        dat = gaussian_filter(dat, blur)

    return dat


def realign_image(arr, shift):
    """
    Translate and rotate image via Fourier

    Parameters
    ----------
    arr : ndarray
        Image array.

    shift: tuple
        Mininum and maximum values to rescale data.

    angle: float, optional
        Mininum and maximum values to rescale data.

    Returns
    -------
    ndarray
        Output array.
    """
    # if both shifts are integers, do circular shift; otherwise perform Fourier shift.
    if np.count_nonzero(np.abs(np.array(shift) - np.round(shift)) < 0.01) == 2:
        temp = np.roll(arr, int(shift[0]), axis=0)
        temp = np.roll(temp, int(shift[1]), axis=1)
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
    return temp


def exp_j(a):

    return np.cos(a) + 1j * np.sin(a)


def multislice_propagate_batch_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm,
                                     free_prop_cm=None, obj_batch_shape=None, kernel=None, fresnel_approx=True,
                                     pure_projection=False):

    minibatch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    wavefront = np.zeros([minibatch_size, obj_batch_shape[1], obj_batch_shape[2]], dtype='complex64')
    wavefront += (probe_real + 1j * probe_imag)

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slice = obj_batch_shape[-1]
    delta_nm = voxel_nm[-1]

    if kernel is not None:
        h = kernel
    else:
        h = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=fresnel_approx)
    k = 2. * PI * delta_nm / lmbda_nm

    for i in range(n_slice):
        delta_slice = grid_delta_batch[:, :, :, i]
        beta_slice = grid_beta_batch[:, :, :, i]
        c = exp_j(k * delta_slice) * np.exp(-k * beta_slice)
        wavefront = wavefront * c
        if i < n_slice - 1 and not pure_projection:
            wavefront = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))

    if free_prop_cm is not None:
        if free_prop_cm == 'inf':
            wavefront = np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2])
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            algorithm = 'TF' if mean_voxel_nm > crit_samp else 'IR'
            # print(algorithm)
            algorithm = 'TF'
            if algorithm == 'TF':
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
    return wavefront


def create_batches(arr, batch_size):

    arr_len = len(arr)
    i = 0
    batches = []
    while i < arr_len:
        batches.append(arr[i:min(i+batch_size, arr_len)])
        i += batch_size
    return batches


def save_rotation_lookup(array_size, n_theta, dest_folder=None):

    image_center = [np.floor(x / 2) for x in array_size]

    coord0 = np.arange(array_size[0])
    coord1 = np.arange(array_size[1])
    coord2 = np.arange(array_size[2])

    coord2_vec = np.tile(coord2, array_size[1])

    coord1_vec = np.tile(coord1, array_size[2])
    coord1_vec = np.reshape(coord1_vec, [array_size[1], array_size[2]])
    coord1_vec = np.reshape(np.transpose(coord1_vec), [-1])

    coord0_vec = np.tile(coord0, [array_size[1] * array_size[2]])
    coord0_vec = np.reshape(coord0_vec, [array_size[1] * array_size[2], array_size[0]])
    coord0_vec = np.reshape(np.transpose(coord0_vec), [-1])

    # move origin to image center
    coord1_vec = coord1_vec - image_center[1]
    coord2_vec = coord2_vec - image_center[2]

    # create matrix of coordinates
    coord_new = np.stack([coord1_vec, coord2_vec]).astype(np.float32)

    # create rotation matrix
    theta_ls = np.linspace(0, 2 * np.pi, n_theta)
    coord_old_ls = []
    for theta in theta_ls:
        m_rot = np.array([[np.cos(theta),  -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
        coord_old = np.matmul(m_rot, coord_new)
        coord1_old = np.round(coord_old[0, :] + image_center[1]).astype(np.int)
        coord2_old = np.round(coord_old[1, :] + image_center[2]).astype(np.int)
        # clip coordinates
        coord1_old = np.clip(coord1_old, 0, array_size[1]-1)
        coord2_old = np.clip(coord2_old, 0, array_size[2]-1)
        coord_old = np.stack([coord1_old, coord2_old], axis=1)
        coord_old_ls.append(coord_old)
    if dest_folder is None:
        dest_folder = 'arrsize_{}_{}_{}_ntheta_{}'.format(array_size[0], array_size[1], array_size[2], n_theta)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    # coord_old_ls are the coordinates in original (0-deg) object frame at each angle, corresponding to each
    # voxel in the object at that angle.
    for i, arr in enumerate(coord_old_ls):
        np.save(os.path.join(dest_folder, '{:04}'.format(i)), arr)

    # coord_vec's are coordinates list of current object (ordered, e.g. (0, 0, 0), (0, 0, 1), ...)
    coord1_vec = coord1_vec + image_center[1]
    coord1_vec = np.tile(coord1_vec, array_size[0])
    coord2_vec = coord2_vec + image_center[2]
    coord2_vec = np.tile(coord2_vec, array_size[0])
    for i, coord in enumerate([coord0_vec, coord1_vec, coord2_vec]):
        np.save(os.path.join(dest_folder, 'coord{}_vec'.format(i)), coord)

    return coord_old_ls


def read_origin_coords(src_folder, index):

    coords = np.load(os.path.join(src_folder, '{:04}.npy'.format(index)), allow_pickle=True)
    return coords


def read_all_origin_coords(src_folder, n_theta):

    coord_ls = []
    for i in range(n_theta):
        coord_ls.append(read_origin_coords(src_folder, i))
    return coord_ls


def apply_rotation(obj, coord_old, src_folder):

    coord_vec_ls = []
    for i in range(3):
        f = os.path.join(src_folder, 'coord{}_vec.npy'.format(i))
        coord_vec_ls.append(np.load(f, allow_pickle=True))
    s = obj.shape
    coord0_vec, coord1_vec, coord2_vec = coord_vec_ls

    coord_old = np.tile(coord_old, [s[0], 1])
    coord1_old = coord_old[:, 0]
    coord2_old = coord_old[:, 1]
    coord_old = np.stack([coord0_vec, coord1_old, coord2_old], axis=1).transpose()
    # print(sess.run(coord_old))

    obj_channel_ls = np.split(obj, s[3], 3)
    obj_rot_channel_ls = []
    for channel in obj_channel_ls:
        channel_flat = channel.flatten()
        ind = coord_old[0] * (s[1] * s[2]) + coord_old[1] * s[2] + coord_old[2]
        ind = ind.astype('int')
        obj_chan_new_val = channel_flat[ind]
        obj_rot_channel_ls.append(np.reshape(obj_chan_new_val, s[:-1]))
    obj_rot = np.stack(obj_rot_channel_ls, axis=3)
    return obj_rot


def get_rotated_subblocks(dset, this_pos_batch, coord_old, probe_size_half, whole_object_size, monochannel=False, mode='hdf5'):
    """
    Get rotated subblocks centering this_pos_batch directly from hdf5.
    :return: [n_pos, y, x, z, 2]
    """
    block_stack = []
    for coords in this_pos_batch:
        if len(coords) == 2:
            # For the case of ptychography
            this_y, this_x = coords
            coord0_vec = np.arange(this_y - probe_size_half[0], this_y + probe_size_half[0])
            coord1_vec = np.arange(this_x - probe_size_half[1], this_x + probe_size_half[1])
            block_shape = [probe_size_half[0] * 2, probe_size_half[1] * 2, whole_object_size[-1]]
        else:
            # For the case of full-field
            line_st, line_end, px_st, px_end = coords
            coord0_vec = np.arange(line_st, line_end)
            coord1_vec = np.arange(px_st, px_end)
            block_shape = [line_end - line_st, px_end - px_st, whole_object_size[-1]]
        coord2_vec = np.arange(whole_object_size[-1])
        coord1_vec = np.clip(coord1_vec, 0, whole_object_size[1] - 1)
        array_size = (len(coord0_vec), len(coord1_vec), len(coord2_vec))

        coord2_vec = np.tile(coord2_vec, array_size[1])
        coord1_vec = np.repeat(coord1_vec, array_size[2])

        # Flattened sub-block indices in current object frame
        ind_new = coord1_vec * whole_object_size[2] + coord2_vec

        # Flattened sub-block indices in original object frame
        ind_old_1 = coord_old[:, 0][ind_new].astype(int)
        ind_old_2 = coord_old[:, 1][ind_new].astype(int)
        if mode == 'hdf5':
            ind_old = ind_old_1 * whole_object_size[1] + ind_old_2
            # if not monochannel:
            #     n_channels = dset.shape[-1]
            #     ind_old_channels = np.zeros(ind_old.size * n_channels, dtype=int)
            #     for i_chan in range(n_channels):
            #         ind_old_channels[i_chan::n_channels] = ind_old * n_channels + i_chan

        this_block = []
        # Take data with flattened 2nd and 3rd dimensions
        if mode == 'npy':
            if not monochannel:
                this_block = dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]),
                                  ind_old_1, ind_old_2, :]
            else:
                this_block = dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]),
                             ind_old_1, ind_old_2]
        elif mode == 'hdf5':
            # H5py only supports taking elements using monotonically increasing indices without repeating.
            sorted_ind = np.argsort(ind_old)
            sorted_coords = ind_old[sorted_ind]
            sorted_coords_unique, unique_pos = np.unique(sorted_coords, return_index=True)
            if not monochannel:
                this_block_temp = dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]), sorted_coords_unique, :]
                repeats = np.roll(unique_pos, -1) - unique_pos
                repeats[-1] += len(ind_old)
                this_block = np.repeat(this_block_temp, repeats, axis=1)
                this_block = this_block[:, np.argsort(sorted_ind), :]
            else:
                this_block_temp = dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]), sorted_coords_unique]
                repeats = np.roll(unique_pos, -1) - unique_pos
                repeats[-1] += len(ind_old)
                this_block = np.repeat(this_block_temp, repeats, axis=1)
                this_block = this_block[:, np.argsort(sorted_ind)]
        # Reshape and pad
        if not monochannel:
            this_block = np.reshape(this_block, [this_block.shape[0], block_shape[1], whole_object_size[2], 2])
            if coord0_vec[0] < 0:
                this_block = np.pad(this_block, [[-coord0_vec[0], 0], [0, 0], [0, 0], [0, 0]], mode='edge')
            if coord0_vec[-1] + 1 - whole_object_size[0] > 0:
                this_block = np.pad(this_block,
                                    [[0, coord0_vec[-1] + 1 - whole_object_size[0]], [0, 0], [0, 0], [0, 0]],
                                    mode='edge')
        else:
            this_block = np.reshape(this_block, [this_block.shape[0], block_shape[1], whole_object_size[2]])
            if coord0_vec[0] < 0:
                this_block = np.pad(this_block, [[-coord0_vec[0], 0], [0, 0], [0, 0]], mode='edge')
            if coord0_vec[-1] + 1 - whole_object_size[0] > 0:
                this_block = np.pad(this_block,
                                    [[0, coord0_vec[-1] + 1 - whole_object_size[0]], [0, 0], [0, 0]],
                                    mode='edge')
        # dxchange.write_tiff(this_block[:, :, :, 0], '/Users/ming/Research/Programs/du/adorym_dev/adhesin_ptycho_2/test/debug/patch', dtype='float32')
        block_stack.append(this_block)
    block_stack = np.stack(block_stack, axis=0)
    return block_stack


def write_subblocks_to_file(dset, this_pos_batch, obj_delta, obj_beta, coord_old, probe_size_half, whole_object_size, mask=False, mode='hdf5'):
    """
    Write data back in the npy. If monochannel, give None to obj_beta.
    """
    # TODO: when calculating coords, allow negative
    for i_batch, coords in enumerate(this_pos_batch):
        if len(coords) == 2:
            this_y, this_x = coords
            coord0_vec = np.arange(this_y - probe_size_half[0], this_y + probe_size_half[0])
            coord1_vec = np.arange(this_x - probe_size_half[1], this_x + probe_size_half[1])
        else:
            line_st, line_end, px_st, px_end = coords
            coord0_vec = np.arange(line_st, line_end)
            coord1_vec = np.arange(px_st, px_end)
        coord2_vec = np.arange(whole_object_size[2])
        coord1_clip_mask = (coord1_vec >= 0) * (coord1_vec <= whole_object_size[1] - 1)
        coord1_vec = np.clip(coord1_vec, 0, whole_object_size[1] - 1)
        array_size = (len(coord0_vec), len(coord1_vec), len(coord2_vec))

        coord2_vec = np.tile(coord2_vec, array_size[1])
        coord1_vec = np.repeat(coord1_vec, array_size[2])
        coord1_clip_mask = np.repeat(coord1_clip_mask, array_size[2])

        # Flattened sub-block indices in current object frame
        ind_new = coord1_vec[coord1_clip_mask] * whole_object_size[2] + coord2_vec[coord1_clip_mask]


        # Flattened sub-block indices in original object frame
        ind_old_1 = coord_old[:, 0][ind_new].astype(int)
        ind_old_2 = coord_old[:, 1][ind_new].astype(int)

        if mode == 'hdf5':
            ind_old = ind_old_1 * whole_object_size[1] + ind_old_2

        obj_crop_top = max([0, -coord0_vec[0]])
        obj_crop_bot = min([obj_delta.shape[1] - (coord0_vec[-1] + 1 - whole_object_size[0]),
                            obj_delta.shape[1]])
        try:
            obj_crop_left = max([0, -(this_x - probe_size_half[1])])
            obj_crop_right = min([obj_delta.shape[1] - (this_x + probe_size_half[1] - whole_object_size[0]),
                                obj_delta.shape[1]])
        except:
            obj_crop_left = max([0, -px_st])
            obj_crop_right = min([obj_delta.shape[1] - (px_end - whole_object_size[0]),
                                obj_delta.shape[1]])

        new_shape = [obj_crop_bot - obj_crop_top,
                     len(ind_old_1)]

        if mode == 'npy':
            if not mask:
                dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]),
                     ind_old_1, ind_old_2, 0] += \
                         np.reshape(obj_delta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)
                dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]),
                     ind_old_1, ind_old_2, 1] += \
                         np.reshape(obj_beta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)
            else:
                temp = np.reshape(obj_delta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)
                dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]),
                     ind_old_1, ind_old_2] *= temp
        elif mode == 'hdf5':

            sorted_ind = np.argsort(ind_old)
            sorted_coords = ind_old[sorted_ind]

            sorted_coords_unique, unique_pos = np.unique(sorted_coords, return_index=True)
            repeats = np.roll(unique_pos, -1) - unique_pos
            repeats[-1] += len(ind_old)
            # Update edge voxels only once
            # edge_mask = (sorted_coords_unique < whole_object_size[2]) + (sorted_coords_unique > whole_object_size[2] * (whole_object_size[1] - 1)) \
            #             + (sorted_coords_unique % whole_object_size[2] == 0) + (sorted_coords_unique % whole_object_size[2] == whole_object_size[2] - 1)
            # repeats[edge_mask] = 1

            # repeats[...] = 1


            if not mask:
                # Sum elements contributing to the same voxel in the object file
                ids = np.repeat(range(len(repeats)), repeats)
                increment_delta_full = np.reshape(obj_delta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)[:, sorted_ind]
                increment_delta_red = np.zeros([new_shape[0], len(unique_pos)])
                increment_beta_full = np.reshape(obj_beta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)[:, sorted_ind]
                increment_beta_red = np.zeros([new_shape[0], len(unique_pos)])
                for i0 in range(new_shape[0]):
                    increment_delta_red[i0, :] = np.bincount(ids, weights=increment_delta_full[i0]) / repeats
                    increment_beta_red[i0, :] = np.bincount(ids, weights=increment_beta_full[i0]) / repeats
                dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]), sorted_coords_unique, :] += \
                    np.stack([increment_delta_red, increment_beta_red], axis=-1)


            else:
                dset[max([0, coord0_vec[0]]):min([whole_object_size[0], coord0_vec[-1] + 1]), ind_old[sorted_ind]] += \
                    np.reshape(obj_delta[i_batch, obj_crop_top:obj_crop_bot, obj_crop_left:obj_crop_right, :], new_shape)[:, np.argsort(sorted_ind), :]
    return


def pad_object(obj_rot, this_obj_size, probe_pos, probe_size_half):
    """
    Pad the object with 0 if any of the probes' extents go beyond the object boundary.
    :return: padded object and padding lengths.
    """
    pad_arr = np.array([[0, 0], [0, 0]])
    if probe_pos[:, 0].min() - probe_size_half[0] < 0:
        pad_len = probe_size_half[0] - probe_pos[:, 0].min()
        obj_rot = np.pad(obj_rot, ((pad_len, 0), (0, 0), (0, 0), (0, 0)), mode='constant')
        pad_arr[0, 0] = pad_len
    if probe_pos[:, 0].max() + probe_size_half[0] > this_obj_size[0]:
        pad_len = probe_pos[:, 0].max() + probe_size_half[0] - this_obj_size[0]
        obj_rot = np.pad(obj_rot, ((0, pad_len), (0, 0), (0, 0), (0, 0)), mode='constant')
        pad_arr[0, 1] = pad_len
    if probe_pos[:, 1].min() - probe_size_half[1] < 0:
        pad_len = probe_size_half[1] - probe_pos[:, 1].min()
        obj_rot = np.pad(obj_rot, ((0, 0), (pad_len, 0), (0, 0), (0, 0)), mode='constant')
        pad_arr[1, 0] = pad_len
    if probe_pos[:, 1].max() + probe_size_half[1] > this_obj_size[1]:
        pad_len = probe_pos[:, 1].max() + probe_size_half[0] - this_obj_size[1]
        obj_rot = np.pad(obj_rot, ((0, 0), (0, pad_len), (0, 0), (0, 0)), mode='constant')
        pad_arr[1, 1] = pad_len

    return obj_rot, pad_arr


def total_variation_3d(arr, axis_offset=0):
    """
    Calculate total variation of a 3D array.
    :param arr: 3D Tensor.
    :return: Scalar.
    """
    res = np.sum(np.abs(np.roll(arr, 1, axis=0 + axis_offset) - arr))
    res = res + np.sum(np.abs(np.roll(arr, 1, axis=1 + axis_offset) - arr))
    res = res + np.sum(np.abs(np.roll(arr, 1, axis=2 + axis_offset) - arr))
    res /= arr.size
    return res


def generate_sphere(shape, radius, anti_aliasing=5):

    shape = np.array(shape)
    radius = int(radius)
    x = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    y = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    z = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    xx, yy, zz = np.meshgrid(x, y, z)
    a = (xx**2 + yy**2 + zz**2 <= radius**2).astype('float')
    res = np.zeros(shape * anti_aliasing)
    center_res = (np.array(res.shape) / 2).astype('int')
    res[center_res[0] - int(a.shape[0] / 2):center_res[0] + int(a.shape[0] / 2),
        center_res[1] - int(a.shape[0] / 2):center_res[1] + int(a.shape[0] / 2),
        center_res[2] - int(a.shape[0] / 2):center_res[2] + int(a.shape[0] / 2)] = a
    res = gaussian_filter(res, 0.5 * anti_aliasing)
    res = res[::anti_aliasing, ::anti_aliasing, ::anti_aliasing]
    return res


def generate_shell(shape, radius, anti_aliasing=5):

    sphere1 = generate_sphere(shape, radius + 0.5, anti_aliasing=anti_aliasing)
    sphere2 = generate_sphere(shape, radius - 0.5, anti_aliasing=anti_aliasing)
    return sphere1 - sphere2


def generate_disk(shape, radius, anti_aliasing=5):
    shape = np.array(shape)
    radius = int(radius)
    x = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    y = np.linspace(-radius, radius, (radius * 2 + 1) * anti_aliasing)
    xx, yy = np.meshgrid(x, y)
    a = (xx**2 + yy**2 <= radius**2).astype('float')
    res = np.zeros(shape * anti_aliasing)
    center_res = (np.array(res.shape) / 2).astype('int')
    res[center_res[0] - int(a.shape[0] / 2):center_res[0] + int(a.shape[0] / 2),
        center_res[1] - int(a.shape[0] / 2):center_res[1] + int(a.shape[0] / 2)] = a
    res = gaussian_filter(res, 0.5 * anti_aliasing)
    res = res[::anti_aliasing, ::anti_aliasing]
    return res


def generate_ring(shape, radius, anti_aliasing=5):

    disk1 = generate_disk(shape, radius + 0.5, anti_aliasing=anti_aliasing)
    disk2 = generate_disk(shape, radius - 0.5, anti_aliasing=anti_aliasing)
    return disk1 - disk2


def fourier_shell_correlation(obj, ref, step_size=1, save_path='fsc', save_mask=True):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    radius_max = int(min(obj.shape) / 2)
    f_obj = np.fft.fftshift(fftn(obj))
    f_ref = np.fft.fftshift(fftn(ref))
    f_prod = f_obj * np.conjugate(f_ref)
    f_obj_2 = np.real(f_obj * np.conjugate(f_obj))
    f_ref_2 = np.real(f_ref * np.conjugate(f_ref))
    radius_ls = np.arange(1, radius_max, step_size)
    fsc_ls = []
    np.save(os.path.join(save_path, 'radii.npy'), radius_ls)

    for rad in radius_ls:
        print(rad)
        if os.path.exists(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad)))):
            mask = dxchange.read_tiff(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))))
        else:
            mask = generate_shell(obj.shape, rad, anti_aliasing=2)
            if save_mask:
                dxchange.write_tiff(mask, os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))),
                                    dtype='float32', overwrite=True)
        fsc = abs(np.sum(f_prod * mask))
        fsc /= np.sqrt(np.sum(f_obj_2 * mask) * np.sum(f_ref_2 * mask))
        fsc_ls.append(fsc)
        np.save(os.path.join(save_path, 'fsc.npy'), fsc_ls)

    matplotlib.rcParams['pdf.fonttype'] = 'truetype'
    fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
    plt.rc('font', **fontProperties)
    plt.plot(radius_ls.astype(float) / radius_ls[-1], fsc_ls)
    plt.xlabel('Spatial frequency (1 / Nyquist)')
    plt.ylabel('FSC')
    plt.savefig(os.path.join(save_path, 'fsc.pdf'), format='pdf')


def fourier_ring_correlation(obj, ref, step_size=1, save_path='frc', save_mask=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    radius_max = int(min(obj.shape) / 2)
    f_obj = np.fft.fftshift(fft2(obj))
    f_ref = np.fft.fftshift(fft2(ref))
    f_prod = f_obj * np.conjugate(f_ref)
    f_obj_2 = np.real(f_obj * np.conjugate(f_obj))
    f_ref_2 = np.real(f_ref * np.conjugate(f_ref))
    radius_ls = np.arange(1, radius_max, step_size)
    fsc_ls = []
    np.save(os.path.join(save_path, 'radii.npy'), radius_ls)

    for rad in radius_ls:
        print(rad)
        if os.path.exists(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad)))):
            mask = dxchange.read_tiff(os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))))
        else:
            mask = generate_ring(obj.shape, rad, anti_aliasing=2)
            if save_mask:
                dxchange.write_tiff(mask, os.path.join(save_path, 'mask_rad_{:04d}.tiff'.format(int(rad))),
                                    dtype='float32', overwrite=True)
        fsc = abs(np.sum(f_prod * mask))
        fsc /= np.sqrt(np.sum(f_obj_2 * mask) * np.sum(f_ref_2 * mask))
        fsc_ls.append(fsc)
        np.save(os.path.join(save_path, 'fsc.npy'), fsc_ls)

    matplotlib.rcParams['pdf.fonttype'] = 'truetype'
    fontProperties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}
    plt.rc('font', **fontProperties)
    plt.plot(radius_ls.astype(float) / radius_ls[-1], fsc_ls)
    plt.xlabel('Spatial frequency (1 / Nyquist)')
    plt.ylabel('FRC')
    plt.savefig(os.path.join(save_path, 'frc.pdf'), format='pdf')


def upsample_2x(arr):

    if arr.ndim == 4:
        out_arr = np.zeros([arr.shape[0] * 2, arr.shape[1] * 2, arr.shape[2] * 2, arr.shape[3]])
        for i in range(arr.shape[3]):
            out_arr[:, :, :, i] = upsample_2x(arr[:, :, :, i])
    else:
        out_arr = np.zeros([arr.shape[0] * 2, arr.shape[1] * 2, arr.shape[2] * 2])
        out_arr[::2, ::2, ::2] = arr[:, :, :]
        out_arr = gaussian_filter(out_arr, 1)
    return out_arr


def print_flush(a, designate_rank=None, this_rank=None):

    if designate_rank is not None:
        if this_rank == designate_rank:
            print(a)
    else:
        print(a)
    sys.stdout.flush()
    return


def real_imag_to_mag_phase(realpart, imagpart):

    a = realpart + 1j * imagpart
    return np.abs(a), np.angle(a)


def mag_phase_to_real_imag(mag, phase):

    a = mag * np.exp(1j * phase)
    return a.real, a.imag


def create_probe_initial_guess(data_fname, dist_nm, energy_ev, psize_nm):

    f = h5py.File(data_fname, 'r')
    dat = f['exchange/data'][...]
    # NOTE: this is for toy model
    wavefront = np.mean(np.abs(dat), axis=0)
    lmbda_nm = 1.24 / energy_ev
    h = get_kernel(-dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
    wavefront = np.fft.fftshift(np.fft.fft2(wavefront)) * h
    wavefront = np.fft.ifft2(np.fft.ifftshift(wavefront))
    return wavefront


def create_probe_initial_guess_ptycho(data_fname, noise=True):

    f = h5py.File(data_fname, 'r')
    dat = f['exchange/data'][...]
    wavefront = np.mean(np.abs(dat), axis=(0, 1))
    wavefront = abs(np.fft.ifftshift(np.fft.ifft2(wavefront)))
    if noise:
        wavefront_mean = np.mean(wavefront)
        wavefront += np.random.normal(size=wavefront.shape, loc=wavefront_mean, scale=wavefront_mean * 0.2)
        wavefront = np.clip(wavefront, 0, None)
    return wavefront


def multidistance_ctf(prj_ls, dist_cm_ls, psize_cm, energy_kev, kappa=50, sigma_cut=0.01, alpha_1=5e-4, alpha_2=1e-16):

    prj_ls = np.array(prj_ls)
    dist_cm_ls = np.array(dist_cm_ls)
    dist_nm_ls = dist_cm_ls * 1.e7
    lmbda_nm = 1.24 / energy_kev
    psize_nm = psize_cm * 1.e7
    prj_shape = prj_ls.shape[1:]

    u_max = 1. / (2. * psize_nm)
    v_max = 1. / (2. * psize_nm)
    u, v = gen_mesh([v_max, u_max], prj_shape)
    xi_mesh = PI * lmbda_nm * (u ** 2 + v ** 2)
    xi_ls = np.zeros([len(dist_cm_ls), *prj_shape])
    for i in range(len(dist_cm_ls)):
        xi_ls[i] = xi_mesh * dist_nm_ls[i]

    abs_nu = np.sqrt(u ** 2 + v ** 2)
    nu_cut = 0.6 * u_max
    f = 0.5 * (1 - erf((abs_nu - nu_cut) / sigma_cut))
    alpha = alpha_1 * f + alpha_2 * (1 - f)
    phase = np.sum(np.fft.fftshift(np.fft.fft2(prj_ls - 1, axes=(-2, -1)), axes=(-2, -1)) * (np.sin(xi_ls) + 1. / kappa * np.cos(xi_ls)), axis=0)
    phase /= (np.sum(2 * (np.sin(xi_ls) + 1. / kappa * np.cos(xi_ls)) ** 2, axis=0) + alpha)
    phase = np.fft.ifft2(np.fft.ifftshift(phase, axes=(-2, -1)), axes=(-2, -1))

    return np.abs(phase)


def split_tasks(arr, split_size):
    res = []
    ind = 0
    while ind < len(arr):
        res.append(arr[ind:min(ind + split_size, len(arr))])
        ind += split_size
    return res


def apply_gradient_adam(x, g, i_batch, m=None, v=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-8):

    g = np.array(g)
    if m is None or v is None:
        m = np.zeros_like(x)
        v = np.zeros_like(v)
    m = (1 - b1) * g + b1 * m  # First moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i_batch + 1))
    x = x - step_size * mhat / (np.sqrt(vhat) + eps)
    return x, m, v


def apply_gradient_gd(x, g, step_size=0.001):

    g = np.array(g)
    x = x - step_size * g
    return x


def get_block_division(original_grid_shape, n_ranks):
    # Must satisfy:
    # 1. n_block_x * n_block_y = n_ranks
    # 2. block_size[0] * n_block_y = original_grid_shape[0]
    # 3. block_size[1] * n_block_x = original_grid_shape[1]
    # 4. At most 1 block per rank
    n_blocks_y = int(np.round(np.sqrt(original_grid_shape[0] / original_grid_shape[1] * n_ranks)))
    n_blocks_x = int(np.round(np.sqrt(original_grid_shape[1] / original_grid_shape[0] * n_ranks)))
    n_blocks = n_blocks_x * n_blocks_y
    block_size = ceil(max([original_grid_shape[0] / n_blocks_y, original_grid_shape[1] / n_blocks_x]))

    while n_blocks > n_ranks:
        if n_blocks_y * block_size - original_grid_shape[0] > n_blocks_x * block_size - original_grid_shape[1]:
            n_blocks_y -= 1
        else:
            n_blocks_x -= 1
        n_blocks = n_blocks_x * n_blocks_y
    # Reiterate for adjusted block arrangement.
    block_size = ceil(max([original_grid_shape[0] / n_blocks_y, original_grid_shape[1] / n_blocks_x]))
    return n_blocks_y, n_blocks_x, n_blocks, block_size


def get_block_range(i_pos, n_blocks_x, block_size):

    line_st = i_pos // n_blocks_x * block_size
    line_end = line_st + block_size
    px_st = i_pos % n_blocks_x * block_size
    px_end = px_st + block_size
    center_y = (line_st + line_end) / 2
    center_x = (px_st + px_end) / 2
    return line_st, line_end, px_st, px_end, center_y, center_x
