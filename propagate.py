import autograd.numpy as np
import dxchange
import h5py
import matplotlib.pyplot as plt
import matplotlib
import warnings
import datetime
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
from util import *


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
        H = np.fft.fftshift(np.fft.fft2(h)) * voxel_nm[0] * voxel_nm[1]
        # dxchange.write_tiff(x, '2d_512/monitor_output/x', dtype='float32', overwrite=True)
    except:
        h = tf.exp(1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * tf.exp(1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
        # h = tf.convert_to_tensor(h, dtype='complex64')
        H = np.fft.fftshift(np.fft.fft2(h)) * voxel_nm[0] * voxel_nm[1]

    return H


def multislice_propagate_batch_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm,
                                     free_prop_cm=None, obj_batch_shape=None, kernel=None, fresnel_approx=True,
                                     pure_projection=False, binning=1):

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
        h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=fresnel_approx)

    for i in range(n_slice):
        if i % binning == 0:
            i_bin = 0
            delta_slice = np.zeros_like(wavefront)
            beta_slice = np.zeros_like(wavefront)
        delta_slice += grid_delta_batch[:, :, :, i]
        beta_slice += grid_beta_batch[:, :, :, i]
        i_bin += 1

        if i_bin == binning or i == n_slice - 1:
            k1 = 2. * PI * delta_nm * i_bin / lmbda_nm
            c = exp_j(k1 * delta_slice) * np.exp(-k1 * beta_slice)
            wavefront = wavefront * c
            if i < n_slice - 1 and not pure_projection:
                if i_bin == binning:
                    wavefront = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
                else:
                    h1 = get_kernel(delta_nm * i_bin, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=fresnel_approx)
                    wavefront = np.fft.ifft2(
                        np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(wavefront), axes=[1, 2]) * h1, axes=[1, 2]))

    if free_prop_cm not in [0, None]:
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
