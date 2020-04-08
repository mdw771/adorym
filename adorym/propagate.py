import numpy as np
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

from adorym.constants import *
import adorym.wrappers as w


def gen_mesh(max, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max[0], max[0], shape[0])
    xx = np.linspace(-max[1], max[1], shape[1])
    res = np.meshgrid(xx, yy)
    return res


def gen_freq_mesh(voxel_nm, shape):
    u = np.fft.fftfreq(shape[0])
    v = np.fft.fftfreq(shape[1])
    vv, uu = np.meshgrid(v, u)
    vv /= voxel_nm[1]
    uu /= voxel_nm[0]
    return uu, vv

def get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True, sign_convention=1):
    """Get unshifted Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    u, v = gen_freq_mesh(voxel_nm, grid_shape[0:2])
    if fresnel_approx:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        H = np.exp(-sign_convention * 1j * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    else:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        quad = 1 - lmbda_nm ** 2 * (u**2 + v**2)
        quad_inner = np.clip(quad, a_min=0, a_max=None)
        H = np.exp(sign_convention * 1j * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))

    return H


def get_kernel_wrapped(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True, device=None, sign_convention=1):
    """Get unshifted Fresnel propagation kernel for TF algorithm.

    Parameters:
    -----------
    simulator : :class:`acquisition.Simulator`
        The Simulator object.
    dist : float
        Propagation distance in cm.
    """
    if fresnel_approx:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        h_real, h_imag = w.exp_complex(0., -sign_convention * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    else:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        quad = 1 - lmbda_nm ** 2 * (u**2 + v**2)
        quad_inner = w.clip(quad, 0, None)
        h_real, h_imag  = w.exp_complex(0., sign_convention * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))

    return h_real, h_imag


def get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=1):

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
    h = np.exp(sign_convention * 1j * k * dist_nm) / (1j * lmbda_nm * dist_nm) * np.exp(sign_convention * 1j * k / (2 * dist_nm) * (x ** 2 + y ** 2))
    H = np.fft.fft2(h)

    return H


def multislice_propagate_batch(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm,
                               free_prop_cm=None, obj_batch_shape=None, kernel=None, fresnel_approx=True,
                               pure_projection=False, binning=1, device=None, type='delta_beta',
                               normalize_fft=False, sign_convention=1):

    minibatch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slices = obj_batch_shape[-1]
    delta_nm = voxel_nm[-1]

    if pure_projection:
        k1 = 2. * PI * delta_nm * n_slices / lmbda_nm
        if type == 'delta_beta':
            delta_slice = w.sum(grid_delta_batch, axis=-1)
            beta_slice = w.sum(grid_beta_batch, axis=-1)
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
        elif type == 'real_imag':
            delta_slice = w.prod(grid_delta_batch, axis=-1)
            beta_slice = w.prod(grid_beta_batch, axis=-1)
            c_real, c_imag = delta_slice, beta_slice
        else:
            raise ValueError('unknown_type must be real_imag or delta_beta.')
        probe_real, probe_imag = (probe_real * c_real - probe_imag * c_imag, probe_real * c_imag + probe_imag * c_real)

    else:
        if kernel is not None:
            h = kernel
        else:
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=fresnel_approx, sign_convention=sign_convention)
        h_real, h_imag = np.real(h), np.imag(h)
        h_real = w.create_variable(h_real, requires_grad=False, device=device)
        h_imag = w.create_variable(h_imag, requires_grad=False, device=device)

        for i in range(n_slices):
            # At the start of bin, initialize slice array.
            if i % binning == 0:
                i_bin = 0
                delta_slice = w.zeros([minibatch_size, *grid_shape[:2]], device=device, requires_grad=False)
                beta_slice = w.zeros([minibatch_size, *grid_shape[:2]], device=device, requires_grad=False)
            delta_slice += grid_delta_batch[:, :, :, i]
            beta_slice += grid_beta_batch[:, :, :, i]
            i_bin += 1
            # When arriving at the last slice of bin or object, do propagation.
            if i_bin == binning or i == n_slices - 1:
                k1 = 2. * PI * delta_nm * i_bin / lmbda_nm
                if type == 'delta_beta':
                    # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
                    # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
                    c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
                elif type == 'real_imag':
                    c_real, c_imag = delta_slice, beta_slice
                else:
                    raise ValueError('unknown_type must be delta_beta or real_imag.')
                probe_real, probe_imag = (probe_real * c_real - probe_imag * c_imag, probe_real * c_imag + probe_imag * c_real)
                if i < n_slices - 1:
                    if i_bin == binning:
                        probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_real, h_imag)
                    else:
                        probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, delta_nm * i_bin, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
    if free_prop_cm not in [0, None]:
        if free_prop_cm == 'inf':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            if sign_convention == 1:
                probe_real, probe_imag = w.fft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
            else:
                probe_real, probe_imag = w.ifft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
    return probe_real, probe_imag


def sparse_multislice_propagate_batch(u, v, grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm,
                                      slice_pos_cm_ls, free_prop_cm=None, obj_batch_shape=None, fresnel_approx=True,
                                      device=None, type='delta_beta', normalize_fft=False, sign_convention=1):

    minibatch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm
    slice_pos_nm_ls = slice_pos_cm_ls * 1e7

    n_slices = obj_batch_shape[-1]
    delta_nm = voxel_nm[-1]

    for i in range(n_slices):
        # At the start of bin, initialize slice array.
        delta_slice = grid_delta_batch[:, :, :, i]
        beta_slice = grid_beta_batch[:, :, :, i]

        k1 = 2. * PI * delta_nm / lmbda_nm
        if type == 'delta_beta':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
        elif type == 'real_imag':
            c_real, c_imag = delta_slice, beta_slice
        else:
            raise ValueError('unknown_type must be delta_beta or real_imag.')
        probe_real, probe_imag = (probe_real * c_real - probe_imag * c_imag, probe_real * c_imag + probe_imag * c_real)

        if i < n_slices - 1:
            probe_real, probe_imag = fresnel_propagate_wrapped(u, v, probe_real, probe_imag, slice_pos_nm_ls[i + 1] - slice_pos_nm_ls[i],
                                                               lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)

    if free_prop_cm not in [0, None]:
        if free_prop_cm == 'inf':
            if sign_convention == 1:
                probe_real, probe_imag = w.fft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
            else:
                probe_real, probe_imag = w.ifft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
    return probe_real, probe_imag


def fresnel_propagate(probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm, h=None, device=None, override_backend=None, sign_convention=1):
    """
    :param h: A List of the real part and imaginary part of the transfer function kernel.
    """
    if len(probe_real.shape) == 3:
        grid_shape = probe_real.shape[1:]
    else:
        grid_shape = probe_real.shape
    if h is None:
        h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention)
    h_real = np.real(h)
    h_imag = np.imag(h)
    h_real = w.create_variable(h_real, requires_grad=False, device=device, override_backend=override_backend)
    h_imag = w.create_variable(h_imag, requires_grad=False, device=device, override_backend=override_backend)
    probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_real, h_imag,
                                                               override_backend=override_backend)
    return probe_real, probe_imag


def fresnel_propagate_wrapped(u, v, probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm, h=None, device=None, override_backend=None, sign_convention=1):
    """
    :param h: A List of the real part and imaginary part of the transfer function kernel.
    """
    if len(probe_real.shape) == 3:
        grid_shape = probe_real.shape[1:]
    else:
        grid_shape = probe_real.shape
    if h is None:
        h_real, h_imag = get_kernel_wrapped(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention)
    probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_real, h_imag,
                                                               override_backend=override_backend)
    return probe_real, probe_imag

