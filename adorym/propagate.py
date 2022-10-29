import time

import numpy as np

import adorym.wrappers as w
from adorym.constants import *
from adorym.misc import *
from adorym.util import *


def gen_mesh(max_val, shape):
    """Generate mesh grid.
    """
    yy = np.linspace(-max_val[0], max_val[0], shape[0])
    xx = np.linspace(-max_val[1], max_val[1], shape[1])
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

    :param dist_nm: propogation distance
    :param lmbda_nm: wavelength
    :param voxel_nm: pixel size
    :param grid_shape: probe shape
    :param u, v: Reciprocal space meshgrids.
    :param dist_nm: Propagation distance in nm.
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
        quad_inner = np.clip(quad, 0, None)
        quad_mask = (quad > 0)
        H = np.exp(sign_convention * 1j * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))
        H = H * quad_mask
    return H


def get_kernel_wrapped(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True, device=None, sign_convention=1):
    """Get unshifted Fresnel propagation kernel for TF algorithm.

    :param u, v: Reciprocal space meshgrids.
    :param dist_nm: Propagation distance in nm.
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
        quad_mask = (quad > 0)
        h_real, h_imag  = w.exp_complex(0., sign_convention * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))
        h_real = h_real * quad_mask
        h_imag = h_imag * quad_mask
    return h_real, h_imag
def get_kernel_wrapped_complex(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=True, device=None, sign_convention=1):
    """Get unshifted Fresnel propagation kernel for TF algorithm.

    :param u, v: Reciprocal space meshgrids.
    :param dist_nm: Propagation distance in nm.
    """
    if fresnel_approx:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        h = w.exp(0. + 1j * -sign_convention * PI * lmbda_nm * dist_nm * (u**2 + v**2))
    else:
        # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
        # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
        quad = 1 - lmbda_nm ** 2 * (u**2 + v**2)
        quad_inner = w.clip(quad, 0, None)
        quad_mask = (quad > 0)
        h = w.exp(0. + 1j * sign_convention * 2 * PI * dist_nm / lmbda_nm * np.sqrt(quad_inner))
        h = h * quad_mask
    return h

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


def multislice_propagate_batch(grid_batch, probe_real, probe_imag, energy_ev, psize_cm, delta_cm=None,
                               free_prop_cm=None, obj_batch_shape=None, kernel=None, fresnel_approx=True,
                               pure_projection=False, binning=1, device=None, unknown_type='delta_beta',
                               normalize_fft=False, sign_convention=1, optimize_free_prop=False, u_free=None, v_free=None,
                               scale_ri_by_k=True, is_minus_logged=False, pure_projection_return_sqrt=False,
                               kappa=None, repeating_slice=None, return_fft_time=False, shift_exit_wave=None,
                               return_intermediate_wavefields=False):

    intermediate_wavefield_real_ls = []
    intermediate_wavefield_imag_ls = []
    minibatch_size = grid_batch.shape[0]
    grid_shape = grid_batch.shape[1:-1]
    if delta_cm is not None:
        voxel_nm = np.array([psize_cm, psize_cm, delta_cm]) * 1.e7
    else:
        voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slices = grid_batch.shape[-2]
    delta_nm = voxel_nm[-1]

    if repeating_slice is not None:
        n_slices = repeating_slice

    if pure_projection:
        k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
        if unknown_type == 'delta_beta':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            p = w.sum(grid_batch, axis=-2)
            delta_slice = p[:, :, :, 0]
            if kappa is not None:
                beta_slice = delta_slice * kappa
            else:
                beta_slice = p[:, :, :, 1]
            # In conventional tomography beta is interpreted as mu. If projection data is minus-logged,
            # the line sum of beta (mu) directly equals image intensity. If raw_data_type is set to 'intensity',
            # measured data will be taken square root at the loss calculation step. To match this, the summed
            # beta must be square-rooted as well. Otherwise, set raw_data_type to 'magnitude' to avoid square-rooting
            # the measured data, and skip sqrt to summed beta here accordingly.
            if is_minus_logged:
                if pure_projection_return_sqrt:
                    c_real, c_imag = w.sqrt(beta_slice + 1e-10), delta_slice * 0
                else:
                    c_real, c_imag = beta_slice, delta_slice * 0
            else:
                c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
        elif unknown_type == 'real_imag':
            p = w.prod(grid_batch, axis=-2)
            delta_slice = p[:, :, :, 0]
            beta_slice = p[:, :, :, 1]
            c_real, c_imag = delta_slice, beta_slice
            if is_minus_logged:
                if pure_projection_return_sqrt:
                    c_real, c_imag = w.sqrt(-w.log(c_real ** 2 + c_imag ** 2) + 1e-10), 0
                else:
                    c_real, c_imag = -w.log(c_real ** 2 + c_imag ** 2), 0
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

        t_tot = 0
        n_steps = int(np.ceil(n_slices / binning))
        for i_step in range(n_steps):
            if return_intermediate_wavefields:
                intermediate_wavefield_real_ls.append(probe_real)
                intermediate_wavefield_imag_ls.append(probe_imag)
            # ==========================================
            # Sampling
            # ==========================================
            k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
            i_slice = i_step * binning
            # At the start of bin, initialize slice array.
            this_step = min([binning, n_slices - i_slice])
            if repeating_slice is None:
                delta_slice = grid_batch[:, :, :, i_slice:i_slice + this_step, 0] if this_step > 1 else grid_batch[:, :, :, i_slice, 0]
            else:
                delta_slice = grid_batch[:, :, :, 0:1, 0]
            if kappa is not None:
                # In sign = +1 convention, phase (delta) should be positive, and kappa is positive too.
                beta_slice = delta_slice * kappa
            else:
                if repeating_slice is None:
                    beta_slice = grid_batch[:, :, :, i_slice:i_slice + this_step, 1] if this_step > 1 else grid_batch[:, :, :, i_slice, 1]
                else:
                    beta_slice = grid_batch[:, :, :, 0:1, 1]
            t0 = time.time()
            # ==========================================
            # Modulation
            # ==========================================
            if unknown_type == 'delta_beta':
                # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
                # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
                if this_step > 1:
                    delta_slice = w.sum(delta_slice, axis=3)
                    beta_slice = w.sum(beta_slice, axis=3)
                c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
            elif unknown_type == 'real_imag':
                if this_step > 1:
                    delta_slice = w.prod(delta_slice, axis=3)
                    beta_slice = w.prod(beta_slice, axis=3)
                c_real, c_imag = delta_slice, beta_slice
            else:
                raise ValueError('unknown_type must be delta_beta or real_imag.')
            probe_real, probe_imag = (probe_real * c_real - probe_imag * c_imag, probe_real * c_imag + probe_imag * c_real)
            # ==========================================
            # When arriving at the last slice of bin or object, do propagation.
            # ==========================================
            if i_step < n_steps - 1:
                if this_step == binning:
                    probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_real, h_imag)
                else:
                    probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, delta_nm * this_step, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
            t_tot += (time.time() - t0)

    if shift_exit_wave is not None:
        probe_real, probe_imag = realign_image_fourier(probe_real, probe_imag, shift_exit_wave, axes=(1, 2), device=device)

    if free_prop_cm not in [0, None]:
        if isinstance(free_prop_cm, str) and free_prop_cm == 'inf':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            if sign_convention == 1:
                probe_real, probe_imag = w.fft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
            else:
                probe_real, probe_imag = w.ifft2_and_shift(probe_real, probe_imag, axes=[1, 2], normalize=normalize_fft)
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            if optimize_free_prop:
                    probe_real, probe_imag = fresnel_propagate_wrapped(u_free, v_free, probe_real, probe_imag, dist_nm,
                                                                       lmbda_nm, voxel_nm,
                                                                       device=device, sign_convention=sign_convention)
            elif not optimize_free_prop:
                probe_real, probe_imag = fresnel_propagate(probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm,
                                                           device=device, sign_convention=sign_convention)
    return_ls = [probe_real, probe_imag]
    if return_fft_time:
        return_ls.append(t_tot)
    if return_intermediate_wavefields:
        # intermediate_wavefield_real_ls = w.stack(intermediate_wavefield_real_ls)
        # intermediate_wavefield_imag_ls = w.stack(intermediate_wavefield_imag_ls)
        return_ls = return_ls + [intermediate_wavefield_real_ls, intermediate_wavefield_imag_ls]
    return return_ls
def multislice_propagate_batch_complex(grid_batch, probe, energy_ev, psize_cm, delta_cm=None,
                               free_prop_cm=None, obj_batch_shape=None, kernel=None, fresnel_approx=True,
                               pure_projection=False, binning=1, device=None, unknown_type='delta_beta',
                               normalize_fft=False, sign_convention=1, optimize_free_prop=False, u_free=None, v_free=None,
                               scale_ri_by_k=True, is_minus_logged=False, pure_projection_return_sqrt=False,
                               kappa=None, repeating_slice=None, return_fft_time=False, shift_exit_wave=None,
                               return_intermediate_wavefields=False):

    intermediate_wavefield_ls = []
    minibatch_size = grid_batch.shape[0]
    grid_shape = grid_batch.shape[1:]
    if delta_cm is not None:
        voxel_nm = np.array([psize_cm, psize_cm, delta_cm]) * 1.e7
    else:
        voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev # TODO - this should be a variable that is in the function definition
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm

    n_slices = grid_batch.shape[-1]
    delta_nm = voxel_nm[-1]

    if repeating_slice is not None:
        n_slices = repeating_slice

    if pure_projection:
        k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
        if unknown_type == 'delta_beta':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            p = w.sum(grid_batch, axis=-1) # this is summing along the slice directino
            delta_slice = w.real(p)
            if kappa is not None:
                beta_slice = delta_slice * kappa
            else:
                beta_slice = w.imag(p)
            # In conventional tomography beta is interpreted as mu. If projection data is minus-logged,
            # the line sum of beta (mu) directly equals image intensity. If raw_data_type is set to 'intensity',
            # measured data will be taken square root at the loss calculation step. To match this, the summed
            # beta must be square-rooted as well. Otherwise, set raw_data_type to 'magnitude' to avoid square-rooting
            # the measured data, and skip sqrt to summed beta here accordingly.
            if is_minus_logged:
                if pure_projection_return_sqrt:
                    c = w.sqrt(beta_slice + 1e-10) + 1j * delta_slice * 0
                else:
                    c = beta_slice + 1j * delta_slice * 0
            else:
                c = w.exp(-k1 * beta_slice + 1j * -sign_convention * k1 * delta_slice)
        elif unknown_type == 'real_imag':
            p = w.prod(grid_batch, axis=-1) # this is multiplying along the slice direction - TODO this might not be ok w/ complex arrays
            c = p[:,:,:]
            if is_minus_logged:
                if pure_projection_return_sqrt:
                    c = w.sqrt(-w.log(w.abs(c)**2) + 1e-10) + 1j * 0
                else:
                    c = -w.log(w.abs(c)**2) + 1j * 0
        else:
            raise ValueError('unknown_type must be real_imag or delta_beta.')
        probe = c * probe

    else:
        if kernel is not None:
            h = kernel
        else:
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            h = get_kernel(delta_nm * binning, lmbda_nm, voxel_nm, grid_shape, fresnel_approx=fresnel_approx, sign_convention=sign_convention)
        h = w.create_variable(h, requires_grad=False, device=device)

        t_tot = 0
        n_steps = int(np.ceil(n_slices / binning))
        for i_step in range(n_steps):
            if return_intermediate_wavefields:
                intermediate_wavefield_ls.append(probe)
            # ==========================================
            # Sampling
            # ==========================================
            k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
            i_slice = i_step * binning
            # At the start of bin, initialize slice array.
            this_step = min([binning, n_slices - i_slice])
            if repeating_slice is None:
                delta_slice = w.real(grid_batch[:, :, :, i_slice:i_slice + this_step]) if this_step > 1 else w.real(grid_batch[:, :, :, i_slice])
            else:
                delta_slice = w.real(grid_batch[:, :, :, 0:1, 0])
            if kappa is not None:
                # In sign = +1 convention, phase (delta) should be positive, and kappa is positive too.
                beta_slice = delta_slice * kappa
            else:
                if repeating_slice is None:
                    beta_slice = w.imag(grid_batch[:, :, :, i_slice:i_slice + this_step]) if this_step > 1 else w.imag(grid_batch[:, :, :, i_slice])
                else:
                    beta_slice = w.imag(grid_batch[:, :, :, 0:1])
            t0 = time.time()
            # ==========================================
            # Modulation
            # ==========================================
            if unknown_type == 'delta_beta':
                # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
                # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
                if this_step > 1:
                    delta_slice = w.sum(delta_slice, axis=3)
                    beta_slice = w.sum(beta_slice, axis=3)
                c = w.exp(-k1 * beta_slice + 1j * -sign_convention * k1 * delta_slice)
            elif unknown_type == 'real_imag':
                if this_step > 1:
                    delta_slice = w.prod(delta_slice, axis=3) # TODO make sure this also works
                    beta_slice = w.prod(beta_slice, axis=3)
                c = delta_slice + 1j * beta_slice
            else:
                raise ValueError('unknown_type must be delta_beta or real_imag.')
            probe = c * probe
            # ==========================================
            # When arriving at the last slice of bin or object, do propagation.
            # ==========================================
            if i_step < n_steps - 1:
                if this_step == binning:
                    probe = w.convolve_with_transfer_function_complex(probe, h)
                else:
                    probe = fresnel_propagate_complex(probe, delta_nm * this_step, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
            t_tot += (time.time() - t0)

    if shift_exit_wave is not None:
        probe = realign_image_fourier_complex(probe, shift_exit_wave, axes=(1, 2), device=device)

    if free_prop_cm not in [0, None]:
        if isinstance(free_prop_cm, str) and free_prop_cm == 'inf':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            if sign_convention == 1:
                probe = w.fft2_and_shift_complex(probe, axes=[1, 2], normalize=normalize_fft)
            else:
                probe = w.ifft2_and_shift_complex(probe, axes=[1, 2], normalize=normalize_fft)
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            if optimize_free_prop:
                    probe = fresnel_propagate_wrapped_complex(u_free, v_free, probe, dist_nm,
                                                                       lmbda_nm, voxel_nm,
                                                                       device=device, sign_convention=sign_convention)
            elif not optimize_free_prop:
                probe = fresnel_propagate_complex(probe, dist_nm, lmbda_nm, voxel_nm,
                                                           device=device, sign_convention=sign_convention)
    return_ls = [probe]
    if return_fft_time:
        return_ls.append(t_tot)
    if return_intermediate_wavefields:
        return_ls = return_ls + [intermediate_wavefield_ls]
    return return_ls


def modulate_and_get_ctf(grid_batch, energy_ev, free_prop_cm, u_free=None, v_free=None, kappa=50.):

    lmbda_nm = 1240. / energy_ev
    dist_nm = free_prop_cm * 1e7

    p = w.sum(grid_batch, axis=-2)
    delta_slice = p[:, :, :, 0]
    beta_slice = p[:, :, :, 1]
    probe_real, probe_imag = pure_phase_ctf(u_free, v_free, delta_slice, beta_slice, dist_nm, lmbda_nm, kappa=kappa)
    return probe_real, probe_imag
def modulate_and_get_ctf_complex(grid_batch, energy_ev, free_prop_cm, u_free=None, v_free=None, kappa=50.):

    lmbda_nm = 1240. / energy_ev
    dist_nm = free_prop_cm * 1e7

    p = w.sum(grid_batch, axis=-2)
    delta_slice = w.real(p)
    beta_slice = w.imag(p)
    probe = pure_phase_ctf_complex(u_free, v_free, delta_slice, beta_slice, dist_nm, lmbda_nm, kappa=kappa)
    return probe


def sparse_multislice_propagate_batch(u, v, grid_batch, probe_real, probe_imag, energy_ev, psize_cm,
                                      slice_pos_cm_ls, free_prop_cm=None, obj_batch_shape=None, fresnel_approx=True,
                                      device=None, unknown_type='delta_beta', normalize_fft=False, sign_convention=1,
                                      scale_ri_by_k=True, shift_exit_wave=None):

    minibatch_size = grid_batch.shape[0]
    grid_shape = grid_batch.shape[1:-1]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm
    slice_pos_nm_ls = slice_pos_cm_ls * 1e7

    n_slices = grid_batch.shape[-2]
    delta_nm = voxel_nm[-1]

    for i in range(n_slices):
        # At the start of bin, initialize slice array.
        delta_slice = grid_batch[:, :, :, i, 0]
        beta_slice = grid_batch[:, :, :, i, 1]

        k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
        if unknown_type == 'delta_beta':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            c_real, c_imag = w.exp_complex(-k1 * beta_slice, -sign_convention * k1 * delta_slice)
        elif unknown_type == 'real_imag':
            c_real, c_imag = delta_slice, beta_slice
        else:
            raise ValueError('unknown_type must be delta_beta or real_imag.')
        probe_real, probe_imag = (probe_real * c_real - probe_imag * c_imag, probe_real * c_imag + probe_imag * c_real)

        if i < n_slices - 1:
            # pr, pi = w.to_numpy(probe_real), w.to_numpy(probe_imag)
            # dxchange.write_tiff(pr ** 2 + pi ** 2, 'debug/probe0', dtype='float32')
            probe_real, probe_imag = fresnel_propagate_wrapped(u, v, probe_real, probe_imag, slice_pos_nm_ls[i + 1] - slice_pos_nm_ls[i],
                                                               lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
            # pr, pi = w.to_numpy(probe_real), w.to_numpy(probe_imag)
            # dxchange.write_tiff(pr ** 2 + pi ** 2, 'debug/probe1', dtype='float32')

    if shift_exit_wave is not None:
        probe_real, probe_imag = realign_image_fourier(probe_real, probe_imag, shift_exit_wave, axes=(1, 2), device=device)

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
def sparse_multislice_propagate_batch_complex(u, v, grid_batch, probe, energy_ev, psize_cm,
                                      slice_pos_cm_ls, free_prop_cm=None, obj_batch_shape=None, fresnel_approx=True,
                                      device=None, unknown_type='delta_beta', normalize_fft=False, sign_convention=1,
                                      scale_ri_by_k=True, shift_exit_wave=None):

    minibatch_size = grid_batch.shape[0]
    grid_shape = grid_batch.shape[1:-1]
    voxel_nm = np.array([psize_cm] * 3) * 1.e7

    lmbda_nm = 1240. / energy_ev
    mean_voxel_nm = np.prod(voxel_nm) ** (1. / 3)
    size_nm = np.array(grid_shape) * voxel_nm
    slice_pos_nm_ls = slice_pos_cm_ls * 1e7

    n_slices = grid_batch.shape[-2]
    delta_nm = voxel_nm[-1]

    for i in range(n_slices):
        # At the start of bin, initialize slice array.
        delta_slice = w.real(grid_batch[:, :, :, i])
        beta_slice = w.imag(grid_batch[:, :, :, i])

        k1 = 2. * PI * delta_nm / lmbda_nm if scale_ri_by_k else 1.
        if unknown_type == 'delta_beta':
            # Use sign_convention = 1 for Goodman convention: exp(ikz); n = 1 - delta + i * beta
            # Use sign_convention = -1 for opposite convention: exp(-ikz); n = 1 - delta - i * beta
            c = w.exp(-k1 * beta_slice + 1j * -sign_convention * k1 * delta_slice)
        elif unknown_type == 'real_imag':
            c = delta_slice + 1j * beta_slice
        else:
            raise ValueError('unknown_type must be delta_beta or real_imag.')
        probe = probe * c

        if i < n_slices - 1:
            # pr, pi = w.to_numpy(probe_real), w.to_numpy(probe_imag)
            # dxchange.write_tiff(pr ** 2 + pi ** 2, 'debug/probe0', dtype='float32')
            probe = fresnel_propagate_wrapped_complex(u, v, probe, slice_pos_nm_ls[i + 1] - slice_pos_nm_ls[i],
                                                               lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
            # pr, pi = w.to_numpy(probe_real), w.to_numpy(probe_imag)
            # dxchange.write_tiff(pr ** 2 + pi ** 2, 'debug/probe1', dtype='float32')

    if shift_exit_wave is not None:
        probe = realign_image_fourier_complex(probe, shift_exit_wave, axes=(1, 2), device=device)

    if free_prop_cm not in [0, None]:
        if free_prop_cm == 'inf':
            if sign_convention == 1:
                probe = w.fft2_and_shift_complex(probe, axes=[1, 2], normalize=normalize_fft)
            else:
                probe = w.ifft2_and_shift_complex(probe, axes=[1, 2], normalize=normalize_fft)
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            probe = fresnel_propagate_complex(probe, dist_nm, lmbda_nm, voxel_nm, device=device, sign_convention=sign_convention)
    return probe


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
def fresnel_propagate_complex(probe, dist_nm, lmbda_nm, voxel_nm, h=None, device=None, override_backend=None, sign_convention=1):
    """
    :param h: the complex transfer function kernel.
    """
    if len(probe.shape) == 3:
        grid_shape = probe.shape[1:]
    else:
        grid_shape = probe.shape
    if h is None:
        h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention)
    h = w.create_variable(h, requires_grad=False, device=device, override_backend=override_backend)
    probe = w.convolve_with_transfer_function_complex(probe, h, override_backend=override_backend)
    return probe


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
def fresnel_propagate_wrapped_complex(u, v, probe, dist_nm, lmbda_nm, voxel_nm, h=None, device=None, override_backend=None, sign_convention=1):
    """
    :param h: A List of the real part and imaginary part of the transfer function kernel.
    """
    if len(probe.shape) == 3:
        grid_shape = probe.shape[1:]
    else:
        grid_shape = probe.shape
    if h is None:
        h = get_kernel_wrapped_complex(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention)
    probe = w.convolve_with_transfer_function_complex(probe, h, override_backend=override_backend)
    return probe


def ctf(u, v, probe_real, probe_imag, dist_nm, lmbda_nm, voxel_nm, h=None,
                                         device=None, override_backend=None, sign_convention=1):
    """
    Calculate the Fourier transform of the wavefront intensity after Fresnel propagation using
    F[I] = [Phi' H] * [Phi H'], where * denotes convolution and ' denotes complex conjugate.
    """
    if len(probe_real.shape) == 3:
        grid_shape = probe_real.shape[1:]
    else:
        grid_shape = probe_real.shape
    probe_real, probe_imag = w.fft2(probe_real, probe_imag, override_backend=override_backend, normalize=True)
    if h is None:
        h_real, h_imag = get_kernel_wrapped(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention, device=device)
    a1_real, a1_imag = w.complex_mul(probe_real, -probe_imag, h_real, h_imag)
    a2_real, a2_imag = w.complex_mul(probe_real, probe_imag, h_real, -h_imag)
    probe_real, probe_imag = w.convolve_with_impulse_response(a1_real, a1_imag, a2_real, a2_imag, override_backend=override_backend, normalize=True)
    return probe_real, probe_imag
def ctf_complex(u, v, probe, dist_nm, lmbda_nm, voxel_nm, h=None,
                                         device=None, override_backend=None, sign_convention=1):
    """
    Calculate the Fourier transform of the wavefront intensity after Fresnel propagation using
    F[I] = [Phi' H] * [Phi H'], where * denotes convolution and ' denotes complex conjugate.
    """
    if len(probe.shape) == 3:
        grid_shape = probe.shape[1:]
    else:
        grid_shape = probe.shape
    probe = w.fft2_complex(probe, override_backend=override_backend, normalize=True)
    if h is None:
        h = get_kernel_wrapped_complex(u, v, dist_nm, lmbda_nm, voxel_nm, grid_shape, sign_convention=sign_convention, device=device)
    a1 = w.conj(probe) *  h
    a2 = probe * w.conj(h)
    probe = w.convolve_with_impulse_response_complex(a1, a2, override_backend=override_backend, normalize=True)
    return probe


def pure_phase_ctf(u, v, delta_slice, beta_slice, dist_nm, lmbda_nm, kappa=50., alpha=1e-10, override_backend=None):

    # Beware: CTF forward model is very sensitive to discontinuity. Delta maps with non-vacuum boundaries can
    # cause the result to blow up, which can't be solved even with edge-mode padding. Vignetting is the only
    # way through. Otherwise, use the result of NON-PADDED CTF phase retrieval (which contains vignetting
    # by itself) as the initial guess.
    print(kappa)
    probe_real, probe_imag = w.fft2(delta_slice, w.zeros_like(delta_slice, requires_grad=False), override_backend=override_backend)
    xi = PI * lmbda_nm * dist_nm * (u ** 2 + v ** 2)
    osc = 2 * (w.sin(xi) + 1. / kappa * w.cos(xi))
    probe_real = osc * probe_real
    probe_imag = osc * probe_imag
    probe_real, probe_imag = w.ifft2(probe_real, probe_imag, override_backend=override_backend)
    probe_real = probe_real + 1
    probe_real = w.sqrt(w.clip(probe_real, 0, None))
    probe_imag = probe_imag * 0
    return probe_real, probe_imag
def pure_phase_ctf_complex(u, v, delta_slice, beta_slice, dist_nm, lmbda_nm, kappa=50., alpha=1e-10, override_backend=None):

    # Beware: CTF forward model is very sensitive to discontinuity. Delta maps with non-vacuum boundaries can
    # cause the result to blow up, which can't be solved even with edge-mode padding. Vignetting is the only
    # way through. Otherwise, use the result of NON-PADDED CTF phase retrieval (which contains vignetting
    # by itself) as the initial guess.
    print(kappa)
    probe = w.fft2_complex(delta_slice + 1j * w.zeros_like(delta_slice, requires_grad=False), override_backend=override_backend)
    xi = PI * lmbda_nm * dist_nm * (u ** 2 + v ** 2)
    osc = 2 * (w.sin(xi) + 1. / kappa * w.cos(xi))
    probe_real = osc * w.real(probe)
    probe_imag = osc * w.imag(probe)
    probe = w.ifft2_complex(probe_real + 1j * probe_imag, override_backend=override_backend)
    probe_real = w.real(probe) + 1
    probe_real = w.sqrt(w.clip(probe_real, 0, None))
    probe_imag = w.imag(probe) * 0
    return probe_real + 1j * probe_imag

