try:
    from pyfftw.interfaces.numpy_fft import fft2, ifft2
    from pyfftw.interfaces.numpy_fft import fftshift as np_fftshift
    from pyfftw.interfaces.numpy_fft import ifftshift as np_ifftshift
except:
    from numpy.fft import fft2, ifft2
    from numpy.fft import fftshift as np_fftshift
    from numpy.fft import ifftshift as np_ifftshift
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from util import get_kernel, get_kernel_ir
from constants import *
import dxchange


def multislice_propagate_batch_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm, free_prop_cm=None, obj_batch_shape=None):

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

    h = get_kernel(delta_nm, lmbda_nm, voxel_nm, grid_shape)
    k = 2. * PI * delta_nm / lmbda_nm

    for i in range(n_slice):
        delta_slice = grid_delta_batch[:, :, :, i]
        beta_slice = grid_beta_batch[:, :, :, i]
        c = np.exp(1j * k * delta_slice) * np.exp(-k * beta_slice)
        wavefront = wavefront * c
        if i < n_slice - 1:
            wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))

    if free_prop_cm is not None:
        dxchange.write_tiff(abs(wavefront), '2d_1024/monitor_output/wv', dtype='float32', overwrite=True)
        if free_prop_cm == 'inf':
            wavefront = np_fftshift(fft2(wavefront), axes=[1, 2])
        else:
            dist_nm = free_prop_cm * 1e7
            l = np.prod(size_nm)**(1. / 3)
            crit_samp = lmbda_nm * dist_nm / l
            algorithm = 'TF' if mean_voxel_nm > crit_samp else 'IR'
            # print(algorithm)
            algorithm = 'TF'
            if algorithm == 'TF':
                h = get_kernel(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            else:
                h = get_kernel_ir(dist_nm, lmbda_nm, voxel_nm, grid_shape)
                wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront), axes=[1, 2]) * h, axes=[1, 2]))
            # dxchange.write_tiff(abs(wavefront), '2d_512/monitor_output/wv', dtype='float32', overwrite=True)
            # dxchange.write_tiff(np.angle(h), '2d_512/monitor_output/h', dtype='float32', overwrite=True)

    return wavefront


def multislice_propagate_spherical_numpy(grid_delta_batch, grid_beta_batch, probe_real, probe_imag, energy_ev, psize_cm,
                                         dist_to_source_cm, det_psize_cm, theta_max=PI/18, phi_max=PI/18, free_prop_cm=None,
                                         obj_batch_shape=None):

    batch_size = obj_batch_shape[0]
    grid_shape = obj_batch_shape[1:]
    probe_size = probe_real.shape
    voxel_nm = np.array([psize_cm] * 3) * 1.e7
    dist_to_source_nm = dist_to_source_cm * 1e7
    delta_nm = voxel_nm[-1]
    grid_delta_sph_batch = []
    grid_beta_sph_batch = []
    for i_batch in range(batch_size):
        this_sph_batch, _ = cartesian_to_spherical_numpy(grid_delta_batch[i_batch], dist_to_source_nm, delta_nm, theta_max=theta_max, phi_max=phi_max)
        grid_delta_sph_batch.append(this_sph_batch)
        this_sph_batch, _ = cartesian_to_spherical_numpy(grid_beta_batch[i_batch], dist_to_source_nm, delta_nm, theta_max=theta_max, phi_max=phi_max)
        grid_beta_sph_batch.append(this_sph_batch)
    grid_delta_sph_batch = np.array(grid_delta_sph_batch)
    grid_beta_sph_batch = np.array(grid_beta_sph_batch)

    wavefront = np.zeros([batch_size, *probe_size], dtype=np.complex128)
    wavefront += probe_real + 1j * probe_imag
    lmbda_nm = 1240. / energy_ev
    n_slice = obj_batch_shape[-1]

    def slice_modify(delta_slice, beta_slice, wavefront, delta_r_nm, wavelen_nm):

        kz = 2 * PI * delta_r_nm / wavelen_nm
        wavefront *= np.exp((kz * delta_slice) * 1j) * np.exp(-kz * beta_slice)
        return wavefront

    for i_slice in range(n_slice):
        delta_slice = grid_delta_sph_batch[:, :, :, i_slice]
        beta_slice = grid_beta_sph_batch[:, :, :, i_slice]
        wavefront = slice_modify(delta_slice, beta_slice, wavefront, delta_nm, lmbda_nm)
        wavefront = free_propagate_spherical_numpy(wavefront, delta_nm * 1e-7, dist_to_source_cm + (i_slice * delta_nm) * 1.e-7,
                                                   lmbda_nm, probe_size, theta_max=theta_max, phi_max=phi_max)

    r_nm = dist_to_source_nm + delta_nm * n_slice
    if free_prop_cm is not None:
        free_prop_nm = free_prop_cm * 1e7
        wavefront = free_propagate_spherical_numpy(wavefront, free_prop_cm, r_nm * 1e-7,
                                                   lmbda_nm, probe_size, theta_max=theta_max, phi_max=phi_max)
        r_nm += free_prop_nm

    # map back to planar space
    wavefront_batch = []
    for i_batch in range(batch_size):
        this_wavefront = get_wavefront_on_plane_numpy(wavefront[i_batch], r_nm, probe_size, delta_nm, energy_ev, det_psize_cm * 1e7, theta_max, phi_max)
        wavefront_batch.append(this_wavefront)
    wavefront = np.array(wavefront_batch)

    return wavefront


def free_propagate_spherical_numpy(wavefront, dist_cm, r_cm, wavelen_nm, probe_size, theta_max=PI/18, phi_max=PI/18):

    dist_nm = dist_cm * 1.e7
    r_nm = r_cm * 1.e7
    k_theta = PI / theta_max * (np.arange(probe_size[0]) - float(probe_size[0] - 1) / 2)
    k_phi = PI / phi_max * (np.arange(probe_size[1]) - float(probe_size[1] - 1) / 2)
    k_phi, k_theta = np.meshgrid(k_phi, k_theta)
    k = 2 * PI / wavelen_nm
    wavefront = np_fftshift(fft2(wavefront), axes=[-1, -2])
    wavefront *= np.exp(-1j / (2 * k) * (k_theta ** 2 + k_phi ** 2) * (1. / (r_nm + dist_nm) - 1. / r_nm))
    wavefront = ifft2(np_ifftshift(wavefront, axes=[-1, -2]))
    return wavefront


def get_wavefront_on_plane_numpy(wavefront_sph, r_nm, detector_size, delta_r_nm, energy_ev, det_psize_nm, theta_max=PI/18, phi_max=PI/18):

    lmbda_nm = 1240. / energy_ev
    x_ind, y_ind = [np.arange(detector_size[1], dtype='int'),
                    np.arange(detector_size[0], dtype='int')]
    x_true = (x_ind - np.median(x_ind)) * det_psize_nm
    y_true = (y_ind - np.median(y_ind)) * det_psize_nm
    x_true_mesh, y_true_mesh = np.meshgrid(x_true, y_true)
    z_true = r_nm
    r_interp_mesh = np.sqrt(x_true_mesh ** 2 + y_true_mesh ** 2 + z_true ** 2)
    theta_interp_mesh = -np.arccos(y_true_mesh / r_interp_mesh) + PI / 2
    phi_interp_mesh = np.arctan(x_true_mesh / z_true)
    sph_wave_array = [wavefront_sph]
    r_current = r_nm
    while r_current < r_interp_mesh.max():
        r_current += delta_r_nm
        wavefront_sph = free_propagate_spherical_numpy(wavefront_sph, delta_r_nm * 1e-7, r_current * 1.e-7, lmbda_nm,
                                                       detector_size, theta_max=theta_max, phi_max=phi_max)
        sph_wave_array.append(wavefront_sph)
    sph_wave_array = np.array(sph_wave_array)
    sph_wave_array = np.transpose(sph_wave_array, [1, 2, 0])
    r_ind, theta_ind, phi_ind = [np.arange(sph_wave_array.shape[2], dtype=int),
                                 np.arange(sph_wave_array.shape[0], dtype=int),
                                 np.arange(sph_wave_array.shape[1], dtype=int)]
    r_true = r_ind * delta_r_nm + r_nm
    theta_true = (theta_ind - np.median(theta_ind)) * (2 * theta_max / (theta_ind.size - 1))
    phi_true = (phi_ind - np.median(phi_ind)) * (2 * phi_max / (phi_ind.size - 1))
    sph_interp = RegularGridInterpolator((theta_true, phi_true, r_true),
                                          sph_wave_array, bounds_error=False, fill_value=None)
    coords_interp = np.vstack([theta_interp_mesh.flatten(),
                               phi_interp_mesh.flatten(), r_interp_mesh.flatten()]).transpose()
    dat_interp = sph_interp(coords_interp)
    wavefront_pla = dat_interp.reshape([256, 256])
    return wavefront_pla


def cartesian_to_spherical_numpy(arr, dist_to_source_nm, psize_nm, theta_max=PI/18, phi_max=PI/18):

    x_ind, y_ind, z_ind = [np.arange(arr.shape[0], dtype=int),
                           np.arange(arr.shape[1], dtype=int),
                           np.arange(arr.shape[2], dtype=int)]
    x_true, y_true, z_true = [(x_ind - np.median(x_ind)) * psize_nm,
                              (y_ind - np.median(y_ind)) * psize_nm,
                              z_ind * psize_nm]
    cart_interp = RegularGridInterpolator((x_true, y_true, z_true), arr, bounds_error=False, fill_value=0)
    theta_ind, phi_ind, r_ind = [np.arange(arr.shape[0], dtype=int),
                                 np.arange(arr.shape[1], dtype=int),
                                 np.arange(arr.shape[2], dtype=int)]
    r_true = r_ind * psize_nm + dist_to_source_nm
    theta_true = (theta_ind - np.median(theta_ind)) * (2 * theta_max / (theta_ind.size - 1))
    phi_true = (phi_ind - np.median(phi_ind)) * (2 * phi_max / (phi_ind.size - 1))
    phi, theta, r = np.meshgrid(phi_true, theta_true, r_true)
    x_interp = r * np.sin(theta)
    y_interp = r * np.cos(theta) * np.sin(phi)
    z_interp = r * np.cos(theta) * np.cos(phi)
    z_interp -= dist_to_source_nm
    x_interp /= psize_nm
    y_interp /= psize_nm
    z_interp /= psize_nm
    coords_interp = np.vstack([x_interp.flatten(), y_interp.flatten(), z_interp.flatten()]).transpose()
    dat_interp = cart_interp(coords_interp)
    # phi_ind_mesh, theta_ind_mesh, r_ind_mesh = np.meshgrid(phi_ind, theta_ind, r_ind)
    # arr_sph = np.zeros_like(arr)
    # arr_sph[theta_ind_mesh.flatten(), phi_ind_mesh.flatten(), r_ind_mesh.flatten()] = dat_interp
    arr_sph = dat_interp.reshape(arr.shape)

    return arr_sph, (r_true, theta_true, phi_true)


def fresnel_propagate_numpy(wavefront, energy_ev, psize_cm, dist_cm):

    lmbda_nm = 1240. / energy_ev
    lmbda_cm = 0.000124 / energy_ev
    psize_nm = psize_cm * 1e7
    dist_nm = dist_cm * 1e7
    if dist_cm == 'inf':
        wavefront = np_fftshift(fft2(wavefront))
    else:
        n = np.mean(wavefront.shape)
        z_crit_cm = (psize_cm * n) ** 2 / (lmbda_cm * n)
        algorithm = 'TF' if dist_cm < z_crit_cm else 'IR'
        algorithm = 'TF'
        if algorithm == 'TF':
            h = get_kernel(dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
            wavefront = ifft2(np_ifftshift(np_fftshift(fft2(wavefront)) * h))
        else:
            h = get_kernel_ir(dist_nm, lmbda_nm, [psize_nm, psize_nm], wavefront.shape)
            wavefront = np_ifftshift(ifft2(np_fftshift(fft2(wavefront)) * h))

    return wavefront

