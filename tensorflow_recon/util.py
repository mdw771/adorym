import tomopy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
try:
    from xdesign.propagation import get_kernel_tf_real, get_kernel
    from xdesign.constants import *
    from xdesign.acquisition import Simulator
    import sys
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import fourier_shift
except:
    pass


def preprocess(dat, blur=None, normalize_bg=False):

    dat[np.abs(dat) < 2e-3] = 2e-3
    dat[dat > 1] = 1
    if normalize_bg:
        dat = tomopy.normalize_bg(dat)
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
        temp = temp.astype('float32')
    else:
        temp = fourier_shift(np.fft.fftn(arr), shift)
        temp = np.fft.ifftn(temp)
        temp = np.abs(temp).astype('float32')
    return temp


def fftshift(tensor):
    ndim = len(tensor.shape)
    for i in range(ndim):
        n = tensor.shape[i].value
        p2 = (n+1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def ifftshift(tensor):
    ndim = len(tensor.shape)
    for i in range(ndim):
        n = tensor.shape[i].value
        p2 = n - (n + 1) // 2
        begin1 = [0] * ndim
        begin1[i] = p2
        size1 = tensor.shape.as_list()
        size1[i] = size1[i] - p2
        begin2 = [0] * ndim
        size2 = tensor.shape.as_list()
        size2[i] = p2
        t1 = tf.slice(tensor, begin1, size1)
        t2 = tf.slice(tensor, begin2, size2)
        tensor = tf.concat([t1, t2], axis=i)
    return tensor


def multislice_propagate(grid_delta, grid_beta, energy_ev, psize_cm):

    sim = Simulator(energy=energy_ev,
                    grid=(grid_delta, grid_beta),
                    psize=[psize_cm] * 3)

    sim.initialize_wavefront('plane')
    wavefront = sim.wavefront
    wavefront = tf.convert_to_tensor(wavefront, dtype=tf.complex64, name='wavefront')
    # wavefront = tf.reshape(wavefront, [1, wavefront.shape[0].value, wavefront.shape[1].value, 1])

    n_slice = grid_delta.shape[-1]

    delta_nm = sim.voxel_nm[-1]
    kernel = get_kernel(sim, delta_nm * 1.e-7)
    h = tf.convert_to_tensor(kernel, dtype=tf.complex64, name='kernel')
    # h = tf.reshape(h, [h.shape[0].value, h.shape[1].value, 1, 1])
    k = 2 * PI * delta_nm / sim.lmbda_nm

    for i_slice in range(n_slice):
        print('Slice: {:d}'.format(i_slice))
        sys.stdout.flush()
        delta_slice = grid_delta[:, :, i_slice].astype(np.complex64)
        beta_slice = grid_beta[:, :, i_slice].astype(np.complex64)
        c = tf.exp(1j * k * delta_slice * 1j) * tf.exp(-k * beta_slice)
        # c = tf.reshape(c, wavefront.shape)
        wavefront = wavefront * c

        dist_nm = delta_nm
        lmbda_nm = sim.lmbda_nm
        l = np.prod(sim.size_nm)**(1. / 3)
        crit_samp = lmbda_nm * dist_nm / l

        if sim.mean_voxel_nm > crit_samp:
            # wavefront = tf.nn.conv2d(wavefront, h, (1, 1, 1, 1), 'SAME')
            wavefront = tf.ifft2d(ifftshift(fftshift(tf.fft2d(wavefront)) * h))
        else:
            wavefront = tf.fft2d(fftshift(wavefront))
            wavefront = ifftshift(tf.ifft2d(wavefront * h))

    return wavefront