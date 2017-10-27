import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from xdesign.propagation import get_kernel_tf_real, get_kernel
from xdesign.constants import *
from xdesign.acquisition import Simulator
import sys


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


grid_delta = np.load('data/sav/grid/64/grid_delta.npy')
grid_beta = np.load('data/sav/grid/64/grid_beta.npy')

sim = Simulator(energy=5000,
                grid=(grid_delta, grid_beta),
                psize=[1e-7, 1e-7, 1e-7])

sim.initialize_wavefront('plane')
wavefront = sim.wavefront
wavefront = tf.convert_to_tensor(wavefront, dtype=tf.complex64, name='wavefront')
# wavefront = tf.reshape(wavefront, [1, wavefront.shape[0].value, wavefront.shape[1].value, 1])

n_slice = grid_delta.shape[-1]

delta_nm = sim.voxel_nm[-1]
kernel = get_kernel(sim, delta_nm * 1.e-7)
# kernel = get_kernel_tf_real(sim, delta_nm * 1.e-7)
# h = tf.convert_to_tensor(kernel, dtype=tf.complex64, name='kernel')
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

sess = tf.Session()
sess.run(tf.global_variables_initializer())
wavefront = sess.run(wavefront)

r = np.abs(wavefront)
plt.imshow(r)
plt.show()
