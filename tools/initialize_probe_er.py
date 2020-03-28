import adorym
import numpy as np
from pyfftw.interfaces.numpy_fft import fft2, ifft2, fftshift, ifftshift
import dxchange
import h5py
import matplotlib.pyplot as plt

import os, argparse


parser = argparse.ArgumentParser()
parser.add_argument('fname', help='Filename of input HDF5.')
parser.add_argument('--n_epochs', default=100, type=int, help='Number of iterations.')
parser.add_argument('--beta', default=0.8, type=float, help='Weight of magnitude reduction outside finite support.')
parser.add_argument('--mask_radius', default=64, type=int, help='Radius of finite support mask to be used for ER probe retrieval.')
parser.add_argument('--normalize', default=0, type=bool, help='Whether to divide diffraction pattern intensity by probe size.')
parser.add_argument('--raw_data_type', default='intensity', type=str, help='Whether raw data is intensity or magnitude.')
args = parser.parse_args()

fname = args.fname
n_epochs = args.n_epochs
mask_radius = args.mask_radius
raw_data_type = args.raw_data_type
normalize = args.normalize
beta = args.beta

f = h5py.File(fname, 'r')
probe_shape = f['exchange/data'].shape[-2:]
img = np.mean(f['exchange/data'][0], axis=0)
if raw_data_type == 'intensity':
    img = np.sqrt(img)
if normalize:
    img /= np.sqrt(np.prod(probe_shape))

mask = adorym.generate_disk(probe_shape, mask_radius)
beta_mask = np.full_like(mask, -beta)
beta_mask = beta_mask * (1 - mask) + mask
probe = np.random.normal(1, 0.1, probe_shape) + np.exp(1j * np.random.normal(0, 0.1, probe_shape))
probe *= mask

for i_epoch in range(n_epochs):
    f_j = probe
    F = fftshift(fft2(probe))
    this_mse = np.mean(abs(F - img) ** 2)
    F = F / abs(F) * img
    f_jp = ifft2(ifftshift(F))
    probe = (1 - mask) * f_j + beta_mask * f_jp
    print('Epoch {}/{}: MSE = {}.'.format(i_epoch, n_epochs, this_mse))

dxchange.write_tiff(abs(probe), 'guessted_probe_mag', dtype='float32', overwrite=True)
dxchange.write_tiff(np.angle(probe), 'guessted_probe_phase', dtype='float32', overwrite=True)
dxchange.write_tiff(abs(F) ** 2, 'guessted_probe_dp', dtype='float32', overwrite=True)
dxchange.write_tiff(img ** 2, 'guessted_probe_dp_measured', dtype='float32', overwrite=True)
