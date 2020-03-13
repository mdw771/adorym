import xommons
import dxchange
import numpy as np

grid_delta = np.load('grid_delta.npy')
grid_beta = np.load('grid_beta.npy')

grid_delta = grid_delta[np.newaxis, :, :, :]
grid_beta = grid_beta[np.newaxis, :, :, :]

probe_real = np.ones(grid_delta.shape[1:3])
probe_imag = np.zeros(grid_delta.shape[1:3])

wavefield, wavefield_ls = xommons.multislice_propagate(grid_delta, grid_beta, probe_real, probe_imag, 800, [0.67e-7] * 3, free_prop_cm=None, return_intermediate=True)

dxchange.write_tiff(np.abs(wavefield), 'fresnel_approx_false_orig', dtype='float32')