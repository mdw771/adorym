import xdesign
from xdesign import Simulator
import numpy as np
import h5py
import tomopy
import dxchange
from scipy.ndimage import rotate

grid_delta = np.load('cone_256_filled/phantom/grid_delta.npy')
grid_beta = np.load('cone_256_filled/phantom/grid_beta.npy')

sim = Simulator(energy=5000,
                grid=(grid_delta, grid_beta),
                psize=[1.e-7, 1.e-7, 1.e-7])

f = h5py.File('cone_256_filled/data_cone_256_1nm_1um.h5', 'r')
dat_exit = f['exchange/data'][...]
dat_retrieved = np.zeros_like(dat_exit, dtype='float32')

dxchange.write_tiff_stack(np.abs(dat_exit) ** 2, 'cone_256_filled/conventional_recon/bp0/wave', overwrite=True, dtype='float32')

for i in range(dat_retrieved.shape[0]):
    print(i)
    wave = xdesign.free_propagate(sim, dat_exit[i], dist=-sim.size_nm[2] * 1.e-7)
    wave = np.abs(wave) ** 2
    dat_retrieved[i] = wave
dxchange.write_tiff_stack(dat_retrieved, 'cone_256_filled/conventional_recon/bp/wave', overwrite=True, dtype='float32')

print(dat_retrieved.dtype)
print(dat_retrieved.max(), dat_retrieved.min())
dat_retrieved = tomopy.retrieve_phase(dat_retrieved,
                                      pixel_size=sim.voxel_nm[2] * 1.e-7,
                                      dist=1.e-4,
                                      energy=5,
                                      alpha=1e-3)
dxchange.write_tiff_stack(dat_retrieved, 'cone_256_filled/conventional_recon/pr/proj', overwrite=True, dtype='float32')
theta = np.linspace(0, 2 * np.pi, dat_retrieved.shape[0])
dat_retrieved /= dat_retrieved.max()
dat_retrieved = tomopy.minus_log(dat_retrieved)
rec = tomopy.recon(dat_retrieved,
                   theta=theta,
                   algorithm='gridrec')
rec = rotate(rec, -90, axes=(1, 2), reshape=False)
dxchange.write_tiff_stack(tomopy.circ_mask(rec, 0, ratio=0.9), 'cone_256_filled/conventional_recon/recon/recon', overwrite=True, dtype='float32')
