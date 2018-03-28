import xdesign
from xdesign import Simulator
import numpy as np
import h5py
import tomopy
import dxchange

grid_delta = np.load('adhesin/phantom/grid_delta.npy')
grid_beta = np.load('adhesin/phantom/grid_beta.npy')

sim = Simulator(energy=800,
                grid=(grid_delta, grid_beta),
                psize=[.67e-7, .67e-7, .67e-7])

f = h5py.File('adhesin/data_adhesin_360_soft.h5', 'r')
dat_exit = f['exchange/data'][...]
dat_retrieved = np.zeros_like(dat_exit, dtype='float32')

dxchange.write_tiff_stack(np.abs(dat_exit) ** 2, 'adhesin/conventional_recon/bp0/wave', overwrite=True, dtype='float32')

for i in range(dat_retrieved.shape[0]):
    print(i)
    wave = xdesign.free_propagate(sim, dat_exit[i], dist=-sim.size_nm[2] * 1.e-7)
    wave = np.abs(wave) ** 2
    dat_retrieved[i] = wave
dxchange.write_tiff_stack(dat_retrieved, 'adhesin/conventional_recon/bp/wave', overwrite=True, dtype='float32')

dat_retrieved = tomopy.retrieve_phase(dat_retrieved,
                                      pixel_size=sim.voxel_nm[2] * 1.e-7,
                                      dist=sim.size_nm[2] / 2 * 1.e-7,
                                      energy=5,
                                      alpha=1e-3)
dxchange.write_tiff_stack(dat_retrieved, 'adhesin/conventional_recon/pr/proj', overwrite=True, dtype='float32')
theta = np.linspace(0, 2 * np.pi, dat_retrieved.shape[0])
dat_retrieved /= dat_retrieved.max()
dat_retrieved = tomopy.minus_log(dat_retrieved)
rec = tomopy.recon(dat_retrieved,
                   theta=theta,
                   center=32,
                   algorithm='gridrec')
dxchange.write_tiff_stack(rec, 'adhesin/conventional_recon/recon/recon', overwrite=True, dtype='float32')
