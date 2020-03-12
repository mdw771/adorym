import xdesign
from xdesign.propagation import *
from xdesign.plot import *
from xdesign.acquisition import Simulator
import h5py
from scipy.ndimage.interpolation import rotate
import numpy as np
import dxchange
import matplotlib.pyplot as plt
import sys


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_theta = 500
img_dim = 256
alpha = 1.e-4
# ============================================


# read model
grid_delta = np.load('cell/phantom/grid_delta.npy')
grid_beta = np.load('cell/phantom/grid_beta.npy')

# list of angles
theta_ls = -np.linspace(theta_st, theta_end, n_theta)

# create data file
# f = h5py.File('test', 'w')
# grp = f.create_group('exchange')
# dat = grp.create_dataset('data', shape=(n_theta, grid_delta.shape[0], grid_delta.shape[1]), dtype=np.complex64)

for i, theta in enumerate(theta_ls):

    print(i)
    this_delta = rotate(grid_delta, np.rad2deg(theta), axes=(1, 2), reshape=False)
    this_beta = rotate(grid_beta, np.rad2deg(theta), axes=(1, 2), reshape=False)

    sim = Simulator(energy=5000,
                    grid=(this_delta, this_beta),
                    psize=[1e-7, 1e-7, 1e-7])

    sim.initialize_wavefront('plane')
    wavefront = sim.multislice_propagate(free_prop_dist=None)
    wavefront = abs(wavefront)
    plt.imshow(wavefront)
    plt.show()
    sys.exit()


    # dat[i, :, :] = wavefront
    # dxchange.write_tiff(np.abs(wavefront), 'diffraction_dat_tf/mag_{:05d}'.format(i), overwrite=True, dtype='float32')
#
# f.close()