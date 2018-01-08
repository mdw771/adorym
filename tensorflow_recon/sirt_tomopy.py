import tomopy
import numpy as np
import dxchange
from util import *
import time
import os


PI = 3.1415927

# ============================================
theta_st = 0
theta_end = PI
n_epochs = 200
sino_range = (600, 601, 1)
center = 958
# ============================================


def reconstrct(fname, sino_range, theta_st=0, theta_end=PI, n_epochs=200,
               output_folder=None, downsample=None, center=None):

    if output_folder is None:
        output_folder = 'sift_niter_{}_ds_{}_{}_{}'.format(n_epochs, *downsample)

    t0 = time.time()
    print('Reading data...')
    prj, flt, drk, _ = dxchange.read_aps_32id(fname, sino=sino_range)
    print('Data reading: {} s'.format(time.time() - t0))
    print('Data shape: {}'.format(prj.shape))
    prj = tomopy.normalize(prj, flt, drk)
    prj = preprocess(prj)
    # scale up to prevent precision issue
    prj *= 1.e2

    if downsample is not None:
        prj = tomopy.downsample(prj, level=downsample[0], axis=0)
        prj = tomopy.downsample(prj, level=downsample[1], axis=1)
        prj = tomopy.downsample(prj, level=downsample[2], axis=2)
        print('Downsampled shape: {}'.format(prj.shape))

    n_theta = prj.shape[0]
    theta = np.linspace(theta_st, theta_end, n_theta)

    print('Starting reconstruction...')
    res = tomopy.recon(prj, theta, center=center, algorithm='sirt', num_iter=n_epochs)
    dxchange.write_tiff_stack(res[:, :, :, 0], fname=os.path.join(output_folder, 'recon'), dtype='float32',
                              overwrite=True)


if __name__ == '__main__':

    reconstrct(fname='data.h5',
               sino_range=sino_range,
               n_epochs=200,
               downsample=(0, 0, 0),
               center=center)