import h5py
import dxchange
import numpy as np


f = h5py.File('data_diff.h5', 'r')
dat = f['exchange/data'].value
dxchange.write_tiff_stack(np.abs(dat), 'diffraction_dat/mag', dtype='float32', overwrite=True)
dxchange.write_tiff_stack(np.angle(dat), 'diffraction_dat/phase', dtype='float32', overwrite=True)
f.close()