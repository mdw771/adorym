import h5py
import numpy as np
import tomopy
import dxchange
import time, os
from scipy.ndimage import gaussian_filter


f = h5py.File('adhesin/data_adhesin_360_unity.h5', 'r')
dat = f['exchange/data'][...]
dat = np.abs(dat) ** 2
dat = (dat - dat.min()) / (dat.max() - dat.min())

dat = tomopy.retrieve_phase(dat,
                            pixel_size=0.67e-7,
                            dist=32e-7,
                            alpha=1e-3,
                            energy=5)
dxchange.write_tiff(dat, 'adhesin/paganin_obj/pr/pr', dtype='float32', overwrite=True)
dat = tomopy.minus_log(dat)



extra_options = {'MinConstraint': 0}
options = {'num_iter': 200}

t0 = time.time()
rec = tomopy.recon(dat,
                   theta=tomopy.angles(dat.shape[0]),
                   center=32,
                   algorithm='gridrec',)
                   # **options)
print(time.time() - t0)
dxchange.write_tiff_stack(rec, 'adhesin/paganin_obj/recon', dtype='float32', overwrite=True)

rec = gaussian_filter(np.abs(rec), sigma=1, mode='constant')
mask = np.zeros_like(rec)
# mask[rec > 3e-5] = 1
mask[np.abs(rec) > 4e-4] = 1
mask = tomopy.circ_mask(mask, 0, ratio=0.9)
dxchange.write_tiff_stack(mask, os.path.join('adhesin', 'fin_sup_mask/mask'), dtype='float32', overwrite=True)
