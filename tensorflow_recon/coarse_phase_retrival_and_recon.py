import h5py
import numpy as np
import tomopy
import dxchange
import time


f = h5py.File('data_diff_tf.h5', 'r')
dat = f['exchange/data'][...]
dat = np.abs(dat) ** 2

dat = tomopy.retrieve_phase(dat,
                            pixel_size=1e-7,
                            dist=32e-7,
                            alpha=1e-2,
                            energy=5)
dxchange.write_tiff(dat[0], 'paganin_obj/pr', dtype='float32', overwrite=True)
dat = tomopy.minus_log(dat)



extra_options = {'MinConstraint': 0}
options = {'num_iter': 200}

t0 = time.time()
rec = tomopy.recon(dat,
                   theta=tomopy.angles(dat.shape[0]),
                   center=32,
                   algorithm='sirt',
                   **options)
print(time.time() - t0)
dxchange.write_tiff_stack(rec, 'paganin_obj/recon', dtype='float32', overwrite=True)