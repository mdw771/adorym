import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import dxchange
from tqdm import trange
import time

np.random.seed(int(time.time()))

src_fname = 'data_nonoise.h5'
n_ph_per_px = 1e2 # Number of photons hitting each pixel of that contains the sample.
n_sample_pixel = 'auto'
dest_fname = 'data_n{:.1e}'.format(n_ph_per_px)
raw_data_type = 'intensity'
is_ptycho = False
ptycho_grad_size = [325, 325] # Size of the scanned area in pixels.


o = h5py.File(src_fname, 'r')['exchange/data']
file_new = h5py.File(dest_fname, 'w')
grp = file_new.create_group('exchange')
n = grp.create_dataset('data', dtype=o.dtype, shape=o.shape)
snr_ls = []

if n_sample_pixel == 'auto':
    n_sample_pixel = o.shape[-2] * o.shape[-1]

if is_ptycho:
    ptycho_grid_size = o.shape[-2:]

    # total photons received by sample
    n_ex = n_ph_per_px * n_sample_pixel
    n_spots = o.shape[1]
    # total photons per image
    print('Far-field ptychography data')
    n_ex *= (np.prod(ptycho_grid_size) / n_sample_pixel)
    print('CHECK IF THIS IS THE CORRECT SCAN SIZE AND SAMPLE AREA:')
    print(ptycho_grid_size, np.prod(ptycho_grid_size), n_sample_pixel)
    time.sleep(3)
    # total photons per spot
    n_ex /= o.shape[1]
    print(o.shape[1])

    for i in trange(o.shape[0]):
        for j in range(o.shape[1]):
            prj_o = o[i, j]
            if raw_data_type == 'intensity':
                prj_o_inten = np.abs(prj_o)
            else:
                prj_o_inten = np.abs(prj_o) ** 2
            spot_integral = np.sum(prj_o_inten)
            multiplier = n_ex / spot_integral
            # scale intensity to match expected photons per spot
            pro_o_inten_scaled = prj_o_inten * multiplier
            # dc_intensity = prj_o_inten[int(o.shape[-2] / 2), int(o.shape[-1] / 2)]
            # prj_o_inten_norm = prj_o_inten / dc_intensity
            # print(n_ph)
            prj_o_inten_noisy = np.random.poisson(pro_o_inten_scaled)
            prj_o_inten_noisy = prj_o_inten_noisy / multiplier
            noise = prj_o_inten_noisy - prj_o_inten
            snr = np.var(prj_o_inten) / np.var(noise)
            snr_ls.append(snr)
            if raw_data_type == 'magnitude':
                data = np.sqrt(prj_o_inten_noisy)
            else:
                data = prj_o_inten_noisy
            n[i] = data

elif 'nf_ptycho' in src_fname:

    print('Near-field ptychography data')
    time.sleep(3)
    print(o.shape)
    n_ph_per_img = n_ph_per_px / o.shape[1]
    for i in range(o.shape[0]):
        for j in trange(o.shape[1]):
            prj_o = o[i, j]
            if raw_data_type == 'intensity':
                prj_o_inten = np.abs(prj_o)
            else:
                prj_o_inten = np.abs(prj_o) ** 2
            prj_o_inten_noisy = np.random.poisson(prj_o_inten * n_ph_per_img)
            # noise = prj_o_inten_noisy - prj_o_inten
            # print(np.var(noise))
            prj_o_inten_noisy = prj_o_inten_noisy / n_ph_per_img
            noise = prj_o_inten_noisy - prj_o_inten
            snr = np.var(prj_o_inten) / np.var(noise)
            snr_ls.append(snr)
            if raw_data_type == 'magnitude':
                data = np.sqrt(prj_o_inten_noisy)
            else:
                data = prj_o_inten_noisy
            n[i] = data
else:
    print('Holography data')
    time.sleep(3)
    for i in trange(o.shape[0]):
        prj_o = o[i]
        if raw_data_type == 'intensity':
            prj_o_inten = np.abs(prj_o)
        else:
            prj_o_inten = np.abs(prj_o) ** 2
        prj_o_inten_noisy = np.random.poisson(prj_o_inten * n_ph_per_px)
        # noise = prj_o_inten_noisy - prj_o_inten
        # print(np.var(noise))
        prj_o_inten_noisy = prj_o_inten_noisy / n_ph_per_px
        noise = prj_o_inten_noisy - prj_o_inten
        snr = np.var(prj_o_inten) / np.var(noise)
        snr_ls.append(snr)
        if raw_data_type == 'magnitude':
            data = np.sqrt(prj_o_inten_noisy)
        else:
            data = prj_o_inten_noisy
        n[i] = data

print('Average SNR is {}.'.format(np.mean(snr_ls)))

dxchange.write_tiff(abs(n[0]), os.path.join(os.path.dirname(dest_fname), dest_fname), dtype='float32', overwrite=True)
