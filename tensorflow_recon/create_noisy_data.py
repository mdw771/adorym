import h5py
import numpy as np
import matplotlib.pyplot as plt
import dxchange
from tqdm import trange


src_fname = 'cone_256_filled/new/data_cone_256_1nm_1um.h5'
dest_fname = 'cone_256_filled/new/data_cone_256_1nm_1um_n2e5.h5'
n_ph = 2.e5

o = h5py.File(src_fname, 'r')['exchange/data']
file_new = h5py.File(dest_fname, 'w')
grp = file_new.create_group('exchange')
n = grp.create_dataset('data', dtype='complex64', shape=o.shape)
snr_ls = []

for i in trange(o.shape[0]):
    prj_o = o[i]
    prj_o_inten = np.abs(prj_o) ** 2
    prj_o_inten_noisy = np.random.poisson(prj_o_inten * n_ph)
    prj_o_inten_noisy = prj_o_inten_noisy / n_ph
    noise = prj_o_inten_noisy - prj_o_inten
    snr = np.var(prj_o_inten) / np.var(noise)
    snr_ls.append(snr)
    data = np.sqrt(prj_o_inten_noisy)
    n[i] = data.astype('complex64')

print('Average SNR is {}.'.format(np.mean(snr_ls)))


# ------- based on SNR -------
# snr = 10

# for i in tqdm(range(o.shape[0])):
# for i in tqdm(range(1)):
#     prj_o = o[i]
#     prj_o_inten = np.abs(prj_o) ** 2
#     var_signal = np.var(prj_o_inten)
#     var_noise = var_signal / snr

    # noise = np.random.poisson(prj_o_inten * (var_noise / np.mean(prj_o_inten)) * 1e10) / 1e5
    # noise = noise * np.sqrt(var_noise / np.var(noise))
    # noise -= np.mean(noise)
    # prj_n_inten = prj_o_inten + noise
    # prj_n = prj_o * np.sqrt(prj_n_inten / prj_o_inten)
    #
    # n[i] = prj_n
