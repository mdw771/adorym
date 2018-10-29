import h5py
import numpy as np
import matplotlib.pyplot as plt
import dxchange
from tqdm import tqdm


src_fname = 'cone_256_filled/new/data_cone_256_1nm_1um.h5'
dest_fname = 'cone_256_filled/new/data_cone_256_1nm_1um_snr10.h5'
snr = 10

o = h5py.File(src_fname, 'r')['exchange/data']
file_new = h5py.File(dest_fname, 'w')
dset = file_new.create_group('exchange')
n = dset.create_dataset('data', dtype='complex64', shape=o.shape)

for i in tqdm(range(o.shape[0])):
    prj_o = o[i]
    prj_o_inten = np.abs(prj_o) ** 2
    var_signal = np.var(prj_o_inten)
    var_noise = var_signal / snr

    noise = np.random.poisson(prj_o_inten * (var_noise / np.mean(prj_o_inten)) * 1e10) / 1e5
    noise = noise * np.sqrt(var_noise / np.var(noise))
    noise -= np.mean(noise)
    prj_n_inten = prj_o_inten + noise
    prj_n = prj_o * np.sqrt(prj_n_inten / prj_o_inten)

    n[i] = prj_n


