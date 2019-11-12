import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from util import get_rotated_subblocks, read_all_origin_coords
import h5py
import dxchange
import numpy as np

# f = h5py.File('intermediate_obj.h5', 'r')
# dset = f['obj']
dset = np.load('intermediate_obj.npy', mmap_mode='r+', allow_pickle=True)

print('Reading coords...')
coord_ls = read_all_origin_coords('../arrsize_256_256_256_ntheta_500', 500)
print('Getting data...')
block = get_rotated_subblocks(dset, [(230, 255)], coord_ls[62], [90, 90])
print(block.shape)

dxchange.write_tiff(block[0, :, :, :, 0], 'block', dtype='float32')