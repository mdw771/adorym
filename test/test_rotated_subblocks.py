import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from util import get_rotated_subblocks, read_all_origin_coords
import h5py
import dxchange
import numpy as np

f = h5py.File('intermediate_obj.h5', 'r')
dset = f['obj']

print('Reading coords...')
coord_ls = read_all_origin_coords('../arrsize_256_256_256_ntheta_500', 500)
print('Getting data...')
block = get_rotated_subblocks(dset, [(128, 128)], coord_ls[62], [18, 18])
print(block.shape)

dxchange.write_tiff(block[0, :, :, :, 0], 'block', dtype='float32')