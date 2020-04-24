"""
Rescale multidistance holography images using Fresnel scaling theorem. Use this
script before executing convert_multidistance_to_adorym.py.
"""
import numpy as np
import dxchange
import datetime
import argparse
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
import os
import glob
import re
import sys

import adorym

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dir', default='.', help='Directory containing raw TIFF files.')
parser.add_argument('prefix', default='data', help='Prefix to TIFF filenames.')
parser.add_argument('--ref', default=0, help='Index of the image to be selected as reference for others.')

args = parser.parse_args()

src_dir = args.dir
prefix = args.prefix
i_ref = args.ref

flist, n_theta, n_dists, raw_img_shape = adorym.parse_source_folder(src_dir, prefix)
print(flist)

new_folder = os.path.join(os.path.dirname(src_dir), os.path.basename(src_dir) + '_registered')
try:
    os.makedirs(new_folder)
except:
    print('Target folder {} exists.'.format(new_folder))

shift_ls = [None] * n_dists
for i_theta in range(n_theta):
    print('Processing theta {}/{}...'.format(i_theta, n_theta))
    data = np.zeros([n_dists] + list(raw_img_shape))
    img_ref = np.squeeze(dxchange.read_tiff(flist[i_theta * n_dists + i_ref]))
    data[i_ref] = img_ref
    for i_dist in range(n_dists):
        if i_dist != i_ref:
            fname = flist[i_theta * n_dists + i_dist]
            img = np.squeeze(dxchange.read_tiff(fname))
            if i_theta == 0:
                shift, _, _ = register_translation(img_ref, img, upsample_factor=10)
                shift_ls[i_dist] = shift
                print(shift_ls)
            else:
                shift = shift_ls[i_dist]
            img = np.fft.ifft2(fourier_shift(np.fft.fft2(img), shift)).real
            data[i_dist] = img
    for i_dist, img in enumerate(data):
        fname = flist[i_theta * n_dists + i_dist]
        dxchange.write_tiff(img, os.path.join(new_folder, os.path.join(os.path.basename(fname))), dtype='float32',
                            overwrite=True)
