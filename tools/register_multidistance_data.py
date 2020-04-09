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

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dir', default='.', help='Directory containing raw TIFF files.')
parser.add_argument('prefix', default='data', help='Prefix to TIFF filenames.')
parser.add_argument('--ref', default=0, help='Index of the image to be selected as reference for others.')

args = parser.parse_args()

src_dir = args.dir
prefix = args.prefix
i_ref = args.ref

flist = glob.glob(os.path.join(src_dir, prefix + '*.tif*'))
raw_img = np.squeeze(dxchange.read_tiff(flist[0]))
raw_img_shape = raw_img.shape
theta_ls = [int(re.findall(r'\d+', f)[-2]) for f in flist]
n_theta = np.max(theta_ls) + 1
flist = np.array(flist)
flist = flist[np.argsort(theta_ls)]
n_dists = len(flist) // n_theta
print(flist)

new_folder = os.path.join(os.path.dirname(src_dir), os.path.basename(src_dir) + '_registered')
try:
    os.makedirs(new_folder)
except:
    print('Target folder {} exists.'.format(new_folder))

for i_theta in range(n_theta):
    print('Processing theta {}/{}...'.format(i_theta, n_theta))
    data = np.zeros([n_dists] + list(raw_img_shape))
    img_ref = np.squeeze(dxchange.read_tiff(flist[i_theta * n_dists + i_ref]))
    data[i_ref] = img_ref
    for i_dist in range(n_dists):
        if i_dist != i_ref:
            fname = flist[i_theta * n_dists + i_dist]
            img = np.squeeze(dxchange.read_tiff(fname))
            shift, _, _ = register_translation(img_ref, img)
            print(shift)
            img = np.fft.ifft2(fourier_shift(np.fft.fft2(img), shift)).real
            data[i_dist] = img
    for i_dist, img in enumerate(data):
        fname = flist[i_theta * n_dists + i_dist]
        dxchange.write_tiff(img, os.path.join(new_folder, os.path.join(os.path.basename(fname))), dtype='float32',
                            overwrite=True)
