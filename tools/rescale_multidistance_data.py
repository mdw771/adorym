"""
Rescale multidistance holography images using Fresnel scaling theorem. Use this
script before executing convert_multidistance_to_adorym.py.
"""
import numpy as np
import dxchange
import datetime
import argparse
from skimage.transform import rescale
import os
import glob
import re
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dir', default='.', help='Directory containing raw TIFF files.')
parser.add_argument('prefix', default='data', help='Prefix to TIFF filenames.')

parser.add_argument('--z_od_ls', default='None', help='Object to detector distance in cm. Separate by comma. '
                                                    'Must match order of input image indexing.')
parser.add_argument('--z_sd', default='None', help='Source to detector distance in cm.')
parser.add_argument('--psize_ls', default=None, help='List of pixel sizes in um. Separate by comma. '
                                                     'Must match order of input image indexing.')
parser.add_argument('--crop', default=True, type=bool, help='Whether to crop image to keep output shape the same.')
args = parser.parse_args()

z_od_ls = args.z_od_ls.split(',')
z_od_ls = np.array([float(z) for z in z_od_ls])
z_sd = float(args.z_sd)
src_dir = args.dir
prefix = args.prefix
psize_ls = args.psize_ls.split(',')
psize_ls = np.array([float(z) for z in psize_ls])
crop = args.crop

def convert_cone_to_parallel(data, z_sd, z_od_ls, psize=None, crop=True):
    """
    :param data: A set of multi-distance images at a single angle, i.e. [n_distances, y, x].
    """
    z_od_ls = np.array(z_od_ls)
    z_so_ls = z_sd - z_od_ls
    z_eff_ls = z_so_ls * z_od_ls / z_sd
    mag_ls = z_sd / z_so_ls
    new_data = []
    if psize is not None:
        psize_norm = np.array(psize) / np.min(psize)
        ind_ref = np.argmin(psize)
        shape_ref = data[ind_ref].shape
        shape_ref_half = (np.array(shape_ref) / 2).astype('int')
        for i, img in enumerate(data):
            if i != ind_ref:
                zoom = psize_norm[i]
                img = rescale(img, zoom, multichannel=False)
                if crop:
                    center = (np.array(img.shape) / 2).astype('int')
                    img = img[center[0] - shape_ref_half[0]:center[0] - shape_ref_half[0] + shape_ref[0],
                          center[1] - shape_ref_half[1]:center[1] - shape_ref_half[1] + shape_ref[1]]
            new_data.append(img)
    else:
        # unify zooming of all images to the one with largest magnification
        mag_norm = mag_ls / mag_ls.max()
        ind_ref = np.argmax(mag_norm)
        shape_ref = data[ind_ref].shape
        shape_ref_half = (np.array(shape_ref) / 2).astype('int')
        for i, img in enumerate(data):
            if i != ind_ref:
                zoom = 1. / mag_norm[i]
                img = rescale(img, zoom, multichannel=False)
                if crop:
                    center = (np.array(img.shape) / 2).astype('int')
                    img = img[center[0] - shape_ref_half[0]:center[0] - shape_ref_half[0] + shape_ref[0],
                              center[1] - shape_ref_half[1]:center[1] - shape_ref_half[1] + shape_ref[1]]
            new_data.append(img)
    return new_data, z_eff_ls, mag_ls

flist = glob.glob(os.path.join(src_dir, prefix + '*.tif*'))
raw_img = np.squeeze(dxchange.read_tiff(flist[0]))
raw_img_shape = raw_img.shape
n_dists = len(z_od_ls)
flist.sort()
print(flist)
n_theta = int(re.findall(r'\d+', flist[-1])[-2]) + 1

new_folder = os.path.join(os.path.dirname(src_dir), os.path.basename(src_dir) + '_rescaled')
try:
    os.makedirs(new_folder)
except:
    print('Target folder {} exists.'.format(new_folder))

for i_theta in range(n_theta):
    print('Processing theta {}/{}...'.format(i_theta, n_theta))
    data = np.zeros([n_dists] + list(raw_img_shape))
    for i_dist in range(n_dists):
        fname = flist[i_theta * n_dists + i_dist]
        data[i_dist] = dxchange.read_tiff(fname)
    data, z_eff_ls, mag_ls = convert_cone_to_parallel(data, z_sd, z_od_ls, psize_ls, crop)
    for i_dist, img in enumerate(data):
        fname = flist[i_theta * n_dists + i_dist]
        dxchange.write_tiff(img, os.path.join(new_folder, os.path.join(os.path.basename(fname))), dtype='float32', overwrite=True)

np.savetxt(os.path.join(new_folder, 'z_eff_ls.txt'), z_eff_ls, fmt='%.3f')
np.savetxt(os.path.join(new_folder, 'mag_ls.txt'), mag_ls, fmt='%.3f')