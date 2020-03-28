"""
Convert a collection of TIFF images of multi-distance holo(tomo)graphy into Adorym readable HDF5.
Raw TIFF files must be named as prefix_iTheta_iDistance.tiff.
"""
import numpy as np
import dxchange
import datetime
import argparse
import os
import glob
import re
import h5py
import sys
import adorym

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('dir', default='.', help='Directory containing raw TIFF files.')
parser.add_argument('distances_cm', default='None', help='Distances in cm, separated by comma (no spaces allowed)')
parser.add_argument('prefix', default='data', help='Prefix to TIFF filenames.')
parser.add_argument('--output', default='data.h5', help='Output filename.')
parser.add_argument('--n_blocks_y', default=1, type=int, help='Number of subblocks in y.')
parser.add_argument('--n_blocks_x', default=1, type=int, help='Number of subblocks in x.')
parser.add_argument('--energy_ev', default=5000., type=float, help='Beam energy in ev.')
parser.add_argument('--psize_cm', default=1e-4, type=float, help='Sample plane pixel size in cm.')

args = parser.parse_args()

src_dir = args.dir
dist_cm_ls = args.distances_cm
dist_cm_ls = [float(d) for d in dist_cm_ls.split(',')]
prefix = args.prefix
out_fname = args.output
n_blocks_y, n_blocks_x = int(args.n_blocks_y), int(args.n_blocks_x)
n_blocks = n_blocks_y * n_blocks_x

flist = glob.glob(os.path.join(src_dir, prefix + '*.tif*'))
raw_img = np.squeeze(dxchange.read_tiff(flist[0]))
raw_img_shape = raw_img.shape
n_dists = len(dist_cm_ls)
flist.sort()
n_theta = int(re.findall(r'\d+', flist[-1])[-2]) + 1

energy_ev = float(args.energy_ev)
lmbda_nm = 1240. / energy_ev 
psize_cm = float(args.psize_cm)

flist = [flist[i * n_dists:(i + 1) * n_dists] for i in range(n_theta)]
print(flist)

if n_blocks == 1:
    block_size_y, block_size_x = raw_img_shape
    block_range_ls = np.array([[0, raw_img.shape[0], 0, raw_img.shape[1]]])
else:
    block_range_ls = adorym.get_subdividing_params(raw_img_shape, n_blocks_y, n_blocks_x)
    block_size_y, block_size_x = (block_range_ls[0][1] - block_range_ls[0][0], block_range_ls[0][3] - block_range_ls[0][2])

if os.path.exists(out_fname):
    print('File exists. Overwrite? (Y/n)')
    cont = input()
    if cont not in ['Y', 'y']:
        sys.exit()
f = h5py.File(out_fname, 'w')
grp = f.create_group('exchange')
dset = grp.create_dataset('data', shape=[n_theta, n_blocks * n_dists, block_size_y, block_size_x], dtype='float32')

for i_theta in range(n_theta):
    for i_dist in range(n_dists):
        img = np.squeeze(dxchange.read_tiff(flist[i_theta][i_dist]))
        if n_blocks == 1:
            dset[i_theta, i_dist] = img
        else:
            block_ls = adorym.subdivide_image(img, block_range_ls, override_backend='numpy')
            dset[i_theta, i_dist * n_blocks:(i_dist + 1) * n_blocks, :, :] = np.array(block_ls)

grp = f.create_group('metadata')
grp.create_dataset('probe_pos_px', data=block_range_ls[:, 0:3:2])
grp.create_dataset('energy_ev', data=energy_ev)
grp.create_dataset('psize_cm', data=psize_cm)
grp.create_dataset('free_prop_cm', data=dist_cm_ls)

f_meta = open('parameters.txt', 'w')
f_meta.write('wavelength_nm:     {}\n'.format(lmbda_nm))
f_meta.write('energy_ev:         {}\n'.format(energy_ev))
f_meta.write('distances_cm:      {}\n'.format(dist_cm_ls))
f_meta.close()
#dxchange.write_tiff_stack(dset[0], 'diffraction_dat.tiff', dtype='float32', overwrite=True)

f.close()
