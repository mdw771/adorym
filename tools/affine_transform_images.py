import numpy as np
import dxchange
import torch as tc
import glob, os, re

import adorym
from adorym import affine_transform

fname_mat = '../rec_distopt1_affineopt1_kappaopt1_init100_ctf_adam_registered/intermediate/prj_affine/prj_affine_99.txt'
image_path = './data_rescaled_registered'
out_path = './data_rescaled_registered_afteropt'

mat_ls = np.loadtxt(fname_mat)
mat_ls = np.split(mat_ls, len(mat_ls) // 2, 0)

flist, n_theta, n_dists, raw_img_shape = adorym.parse_source_folder(image_path, '*')

for i_dist in range(n_dists):
    stack = []
    for i_theta in range(n_theta):
        stack.append(dxchange.read_tiff(flist[i_dist + i_theta * n_dists]))
    stack = np.stack(stack)
    mat = mat_ls[i_dist]
    mat = tc.tensor(mat, requires_grad=False)
    stack = tc.tensor(stack, requires_grad=False)
    stack = affine_transform(stack, mat, override_backend='pytorch')
    stack = stack.data.numpy()
    for i_theta, img in enumerate(stack):
        dxchange.write_tiff(img, os.path.join(out_path, os.path.basename(flist[i_dist + i_theta * n_dists])), dtype='float32', overwrite=True)