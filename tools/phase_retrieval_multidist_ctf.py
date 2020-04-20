import numpy as np
from numpy.fft import *
import dxchange
from scipy.special import erf
import os, glob, re
from adorym import gen_mesh


def multidistance_ctf(prj_ls, dist_cm_ls, psize_cm, energy_kev, kappa=50, sigma_cut=0.01, alpha_1=5e-4, alpha_2=1e-16):

    prj_ls = np.array(prj_ls)
    dist_cm_ls = np.array(dist_cm_ls)
    dist_nm_ls = dist_cm_ls * 1.e7
    lmbda_nm = 1.24 / energy_kev
    psize_nm = psize_cm * 1.e7
    prj_shape = prj_ls.shape[1:]

    u_max = 1. / (2. * psize_nm)
    v_max = 1. / (2. * psize_nm)
    u, v = gen_mesh([v_max, u_max], prj_shape)
    xi_mesh = np.pi * lmbda_nm * (u ** 2 + v ** 2)
    xi_ls = np.zeros([len(dist_cm_ls), *prj_shape])
    for i in range(len(dist_cm_ls)):
        xi_ls[i] = xi_mesh * dist_nm_ls[i]

    abs_nu = np.sqrt(u ** 2 + v ** 2)
    nu_cut = 0.6 * u_max
    f = 0.5 * (1 - erf((abs_nu - nu_cut) / sigma_cut))
    alpha = alpha_1 * f + alpha_2 * (1 - f)
    phase = np.sum(fftshift(fft2(prj_ls - 1, axes=(-2, -1)), axes=(-2, -1)) * (np.sin(xi_ls) + 1. / kappa * np.cos(xi_ls)), axis=0)
    phase /= (np.sum(2 * (np.sin(xi_ls) + 1. / kappa * np.cos(xi_ls)) ** 2, axis=0) + alpha)
    phase = ifft2(ifftshift(phase, axes=(-2, -1)), axes=(-2, -1))
    phase = phase.real

    return phase


src_dir = 'data_rescaled'
prefix = '*'
psize_cm = 99.8e-7
dist_cm_ls = np.array([7.36, 7.42, 7.70])
energy_ev = 17500
alpha_1=5e-4
alpha_2=1e-16
energy_kev = energy_ev * 1e-3

flist = np.array(glob.glob(os.path.join(src_dir, prefix + '*.tif*')))
raw_img = np.squeeze(dxchange.read_tiff(flist[0]))
raw_img_shape = raw_img.shape
n_dists = len(dist_cm_ls)
theta_ls_full = [int(re.findall(r'\d+', f)[-2]) for f in flist]
theta_ls = np.unique(theta_ls_full)
n_theta = np.max(theta_ls) + 1
flist = flist[np.argsort(theta_ls_full)]
print(flist)

for i_theta in range(n_theta):
    prj_ls = []
    for i_dist in range(n_dists):
        img = np.squeeze(dxchange.read_tiff(flist[i_theta * n_dists + i_dist]))
        prj_ls.append(img)
    phase = multidistance_ctf(prj_ls, dist_cm_ls, psize_cm, energy_kev, kappa=50, sigma_cut=0.01, alpha_1=alpha_1, alpha_2=alpha_2)
    dxchange.write_tiff(np.squeeze(phase), os.path.join('data_ctf', os.path.basename(flist[i_theta * n_dists])), dtype='float32', overwrite=True)

