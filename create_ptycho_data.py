import numpy as np

from simulation import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_theta = 500
energy_ev = 5000
# energy_ev = 800
psize_cm = 1.e-7
# psize_cm = 0.67e-7
# free_prop_cm = None
# phantom_path = 'cell/ptychography/phantom'
phantom_path = 'cone_256_foam_ptycho/phantom'
# save_folder = 'cell/ptychography'
save_folder = 'cone_256_foam_ptycho'
fname = 'data_cone_256_foam_1nm.h5'
# fname = 'data_cell_phase.h5'
probe_size = [72, 72]
# probe_size = [18, 18]
probe_mag_sigma = 6
# probe_mag_sigma = 2
probe_phase_sigma = 6
# probe_phase_sigma = 2
probe_phase_max = 0.5
# ============================================

# probe_mag = np.ones([img_dim, img_dim], dtype=np.float32)
# probe_phase = np.zeros([img_dim, img_dim], dtype=np.float32)
# probe_phase[int(img_dim / 2), int(img_dim / 2)] = 0.1
# probe_phase = gaussian_filter(probe_phase, 3)
# wavefront_initial = [probe_mag, probe_phase]

probe_pos = [(y, x) for y in np.arange(23) * 12 for x in np.arange(23) * 12 ]
# probe_pos = [(y, x) for y in np.linspace(340, 665, 27) for x in np.linspace(340, 665, 27)]
# probe_pos = [(y, x) for y in np.arange(33) * 10 for x in np.arange(34) * 10]
# probe_pos = [(y, x) for y in np.linspace(9, 55, 23) for x in np.linspace(9, 55, 23)]
# probe_pos = [(y, x) for y in np.linspace(18, 120, 35) for x in np.linspace(54, 198, 49)] + \
#             [(y, x) for y in np.linspace(120, 222, 35) for x in np.linspace(22, 230, 70)]

create_ptychography_data(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                         probe_type='gaussian', probe_size=probe_size, theta_st=theta_st, theta_end=theta_end,
                         probe_mag_sigma=probe_mag_sigma, probe_phase_sigma=probe_phase_sigma, probe_phase_max=probe_phase_max, probe_circ_mask=None)