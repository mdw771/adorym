import numpy as np

from simulation import create_ptychography_data


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
psize_cm = 1.e-7
free_prop_cm = None
phantom_path = 'cone_256_filled_ptycho/phantom'
save_folder = 'cone_256_filled_ptycho'
fname = 'data_cone_256_1nm_1um'
probe_size = [72, 72]
# ============================================

# probe_mag = np.ones([img_dim, img_dim], dtype=np.float32)
# probe_phase = np.zeros([img_dim, img_dim], dtype=np.float32)
# probe_phase[int(img_dim / 2), int(img_dim / 2)] = 0.1
# probe_phase = gaussian_filter(probe_phase, 3)
# wavefront_initial = [probe_mag, probe_phase]

probe_pos = [(y, x) for y in np.linspace(36, 220, 23) for x in np.linspace(36, 220, 23)]

create_ptychography_data(energy_ev, psize_cm, n_theta, phantom_path, save_folder, fname, probe_pos,
                         probe_type='gaussian', probe_size=probe_size, theta_st=theta_st, theta_end=theta_end,
                         probe_mag_sigma=14, probe_phase_sigma=14, probe_phase_max=0.5)