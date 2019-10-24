from simulation import create_fullfield_data


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
free_prop_cm = 1.e-4
phantom_path = 'cone_256_filled/phantom'
save_folder = 'cone_256_filled'
fname = 'test.h5'
probe_type = 'plane'
# ============================================

# probe_mag = np.ones([img_dim, img_dim], dtype=np.float32)
# probe_phase = np.zeros([img_dim, img_dim], dtype=np.float32)
# probe_phase[int(img_dim / 2), int(img_dim / 2)] = 0.1
# probe_phase = gaussian_filter(probe_phase, 3)
# wavefront_initial = [probe_mag, probe_phase]

create_fullfield_data(energy_ev, psize_cm, free_prop_cm, n_theta, phantom_path, save_folder, fname,
                      probe_type=probe_type, theta_st=theta_st, theta_end=theta_end)