from fullfield import *
import dxchange
import numpy as np
import tomopy

# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_epochs = 10
alpha_d_ls = [1.e-9]
alpha_b_ls = [1.e-10]
gamma_ls = [0]
learning_rate_ls = [1e-7]
center = 32
energy_ev = 800
psize_cm = 0.67e-7
batch_size = 10
n_epochs_mask_release = 200
free_prop_cm = None
n_batch_per_update = 1
# ============================================


if __name__ == '__main__':

    # probe_mag = dxchange.read_tiff('adhesin_aperture_update/probe_mag.tiff')
    # probe_phase = dxchange.read_tiff('adhesin_aperture_update/probe_phase.tiff')

    # offset aperture position
    # probe_mag = np.roll(probe_mag, shift=[2, 2], axis=[0, 1])
    # dxchange.write_tiff(probe_mag, 'adhesin_aperture/probe_mag_shift', dtype='float32', overwrite=True)

    # pupil
    # img_dim = probe_mag.shape[0]
    # aperture = tomopy.misc.corr._get_mask(img_dim, img_dim, 0.4)

    # initial_delta = np.load('adhesin_aperture_update/phantom/grid_delta.npy')
    # initial_beta = np.load('adhesin_aperture_update/phantom/grid_beta.npy')

    for alpha_d, alpha_b in zip(alpha_d_ls, alpha_b_ls):
        for gamma in gamma_ls:
            for learning_rate in learning_rate_ls:
                print('Rate: {}; gamma: {}'.format(learning_rate, gamma))
                reconstruct_diff(fname='data_adhesin_360_soft.h5',
                                 n_epochs=n_epochs,
                                 theta_st=theta_st,
                                 theta_end=theta_end,
                                 gamma=gamma,
                                 alpha_d=alpha_d,
                                 alpha_b=alpha_b,
                                 learning_rate=learning_rate,
                                 save_intermediate=True,
                                 n_epochs_mask_release=n_epochs_mask_release,
                                 minibatch_size=batch_size,
                                 energy_ev=energy_ev,
                                 psize_cm=psize_cm,
                                 cpu_only=False,
                                 save_path='adhesin',
                                 phantom_path='adhesin/phantom',
                                 output_folder='test',
                                 initial_guess=None,
                                 # initial_guess=[initial_delta, initial_beta],
                                 shrink_cycle=10,
                                 multiscale_level=1,
                                 n_batch_per_update=n_batch_per_update,
                                 wavefront_type='plane',
                                 wavefront_initial=None,
                                 # wavefront_initial=[probe_mag, probe_phase],
                                 dynamic_rate=True,
                                 probe_learning_rate=1e-3,
                                 pupil_function=None)
