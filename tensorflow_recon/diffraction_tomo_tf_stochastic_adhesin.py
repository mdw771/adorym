from recon import *
import dxchange

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

    probe_mag = dxchange.read_tiff('adhesin_phase_ramp/probe_mag.tiff')
    probe_phase = dxchange.read_tiff('adhesin_phase_ramp/probe_phase.tiff')

    initial_delta = np.load('adhesin_phase_ramp/phantom/grid_delta.npy')
    initial_beta = np.load('adhesin_phase_ramp/phantom/grid_beta.npy')

    for alpha_d, alpha_b in zip(alpha_d_ls, alpha_b_ls):
        for gamma in gamma_ls:
            for learning_rate in learning_rate_ls:
                print('Rate: {}; gamma: {}'.format(learning_rate, gamma))
                reconstruct_diff(fname='data_adhesin_ramp_256_1nm_1um.h5',
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
                                 cpu_only=True,
                                 save_path='adhesin_phase_ramp',
                                 phantom_path='adhesin_phase_ramp/phantom',
                                 output_folder='probe_test',
                                 initial_guess=None,
                                 shrink_cycle=10,
                                 multiscale_level=1,
                                 n_batch_per_update=n_batch_per_update,
                                 wavefront_type='optimizable',
                                 wavefront_initial=None,
                                 dynamic_rate=True)
