from recon import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_epochs = 'auto'
n_epoch_final_pass = 4
alpha_d_ls = [1.5e-6]
alpha_b_ls = [alpha_d_ls[0] * 0.1]
gamma_ls = [5e-7]
learning_rate_ls = [1e-7]
center = 128
energy_ev = 5000
psize_cm = 1e-7
batch_size = 10
n_epochs_mask_release = 10
free_prop_cm = 1e-4
shrink_cycle = 1
multiscale_level = 3
# initial_guess = ['cone_256_filled/recon_360_minibatch_10_mskrls_10_shrink_1_iter_auto_alphad_1.5e-06_alphab_1.5000000000000002e-07_gamma_1e-07_rate_1e-07_energy_5000_size_256_ntheta_500_prop_0.0001_ms_3_cpu_True/delta_ds_1.tiff',
#                  'cone_256_filled/recon_360_minibatch_10_mskrls_10_shrink_1_iter_auto_alphad_1.5e-06_alphab_1.5000000000000002e-07_gamma_1e-07_rate_1e-07_energy_5000_size_256_ntheta_500_prop_0.0001_ms_3_cpu_True/beta_ds_1.tiff']
initial_guess = None
# ============================================


if __name__ == '__main__':

    for alpha_d, alpha_b in zip(alpha_d_ls, alpha_b_ls):
        for gamma in gamma_ls:
            for learning_rate in learning_rate_ls:
                print('Rate: {}; gamma: {}'.format(learning_rate, gamma))
                reconstruct_diff(fname='data_cone_256_1nm_1um.h5',
                                 save_path='cone_256_filled',
                                 output_folder=None,
                                 phantom_path='cone_256_filled/phantom',
                                 n_epochs=n_epochs,
                                 theta_st=theta_st,
                                 theta_end=theta_end,
                                 gamma=gamma,
                                 alpha_d=alpha_d,
                                 alpha_b=alpha_b,
                                 learning_rate=learning_rate,
                                 save_intermediate=True,
                                 full_intermediate=True,
                                 n_epochs_mask_release=n_epochs_mask_release,
                                 minibatch_size=batch_size,
                                 energy_ev=energy_ev,
                                 psize_cm=psize_cm,
                                 cpu_only=True,
                                 free_prop_cm=free_prop_cm,
                                 shrink_cycle=shrink_cycle,
                                 multiscale_level=multiscale_level,
                                 n_epoch_final_pass=n_epoch_final_pass,
                                 initial_guess=initial_guess)
