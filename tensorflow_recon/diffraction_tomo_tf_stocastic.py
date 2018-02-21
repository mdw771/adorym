from recon import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = 2 * PI
n_epochs = 200
alpha_ls = [1e-7]
gamma_ls = [0]
learning_rate_ls = [5e-6]
center = 32
energy_ev = 5000
psize_cm = 1e-7
batch_size = 50
n_epochs_mask_release = 100
# ============================================


if __name__ == '__main__':

    for alpha in alpha_ls:
        for gamma in gamma_ls:
            for learning_rate in learning_rate_ls:
                print('Rate: {}; gamma: {}'.format(learning_rate, gamma))
                reconstruct_diff(fname='data_diff_tf_360.h5',
                                 n_epochs=n_epochs,
                                 theta_st=theta_st,
                                 theta_end=theta_end,
                                 gamma=gamma,
                                 alpha=alpha,
                                 learning_rate=learning_rate,
                                 downsample=(0, 0, 0),
                                 save_intermediate=True,
                                 n_epochs_mask_release=n_epochs_mask_release,
                                 minibatch_size=batch_size,
                                 energy_ev=energy_ev,
                                 psize_cm=psize_cm)
