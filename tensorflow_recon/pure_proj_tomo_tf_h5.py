from recon import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927

# ============================================
theta_st = 0
theta_end = PI
n_epochs = 200
sino_range = (600, 601, 1)
# alpha_ls = np.concatenate([np.arange(1e-6, 1e-5, 1e-6), np.arange(1e-5, 1e-4, 1e-5), np.arange(1e-4, 1e-3, 1e-4)])
alpha_ls = [1e-5]
learning_rate_ls = [0.5]
# learning_rate_ls = [1]
center = 958
downsample = (0, 0, 0)
# output_folder = 'recon_h5_{}_alpha{}'.format(n_epochs, alpha)
# ============================================


if __name__ == '__main__':

    for alpha in alpha_ls:
        for learning_rate in learning_rate_ls:
            print('Rate: {}; alpha: {}'.format(learning_rate, alpha))
            reconstruct_pureproj(fname='data.h5',
                               sino_range=sino_range,
                               n_epochs=n_epochs,
                               alpha=alpha,
                               learning_rate=learning_rate,
                               downsample=downsample,
                               save_intermediate=True,
                               center=center)
