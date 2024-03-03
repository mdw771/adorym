import adorym
from adorym.ptychography import reconstruct_ptychography
import numpy as np
import dxchange
import datetime
import os


def test_multislice_tomography_64():
    timestr = str(datetime.datetime.today())
    timestr = timestr[:timestr.find('.')]
    for i in [':', '-', ' ']:
        if i == ' ':
            timestr = timestr.replace(i, '_')
        else:
            timestr = timestr.replace(i, '')

    reg_l1 = adorym.L1Regularizer(alpha_d=1.e-9 * 64 ** 3, alpha_b=1.e-10 * 64 ** 3)

    params_adhesin_ff = {'fname': 'data_adhesin_360_soft_4d.h5',
                      'theta_st': 0,
                      'theta_end': 2 * np.pi,
                      'theta_downsample': 10,
                      'n_epochs': 1,
                      'regularizers': [reg_l1],
                      'obj_size': (64, 64, 64),
                      'probe_size': (64, 64),
                      # 'learning_rate': 1., # for non-shared file mode gd
                      'learning_rate': 1e-7, # for non-shared-file mode adam
                      # 'learning_rate': 1e-6, # for shared-file mode adam
                      # 'learning_rate': 1, # for shared-file mode gd
                      'center': 32,
                      'energy_ev': 800,
                      'psize_cm': 0.67e-7,
                      'binning': 1,
                      'minibatch_size': 1,
                      'n_batch_per_update': 1,
                      'output_folder': 'test',
                      'cpu_only': True,
                      'save_path': '../demos/adhesin',
                      'multiscale_level': 1,
                      'n_epoch_final_pass': None,
                      'free_prop_cm': 0,
                      'save_intermediate': True,
                      'full_intermediate': True,
                      # 'initial_guess': [np.load('adhesin_ptycho_2/phantom/grid_delta.npy'), np.load('adhesin_ptycho_2/phantom/grid_beta.npy')],
                      'initial_guess': None,
                      # 'probe_initial': [dxchange.read_tiff('adhesin_ptycho_2/probe_mag_defocus_10nm.tiff'), dxchange.read_tiff('adhesin_ptycho_2/probe_phase_defocus_10nm.tiff')],
                      'probe_initial': None,
                      'n_dp_batch': 1,
                      'fresnel_approx': True,
                      'probe_type': 'plane',
                      'finite_support_mask_path': '../demos/adhesin/fin_sup_mask/mask.tiff',
                      # 'finite_support_mask_path': None,
                      'forward_algorithm': 'fresnel',
                      'object_type': 'normal',
                      'probe_pos': [(0, 0)],
                      'optimize_probe_defocusing': False,
                      'probe_defocusing_learning_rate': 1e-7,
                      'distribution_mode': None,
                      'optimizer': 'adam',
                      'use_checkpoint': False,
                      'backend': 'pytorch',
                      'reweighted_l1': True,
                      }

    params = params_adhesin_ff

    reconstruct_ptychography(**params)


if  __name__ == '__main__':
    test_multislice_tomography_64()
