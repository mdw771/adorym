from adorym.ptychography import reconstruct_ptychography
import numpy as np
import dxchange
import datetime
import argparse
import os

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--save_path', default='cone_256_foam_ptycho')
parser.add_argument('--output_folder', default='test') # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]


params_adhesin_ff = {'fname': 'data_adhesin_360_soft_4d.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'theta_downsample': None,
                  'n_epochs': 10,
                  'obj_size': (64, 64, 64),
                  'alpha_d': 1.e-9 * 64 ** 3,
                  # 'alpha_d': 0,
                  'alpha_b': 1.e-10 * 64 ** 3,
                  # 'alpha_b': 0,
                  'gamma': 0,
                  'probe_size': (64, 64),
                  # 'learning_rate': 1., # for non-shared file mode gd
                  'learning_rate': 1e-7, # for non-shared-file mode adam
                  # 'learning_rate': 1e-6, # for shared-file mode adam
                  # 'learning_rate': 1, # for shared-file mode gd
                  'center': 32,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'minibatch_size': 1,
                  'n_batch_per_update': 1,
                  'output_folder': 'test_backend',
                  'cpu_only': False,
                  'save_path': 'adhesin',
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
                  'finite_support_mask_path': 'adhesin/fin_sup_mask/mask.tiff',
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
