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
parser.add_argument('--save_path', default='adhesin')
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


params_adhesin_2 = {'fname': 'data_adhesin_64_1nm_1um.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'theta_downsample': None,
                  'n_epochs': 10,
                  'obj_size': (64, 64, 64),
                  'alpha_d': 0,
                  'alpha_b': 0,
                  'gamma': 0,
                  'probe_size': (72, 72),
                  # 'learning_rate': 1e-5, # for non-shared file mode
                  'learning_rate': 1e-7, # for shared-file mode adam
                  # 'learning_rate': 1e-3, # for shared-file mode gd
                  'center': 32,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'minibatch_size': 23,
                  'n_batch_per_update': 1,
                  'output_folder': os.path.join(args.output_folder, 'epoch_{}'.format(epoch)),
                  'cpu_only': True,
                  'save_path': '../demos/adhesin_ptycho_2',
                  'multiscale_level': 1,
                  'n_epoch_final_pass': None,
                  'save_intermediate': True,
                  'initial_guess': init,
                  'probe_initial': None,
                  'n_dp_batch': 529,
                  'fresnel_approx': True,
                  'probe_type': 'gaussian',
                  'probe_learning_rate': 1e-3,
                  'probe_learning_rate_init': 1e-3,
                  'finite_support_mask': None,
                  'forward_algorithm': 'fresnel',
                  'object_type': 'normal',
                  'probe_pos': [(y, x) for y in np.linspace(-27, 19, 23, dtype=int) for x in np.linspace(-27, 19, 23, dtype=int)],
                  'probe_mag_sigma': 6,
                  'probe_phase_sigma': 6,
                  'probe_phase_max': 0.5,
                  'optimize_probe_defocusing': False,
                  'probe_defocusing_learning_rate': 1e-7,
                  'shared_file_object': True,
                  'shared_file_mode_n_batch_per_update': 1,
                  'optimizer': 'adam',
                  'store_checkpoint': True,
                  'use_checkpoint': False,
                  'free_prop_cm': 'inf',
                  'debug': True,
                  'raw_data_type': 'magnitude',
                  'backend': 'autograd',
                  }

params = params_adhesin_2

reconstruct_ptychography(**params)