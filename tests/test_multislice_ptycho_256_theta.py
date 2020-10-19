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


params_cone_marc_theta = {'fname': 'data_cone_256_foam_1nm.h5',
                        'theta_st': 0,
                        'theta_end': 2 * np.pi,
                        'theta_downsample': 10,
                        'n_epochs': 4,
                        'obj_size': (256, 256, 256),
                        'alpha_d': 0, #1e-9 * 1.7e7,
                        'alpha_b': 0, #1e-10 * 1.7e7,
                        'gamma': 0, #1e-9 * 1.7e7,
                        'probe_size': (72, 72),
                        'learning_rate': 1e-5, #5e-5,
                        'center': 128,
                        'energy_ev': 5000,
                        'psize_cm': 1.e-7,
                        'minibatch_size': 23,
                        'n_batch_per_update': 1,
                        # 'output_folder': 'theta_' + timestr,
                        'output_folder': os.path.join(args.output_folder, 'epoch_{}'.format(epoch)),
                        'cpu_only': True,
                        'use_checkpoint': False,
                        'save_path': '../demos/cone_256_foam_ptycho',
                        'multiscale_level': 1,
                        'n_epoch_final_pass': None,
                        'save_intermediate': True,
                        'full_intermediate': True,
                        'initial_guess': None,
                        'n_dp_batch': 23,
                        'probe_type': 'gaussian',
                        'forward_algorithm': 'fresnel',
                        'probe_pos': [(y, x) for y in (np.arange(23) * 12) - 36 for x in (np.arange(23) * 12) - 36],
                        'finite_support_mask': None,
                        'probe_mag_sigma': 6,
                        'probe_phase_sigma': 6,
                        'probe_phase_max': 0.5,
                        'reweighted_l1': False,
                        'optimizer': 'gd',
                        'free_prop_cm': 'inf',
                        'backend': 'pytorch',
                        'binning': 16,}

params = params_cone_marc_theta

reconstruct_ptychography(**params)
