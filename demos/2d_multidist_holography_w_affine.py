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

params_cameraman = {'fname': 'data_nonoise.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'theta_downsample': None,
                    'n_epochs': 1000,
                    'obj_size': (512, 512, 1),
                    'safe_zone_width': 0,
                    'alpha_d': 0,
                    'alpha_b': 0,
                    'gamma': 0,
                    'two_d_mode': True,
                    'learning_rate': 1e-2,
                    'probe_learning_rate': 1e-3,
                    'minibatch_size': 1,
                    'randomize_probe_pos': True,
                    'n_batch_per_update': 1,
                    'output_folder': 'rec_nonoise_distopt1_transopt1_ref',
                    'cpu_only': False,
                    'save_path': '.',
                    'multiscale_level': 1,
                    'use_checkpoint': False,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    'initial_guess': None,
                    #'initial_guess': [dxchange.read_tiff('raw/cameraman_530.tiff')[9:9+512, 9:9+512, np.newaxis], dxchange.read_tiff('raw/baboon_530.tiff')[9:9+512, 9:9+512, np.newaxis]],
                    'random_guess_means_sigmas': (1., 0., 0., 0.01),
                    'n_dp_batch': 1,
                    'optimize_probe': False,
                    'probe_type': 'plane',
                    'probe_initial': None,
                    'rescale_probe_intensity': False,
                    'forward_algorithm': 'fresnel',
                    'probe_pos': None,
                    'finite_support_mask': None,
                    'shared_file_object': False,
                    'reweighted_l1': False,
                    'optimizer': 'adam',
                    'backend': 'pytorch',
                    'raw_data_type': 'intensity',
                    'debug': False,
                    'optimize_all_probe_pos': False,
                    'all_probe_pos_learning_rate': 1e-1,
                    'optimize_free_prop': True, 
                    'free_prop_learning_rate': 1e-1,
                    'optimize_prj_affine': True,
                    'prj_affine_learning_rate': 1e-3,
                    'save_history': True,
                    'update_scheme': 'immediate',
                    'unknown_type': 'real_imag',
                    'save_stdout': True,
                    'loss_function_type': 'lsq',
                    }


params = params_cameraman

#params['initial_guess'] = [src_mag, src_phase]
#params['initial_guess'] = [np.ones([512, 512, 1]), np.zeros([512, 512, 1])]

reconstruct_ptychography(**params)
