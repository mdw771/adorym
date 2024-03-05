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


params_cameraman = {'fname': 'data_cameraman_err_10.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'theta_downsample': 1,
                    'n_epochs': 1000,
                    'obj_size': (256, 256, 1),
                    'alpha_d': 0,
                    'alpha_b': 0,
                    'gamma': 0,
                    'probe_size': (72, 72),
                    'learning_rate': 4e-3,
                    'center': 512,
                    'energy_ev': 5000,
                    'psize_cm': 1.e-7,
                    #'minibatch_size': 2704,
                    'minibatch_size': 52,
                    'n_batch_per_update': 1,
                    'output_folder': 'recon',
                    'cpu_only': False,
                    'save_path': '../demos/cameraman_pos_error',
                    'multiscale_level': 1,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    'initial_guess': None,
                    'n_dp_batch': 20,
                    'probe_type': 'ifft',
                    'probe_initial': None,
                    'forward_algorithm': 'fresnel',
                    'object_type': 'phase_only',
                    'probe_pos': np.array([(y, x) for y in np.arange(-10, 246, 5) for x in np.arange(-10, 246, 5)]),
                    'finite_support_mask': None,
                    'free_prop_cm': 'inf',
                    'optimizer': 'adam',
                    'two_d_mode': True,
                    'shared_file_object': False,
                    'use_checkpoint': False,
                    'optimize_all_probe_pos': True,
                    'optimize_probe': True,
                    'probe_learning_rate': 0.001,
                    'save_history': True,
                    'backend': 'pytorch'
                    }

params = params_cameraman

reconstruct_ptychography(**params)
