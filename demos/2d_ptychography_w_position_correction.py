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
                    'minibatch_size': 2704,
                    'n_batch_per_update': 1,
                    'output_folder': 'recon',
                    'cpu_only': True,
                    'save_path': 'cameraman_pos_error',
                    'multiscale_level': 1,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    'initial_guess': None,
                    'n_dp_batch': 20,
                    'probe_type': 'supplied',
                    'probe_initial': [np.squeeze(dxchange.read_tiff('cameraman_pos_error/probe_mag_true.tiff')), np.squeeze(dxchange.read_tiff('cameraman_pos_error/probe_phase_true.tiff'))],
                    'forward_algorithm': 'fresnel',
                    'object_type': 'phase_only',
                    'probe_pos': np.array([(y, x) for y in np.arange(-10, 246, 5) for x in np.arange(-10, 246, 5)]),
                    'finite_support_mask': None,
                    'free_prop_cm': 'inf',
                    'optimizer': 'adam',
                    'two_d_mode': True,
                    'distribution_mode': None,
                    'use_checkpoint': False,
                    'optimize_all_probe_pos': True,
                    'save_history': True,
                    'backend': 'pytorch'
                    }

params = params_cameraman

reconstruct_ptychography(**params)
