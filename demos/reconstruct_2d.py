from fullfield import reconstruct_fullfield
import numpy as np
import dxchange
from constants import *

# init_delta_adhesin = np.zeros([64, 64, 64])
# init_delta_adhesin[...] = 8e-7
# init_beta_adhesin = np.zeros([64, 64, 64])
# init_beta_adhesin[...] = 8e-8
# init_delta_adhesin = np.load('adhesin/phantom/grid_delta.npy')
# init_beta_adhesin = np.load('adhesin/phantom/grid_beta.npy')
# init_delta = dxchange.read_tiff('cone_256_filled/new/2d_dose/n2e5/delta_ds_1.tiff')
# init_beta = dxchange.read_tiff('cone_256_filled/new/2d_dose/n2e5/beta_ds_1.tiff')
# init = [init_delta, init_beta]

params_cone_noisy = {'fname': 'data_cone_256_1nm_1um_n2e3.h5',
                     'theta_st': 0,
                     'theta_end': 2 * np.pi,
                     'n_epochs': 20,
                     'alpha_d': 1.5e-7,
                     'alpha_b': 1.5e-8,
                     'gamma': 5e-8,
                     'learning_rate': 1e-7,
                     'center': 128,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 10,
                     'theta_downsample': 500,
                     'n_epochs_mask_release': 20,
                     'shrink_cycle': 1,
                     'free_prop_cm': 1e-4,
                     'n_batch_per_update': 1,
                     'output_folder': '2d_dose/n2e3',
                     'cpu_only': True,
                     'save_folder': 'cone_256_filled/new',
                     'phantom_path': 'cone_256_filled/phantom',
                     'multiscale_level': 1,
                     'n_epoch_final_pass': 20,
                     'save_intermediate': True,
                     'full_intermediate': True,
                     'initial_guess': None,
                     'probe_type': 'plane',
                     'forward_algorithm': 'fresnel',
                     'kwargs': {}}

params = params_cone_noisy


reconstruct_fullfield(fname=params['fname'],
                      theta_st=0,
                      theta_end=params['theta_end'],
                      n_epochs=params['n_epochs'],
                      n_epochs_mask_release=params['n_epochs_mask_release'],
                      shrink_cycle=params['shrink_cycle'],
                      crit_conv_rate=0.03,
                      max_nepochs=200,
                      alpha_d=params['alpha_d'],
                      alpha_b=params['alpha_b'],
                      gamma=params['gamma'],
                      free_prop_cm=params['free_prop_cm'],
                      learning_rate=params['learning_rate'],
                      output_folder=params['output_folder'],
                      minibatch_size=params['batch_size'],
                      theta_downsample=params['theta_downsample'],
                      save_intermediate=params['save_intermediate'],
                      full_intermediate=params['full_intermediate'],
                      energy_ev=params['energy_ev'],
                      psize_cm=params['psize_cm'],
                      cpu_only=params['cpu_only'],
                      save_path=params['save_folder'],
                      phantom_path=params['phantom_path'],
                      multiscale_level=params['multiscale_level'],
                      n_epoch_final_pass=params['n_epoch_final_pass'],
                      initial_guess=params['initial_guess'],
                      n_batch_per_update=params['n_batch_per_update'],
                      dynamic_rate=True,
                      probe_type=params['probe_type'],
                      probe_initial=None,
                      probe_learning_rate=1e-3,
                      pupil_function=None,
                      forward_algorithm=params['forward_algorithm'],
                      random_theta=False,
                      **params['kwargs'])
