import numpy as np

from simulation import *


# ============================================
# DO NOT ROTATE PROGRESSIVELY
# (DO NOT CONTINUE TO ROTATE AN INTERPOLATED OBJECT)
# ============================================

PI = 3.1415927


params_adhesin = {'fname': 'data_adhesin_360_soft.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_theta': 500,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'batch_size': 1,
                  'free_prop_cm': None,
                  'n_batch_per_update': 1,
                  'save_folder': 'adhesin',
                  'phantom_path': 'adhesin/phantom',
                  'probe_type': 'plane',
                  'dist_to_source_cm': None,
                  'det_psize_cm': None,
                  'theta_max': None,
                  'phi_max': None,
                  'probe_options': {}
                  }

params_cone_point = {'fname': 'data_cone_256_1nm_1um.h5',
                     'theta_st': 0,
                     'theta_end': 2 * PI,
                     'n_theta': 500,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': 'cone_256_filled_pp',
                     'phantom_path': 'cone_256_filled_pp/phantom',
                     'probe_type': 'point',
                     'dist_to_source_cm': 1e-4,
                     'det_psize_cm': 3e-7,
                     'theta_max': PI / 15,
                     'phi_max': PI / 15,
                     'probe_options': {}
                     }

params_cone_512 = {'fname': 'data_cone_512_1nm_1um.h5',
                     'theta_st': 0,
                     'theta_end': 2 * PI,
                     'n_theta': 500,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': 'cone_512_filled',
                     'phantom_path': 'cone_512_filled/phantom',
                     'probe_type': 'plane',
                     'dist_to_source_cm': None,
                     'det_psize_cm': None,
                     'theta_max': None,
                     'phi_max': None,
                     'probe_options': None
                     }

params_cone = {'fname': 'data_cone_256_1nm_1um.h5',
                     'theta_st': 0,
                     'theta_end': 2 * PI,
                     'n_theta': 500,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': 'cone_256_foam',
                     'phantom_path': 'cone_256_foam/phantom',
                     'probe_type': 'plane',
                     'dist_to_source_cm': None,
                     'det_psize_cm': None,
                     'theta_max': None,
                     'phi_max': None,
                     'probe_options': {}
                     }

params_cone_180 = {'fname': 'data_cone_256_1nm_1um_180_tilt.h5',
                     'theta_st': PI/6,
                     'theta_end': PI*7/6,
                     'n_theta': 500,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': 'cone_256_filled/new',
                     'phantom_path': 'cone_256_filled/phantom',
                     'probe_type': 'plane',
                     'dist_to_source_cm': None,
                     'det_psize_cm': None,
                     'theta_max': None,
                     'phi_max': None,
                     'probe_options': {}
                     }


params_2d = {'fname': 'data_cone_2d_1nm_1um_phase.h5',
                     'theta_st': 0,
                     'theta_end': 0,
                     'n_theta': 1,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1e-4,
                     'save_folder': '2d_512',
                     'phantom_path': '2d_512/phantom_phase',
                     'probe_type': 'plane',
                     'dist_to_source_cm': None,
                     'det_psize_cm': None,
                     'theta_max': None,
                     'phi_max': None,
                     'probe_options': {'probe_mag_sigma': 100,
                                       'probe_phase_sigma': 100,
                                       'probe_phase_max': 0.5},
                     }

params_2d_cell = {'fname': 'data_cell_phase.h5',
                     'theta_st': 0,
                     'theta_end': 0,
                     'n_theta': 1,
                     'energy_ev': 5000,
                     'psize_cm': 1.e-7,
                     'batch_size': 1,
                     'free_prop_cm': 1. / (0.001 * (1.24 / 5)) * 1e-7,
                     'save_folder': 'cell/fullfield',
                     'phantom_path': 'cell/fullfield/phantom',
                     'probe_type': 'plane',
                     'dist_to_source_cm': None,
                     'det_psize_cm': None,
                     'theta_max': None,
                     'phi_max': None,
                     'probe_options': {'probe_mag_sigma': 100,
                                       'probe_phase_sigma': 100,
                                       'probe_phase_max': 0.5},
                     }

params = params_cone

create_fullfield_data_numpy(energy_ev=params['energy_ev'],
                            psize_cm=params['psize_cm'],
                            free_prop_cm=params['free_prop_cm'],
                            n_theta=params['n_theta'],
                            phantom_path=params['phantom_path'],
                            save_folder=params['save_folder'],
                            fname=params['fname'],
                            batch_size=params['batch_size'],
                            probe_type=params['probe_type'],
                            wavefront_initial=None,
                            theta_st=params['theta_st'],
                            theta_end=params['theta_end'],
                            dist_to_source_cm=params['dist_to_source_cm'],
                            det_psize_cm=params['det_psize_cm'],
                            theta_max=params['theta_max'],
                            phi_max=params['phi_max'],
                            monitor_output=True,
                            **params['probe_options']
                            )
