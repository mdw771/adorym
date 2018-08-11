from ptychography import reconstruct_ptychography
import numpy as np
import dxchange


params_adhesin = {'fname': 'data_adhesin_64_1nm_1um.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_epochs': 1,
                  'obj_size': (64, 64, 64),
                  'alpha_d': 1e-9,
                  'alpha_b': 1e-10,
                  'gamma': 0,
                  'probe_size': (18, 18),
                  'learning_rate': 1e-7,
                  'center': 32,
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'batch_size': 1,
                  'n_batch_per_update': 1,
                  'output_folder': 'test',
                  'cpu_only': True,
                  'save_folder': 'adhesin_ptycho',
                  'phantom_path': 'adhesin_ptycho/phantom',
                  'multiscale_level': 1,
                  'n_epoch_final_pass': None,
                  'save_intermediate': True,
                  'full_intermediate': True,
                  'n_dp_batch': 50,
                  'probe_type': 'gaussian',
                  'probe_options': {'probe_mag_sigma': 10,
                                    'probe_phase_sigma': 10,
                                    'probe_phase_max': 0.5},
                  'finite_support_mask': None,
                  'probe_pos': [(y, x) for y in np.linspace(9, 55, 23, dtype=int) for x in np.linspace(9, 55, 23, dtype=int)]}

params_cone = {'fname': 'data_cone_256_1nm_marc.h5',
               'theta_st': 0,
               'theta_end': 2 * np.pi,
               'n_epochs': 1,
               'obj_size': (256, 256, 256),
               'alpha_d': 0,
               'alpha_b': 0,
               'gamma': 0,
               'probe_size': (72, 72),
               'learning_rate': 1e-7,
               'center': 128,
               'energy_ev': 5000,
               'psize_cm': 1.e-7,
               'batch_size': 1,
               'n_batch_per_update': 1,
               'output_folder': 'test',
               'cpu_only': True,
               'save_folder': 'cone_256_filled_ptycho',
               'phantom_path': 'cone_256_filled_ptycho/phantom',
               'multiscale_level': 1,
               'n_epoch_final_pass': None,
               'save_intermediate': True,
               'full_intermediate': True,
               'n_dp_batch': 100,
               'probe_type': 'gaussian',
               'probe_options': {'probe_mag_sigma': 6,
                                 'probe_phase_sigma': 6,
                                 'probe_phase_max': 0.5},
               # 'probe_pos': [(y, x) for y in np.linspace(18, 120, 35, dtype=int) for x in np.linspace(54, 198, 49, dtype=int)] +
               #              [(y, x) for y in np.linspace(120, 222, 35, dtype=int) for x in np.linspace(22, 230, 70, dtype=int)],
               'probe_pos': [(y, x) for y in np.arange(23) * 12 for x in np.arange(23) * 12],
               'finite_support_mask': dxchange.read_tiff('cone_256_filled_ptycho/mask.tiff')
               }

params = params_cone

# init_delta = np.load('adhesin_ptycho/phantom/grid_delta.npy')
# init_beta = np.load('adhesin_ptycho/phantom/grid_beta.npy')
# init = [init_delta, init_beta]


reconstruct_ptychography(fname=params['fname'],
                         probe_pos=params['probe_pos'],
                         probe_size=params['probe_size'],
                         theta_st=0,
                         theta_end=params['theta_end'],
                         obj_size=params['obj_size'],
                         n_epochs=params['n_epochs'],
                         crit_conv_rate=0.03,
                         max_nepochs=200,
                         alpha_d=params['alpha_d'],
                         alpha_b=params['alpha_b'],
                         gamma=params['gamma'],
                         learning_rate=params['learning_rate'],
                         output_folder=params['output_folder'],
                         minibatch_size=params['batch_size'],
                         save_intermediate=params['save_intermediate'],
                         full_intermediate=params['full_intermediate'],
                         energy_ev=params['energy_ev'],
                         psize_cm=params['psize_cm'],
                         cpu_only=params['cpu_only'],
                         save_path=params['save_folder'],
                         phantom_path=params['phantom_path'],
                         multiscale_level=params['multiscale_level'],
                         n_epoch_final_pass=params['n_epoch_final_pass'],
                         initial_guess=None,
                         n_batch_per_update=params['n_batch_per_update'],
                         dynamic_rate=True,
                         probe_type=params['probe_type'],
                         probe_initial=None,
                         probe_learning_rate=1e-3,
                         pupil_function=None,
                         probe_circ_mask=None,
                         n_dp_batch=params['n_dp_batch'],
                         finite_support_mask=params['finite_support_mask'],
                         **params['probe_options'])