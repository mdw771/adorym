from adorym.simulation import *

PI = 3.1415927

#==============================================================#
#             Guidelines for parameter settings                #
#==============================================================#
# 1. Keep 'free_prop_cm = 'inf' for far-field imaging.         #
# 2. Ptychography scan positions are now interpreted as the    #
#    top-left corner of the probe array in object frame, NOT   #
#    center.                                                   #
# 3. Generate fullfield/single-shot images by giving only one  #
#    pair of coordinates in probe_pos. There is no longer      #
#    dedicated script/function for fullfield simulation.       #
# 4. Output file will be saved as a HDF5 at save_path/fname.   #
# 5. probe_type can be 'gaussian', 'plane', or 'fixed'. If set #
#    to 'fixed', you must supply 2D wavefield values to        #
#    probe_init as [probe_mag, probe_phase]. If set to         #
#    'gaussian', you should specify 'probe_mag_sigma',         #
#    'probe_phase_sigma', and 'probe_phase_max'.               #
# 6. Delta and beta of the refractive index of the simulated   #
#    object are read from Numpy data files named as            #
#    'grid_delta.npy' and 'grid_beta.npy' contained in the     #
#    path given in 'phantom_path'.                             #
#==============================================================#

params_adhesin_ff = {'fname': 'data_adhesin_360_soft_4d_2.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_theta': 500,
                  'theta_downsample': 10,
                  'obj_size': (64, 64, 64),
                  'probe_size': (64, 64),
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'phantom_path': '../demos/adhesin/phantom',
                  'minibatch_size': 1,
                  'cpu_only': True,
                  'save_path': '../demos/adhesin',
                  'probe_initial': None,
                  'fresnel_approx': True,
                  'probe_type': 'plane',
                  'forward_algorithm': 'fresnel',
                  'object_type': 'normal',
                  'probe_pos': [(0, 0)],
                  'free_prop_cm': None,
                  'normalize_fft': False
                  }

params_adhesin_2 = {'fname': 'data_adhesin_64_1nm_1um_2.h5',
                  'theta_st': 0,
                  'theta_end': 2 * np.pi,
                  'n_theta': 500,
                  'theta_downsample': None,
                  'obj_size': (64, 64, 64),
                  'probe_size': (72, 72),
                  'energy_ev': 800,
                  'psize_cm': 0.67e-7,
                  'phantom_path': '../demos/adhesin/phantom',
                  'minibatch_size': 23,
                  'cpu_only': True,
                  'save_path': '../demos/adhesin_ptycho_2',
                  'probe_initial': None,
                  'fresnel_approx': True,
                  'probe_type': 'gaussian',
                  'forward_algorithm': 'fresnel',
                  'object_type': 'normal',
                  'probe_pos': [(y, x) for y in np.linspace(-27, 19, 23, dtype=int) for x in np.linspace(-27, 19, 23, dtype=int)],
                  'probe_mag_sigma': 6,
                  'probe_phase_sigma': 6,
                  'probe_phase_max': 0.5,
                  'free_prop_cm': 'inf',
                  'normalize_fft': False
                  }

pos_error = np.round(np.random.normal(0, 1, [500, 2])).astype(int)
pos_error = np.clip(pos_error, -4, 4)
params_adhesin_2_noisy_pos = {'fname': 'data_adhesin_64_1nm_1um.h5',
                              'theta_st': 0,
                              'theta_end': 2 * np.pi,
                              'n_theta': 500,
                              'theta_downsample': None,
                              'obj_size': (64, 64, 64),
                              'probe_size': (72, 72),
                              'energy_ev': 800,
                              'psize_cm': 0.67e-7,
                              'phantom_path': 'adhesin/phantom',
                              'minibatch_size': 23,
                              'cpu_only': True,
                              'save_path': 'adhesin_ptycho_pos_error',
                              'probe_initial': None,
                              'fresnel_approx': True,
                              'probe_type': 'gaussian',
                              'forward_algorithm': 'fresnel',
                              'object_type': 'normal',
                              'probe_pos': [(y, x) for y in np.linspace(-27, 19, 23, dtype=int) for x in np.linspace(-27, 19, 23, dtype=int)],
                              'probe_mag_sigma': 6,
                              'probe_phase_sigma': 6,
                              'probe_phase_max': 0.5,
                              'free_prop_cm': 'inf',
                              'pos_offset_vec': pos_error,
                              'normalize_fft': False
                              }

try:
    probe_pos_cameraman = np.array([(y, x) for y in np.arange(-10, 246, 5) for x in np.arange(-10, 246, 5)])
    probe_pos_cameraman_err = np.round(np.random.normal(0, 1, probe_pos_cameraman.shape)).astype(int)
    probe_pos_cameraman_err = np.clip(probe_pos_cameraman_err, -3, 3)
    probe_pos_cameraman += probe_pos_cameraman_err
    params_cameraman_noisy_pos = {'fname': 'data_cameraman.h5',
                                  'theta_st': 0,
                                  'theta_end': 0,
                                  'n_theta': 1,
                                  'theta_downsample': None,
                                  'obj_size': (256, 256, 1),
                                  'probe_size': (72, 72),
                                  'energy_ev': 5000,
                                  'psize_cm': 1e-7,
                                  'phantom_path': 'cameraman_pos_error/phantom',
                                  'minibatch_size': 2704,
                                  'cpu_only': True,
                                  'save_path': 'cameraman_pos_error',
                                  'probe_initial': [dxchange.read_tiff('cameraman_pos_error/probe_mag.tiff'), dxchange.read_tiff('cameraman_pos_error/probe_phase.tiff')],
                                  'fresnel_approx': True,
                                  'probe_type': 'fixed',
                                  'forward_algorithm': 'fresnel',
                                  'object_type': 'phase_only',
                                  'probe_pos': probe_pos_cameraman,
                                  'probe_mag_sigma': 6,
                                  'probe_phase_sigma': 6,
                                  'probe_phase_max': 0.5,
                                  'free_prop_cm': 'inf',
                                  'pos_offset_vec': None,
                                  'normalize_fft': False
                                  }
except: pass

try:
    params_cone_marc = {'fname': 'data_cone_256_foam_1nm_new.h5',
                        'theta_st': 0,
                        'theta_end': 2 * np.pi,
                        'n_theta': 500,
                        'theta_downsample': None,
                        'obj_size': (256, 256, 256),
                        'probe_size': (72, 72),
                        'energy_ev': 5000,
                        'psize_cm': 1.e-7,
                        'phantom_path': '/home/beams/B282788/Data/programs/adorym_tests/cone_256_foam_ptycho/phantom',
                        'minibatch_size': 1,
                        'save_path': '../demos/cone_256_foam_ptycho',
                        'initial_guess': None,
                        'probe_type': 'gaussian',
                        'forward_algorithm': 'fresnel',
                        'probe_pos': [(y, x) for y in (np.arange(23) * 12) - 36 for x in (np.arange(23) * 12) - 36],
                        'probe_mag_sigma': 6,
                        'probe_phase_sigma': 6,
                        'probe_phase_max': 0.5,
                        'free_prop_cm': 'inf',
                        'normalize_fft': False
                        }
except: pass

try:
    params_cone_ff = {'fname': 'data_cone_256_1nm_1um_2.h5',
                      'theta_st': 0,
                      'theta_end': 0,
                      'n_theta': 1,
                      'theta_downsample': 100,
                      'obj_size': (256, 256, 256),
                      'probe_size': (256, 256),
                      'energy_ev': 5000,
                      'psize_cm': 1.e-7,
                      'phantom_path': '/home/beams/B282788/Data/programs/adorym_tests/cone_256_foam/phantom',
                      'minibatch_size': 1,
                      'save_path': '../demos/cone_256_foam',
                      'initial_guess': None,
                      'probe_type': 'plane',
                      'forward_algorithm': 'fresnel',
                      'probe_pos': [(0, 0)],
                      'free_prop_cm': None
                        }
except: pass


params_2d_cell = {'fname': 'data_cell_phase.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'n_theta': 1,
                    'theta_downsample': 1,
                    'obj_size': (325, 325, 1),
                    'probe_size': (72, 72),
                    'energy_ev': 5000,
                    'psize_cm': 1.e-7,
                    'phantom_path': 'cell/ptychography/phantom',
                    'minibatch_size': 66,
                    'n_batch_per_update': 1,
                    'cpu_only': True,
                    'save_path': 'cell/test',
                    'probe_type': 'gaussian',
                    'probe_mag_sigma': 6,
                    'probe_phase_sigma': 6,
                    'probe_phase_max': 0.5,
                    'forward_algorithm': 'fresnel',
                    'object_type': 'phase_only',
                    'probe_pos': [(y, x) for y in (np.arange(66) * 5) - 36 for x in (np.arange(68) * 5) - 36],
                    'free_prop_cm': 'inf',
                    'normalize_fft': False
                    }

# params = params_2d_cell
# params = params_adhesin_2_noisy_pos
# params = params_cameraman_noisy_pos
params = params_cone_ff
# params = params_adhesin_2
# params = params_cone_marc
# params = params_adhesin_ff

simulate_ptychography(**params)