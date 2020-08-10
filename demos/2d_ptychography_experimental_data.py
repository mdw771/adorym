from adorym.ptychography import reconstruct_ptychography
import adorym
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

output_folder = 'test'
distribution_mode = None
optimizer_obj = adorym.AdamOptimizer('obj', output_folder=output_folder, distribution_mode=distribution_mode,
                                     options_dict={'step_size': 1e-3})
optimizer_probe = adorym.AdamOptimizer('probe', output_folder=output_folder, distribution_mode=distribution_mode,
                                        options_dict={'step_size': 1e-3, 'eps': 1e-7})
optimizer_all_probe_pos = adorym.AdamOptimizer('probe_pos_correction', output_folder=output_folder, distribution_mode=distribution_mode,
                                               options_dict={'step_size': 1e-2})

params_2idd_gpu = {'fname': 'data.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'n_epochs': 1000,
                    'obj_size': (618, 606, 1),
                    'two_d_mode': True,
                    'energy_ev': 8801.121930115722,
                    'psize_cm': 1.32789376566526e-06,
                    'minibatch_size': 35,
                    'output_folder': 'test',
                    # 'output_folder': 'rec_ukp_perline_posopt1_olr1e-3_defoc_modes5',
                    'cpu_only': False,
                    'save_path': '../demos/siemens_star_aps_2idd',
                    'use_checkpoint': False,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    'initial_guess': None,
                    'random_guess_means_sigmas': (1., 0., 0.001, 0.002),
                    'n_dp_batch': 350,
                    # ===============================
                    'probe_type': 'aperture_defocus',
                    'n_probe_modes': 5,
                    'aperture_radius': 10,
                    'beamstop_radius': 5,
                    'probe_defocus_cm': 0.0069,
                    # ===============================
                    'rescale_probe_intensity': True,
                    'free_prop_cm': 'inf',
                    'backend': 'pytorch',
                    'raw_data_type': 'intensity',
                    'beamstop': None,
                    'optimizer': optimizer_obj,
                    'optimize_probe': True,
                    'optimizer_probe': optimizer_probe,
                    'optimize_all_probe_pos': True,
                    'optimizer_all_probe_pos': optimizer_all_probe_pos,
                    'save_history': True,
                    'update_scheme': 'immediate',
                    'unknown_type': 'real_imag',
                    'save_stdout': True,
                    'loss_function_type': 'lsq',
                    'normalize_fft': False
                    }

params = params_2idd_gpu

reconstruct_ptychography(**params)
