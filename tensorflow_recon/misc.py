import os

SUMMARY_PRESET_PTYCHO = ['obj_size',
                         'probe_size',
                         'output_folder',
                         'theta_downsample',
                         'n_theta',
                         'n_pos',
                         'n_epochs',
                         'learning_rate',
                         'alpha_d',
                         'alpha_b',
                         'gamma',
                         'n_dp_batch',
                         'minibatch_size',
                         'free_prop_cm',
                         'psize_cm',
                         'energy_ev',
                         'fname',
                         'probe_mag_sigma',
                         'probe_phase_sigma',
                         'probe_phase_max']

SUMMARY_PRESET_PP = ['obj_size',
                     'output_folder',
                     'theta_downsample',
                     'n_theta',
                     'n_epochs',
                     'learning_rate',
                     'alpha_d',
                     'alpha_b',
                     'gamma',
                     'minibatch_size',
                     'free_prop_cm',
                     'psize_cm',
                     'energy_ev',
                     'fname',
                     'dist_to_source_cm',
                     'det_psize_cm',
                     'theta_max',
                     'phi_max',
                     'probe_type']

SUMMARY_PRESET_FF = ['obj_size',
                     'output_folder',
                     'theta_downsample',
                     'n_theta',
                     'n_epochs',
                     'learning_rate',
                     'alpha_d',
                     'alpha_b',
                     'gamma',
                     'minibatch_size',
                     'free_prop_cm',
                     'psize_cm',
                     'energy_ev',
                     'fname',
                     'object_type']


def create_summary(save_path, locals_dict, var_list=None, preset=None):

    if preset == 'ptycho':
        var_list = SUMMARY_PRESET_PTYCHO
    elif preset == 'pp':
        var_list = SUMMARY_PRESET_PP
    elif preset == 'fullfield':
        var_list = SUMMARY_PRESET_FF
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, 'summary.txt'), 'w')
    for var_name in var_list:
        line = '{:<20}{}\n'.format(var_name, str(locals_dict[var_name]))
        f.write(line)
    f.close()

    return