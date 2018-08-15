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
                         'fname']


def create_summary(save_path, locals_dict, var_list=None, preset=None):

    if preset == 'ptycho':
        var_list = SUMMARY_PRESET_PTYCHO
    f = open(os.path.join(save_path, 'summary.txt'), 'w')
    for var_name in var_list:
        line = '{:<20}{}\n'.format(var_name, str(locals_dict[var_name]))
        f.write(line)
    f.close()

    return