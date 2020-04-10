import os
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

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
                         'cpu_only',
                         'optimizer',
                         'probe_mag_sigma',
                         'probe_phase_sigma',
                         'probe_phase_max',
                         'probe_learning_rate',
                         'probe_learning_rate_init',
                         'probe_type',
                         'optimize_probe_defocusing',
                         'probe_defocusing_learning_rate',
                         'optimize_all_probe_pos',
                         'all_probe_pos_learning_rate',
                         'optimize_probe_pos_offset',
                         'probe_pos_offset_learning_rate',
                         'distribution_mode',
                         'n_ranks',
                         'backend',
                         'reweighted_l1',
                         'update_scheme',
                         'unknown_type',
                         'randomize_probe_pos',
                         'normalize_fft',
                         'n_probe_modes',
                         'loss_function_type',
                         'slice_pos_cm_ls',
                         ]

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
                     'shrink_cycle',
                     'fname',
                     'object_type',
                     'distribution_mode',
                     'n_blocks_x',
                     'n_blocks_y',
                     'block_size',
                     'safe_zone_width',
                     'reweighted_l1',
                     'n_ranks']


def create_summary(save_path, locals_dict, var_list=None, preset=None, verbose=True):

    if preset == 'ptycho':
        var_list = SUMMARY_PRESET_PTYCHO
    elif preset == 'pp':
        var_list = SUMMARY_PRESET_PP
    elif preset == 'fullfield':
        var_list = SUMMARY_PRESET_FF
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, 'summary.txt'), 'w')
    print('============== PARAMETERS ==============')
    for var_name in var_list:
        try:
            line = '{:<30}{}\n'.format(var_name, str(locals_dict[var_name]))
            if verbose:
                print(line)
            f.write(line)
        except:
            pass
    print('========================================')
    f.close()
    return


def save_checkpoint(i_epoch, i_batch, output_folder, distribution_mode=None, obj_array=None, optimizer=None):

    path = os.path.join(output_folder, 'checkpoint')
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(os.path.join(path, 'checkpoint.txt'),
               np.array([i_epoch, i_batch]), fmt='%d')
    if distribution_mode is None:
        np.save(os.path.join(path, 'obj_checkpoint.npy'), obj_array)
        optimizer.save_param_arrays_to_checkpoint()
    elif distribution_mode == 'distributed_object':
        np.save(os.path.join(path, 'obj_checkpoint_rank_{}.npy'.format(rank)), obj_array)
        optimizer.save_distributed_param_arrays_to_checkpoint()
    else:
        pass
    return


def restore_checkpoint(output_folder, distribution_mode=None, optimizer=None):

    i_epoch, i_batch = [int(i) for i in np.loadtxt(os.path.join(output_folder, 'checkpoint.txt'))]
    if distribution_mode is None:
        obj = np.load(os.path.join(output_folder, 'obj_checkpoint.npy'))
        obj_delta = np.take(obj, 0, axis=-1)
        obj_beta = np.take(obj, 1, axis=-1)
        optimizer.restore_param_arrays_from_checkpoint()
        return i_epoch, i_batch, obj_delta, obj_beta
    elif distribution_mode == 'distributed_object':
        obj = np.load(os.path.join(output_folder, 'obj_checkpoint_rank_{}.npy'.format(rank)))
        obj_delta = np.take(obj, 0, axis=-1)
        obj_beta = np.take(obj, 1, axis=-1)
        optimizer.restore_distributed_param_arrays_from_checkpoint()
        return i_epoch, i_batch, obj_delta, obj_beta
    else:
        return i_epoch, i_batch
