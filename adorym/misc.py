import os
import numpy as np
import glob
import dxchange
import re
import sys
import datetime
import adorym.global_settings as global_settings

def check_config_indept_mpi():
    d = {}
    try:
        f = open(os.path.join(os.getcwd(), 'project_config.txt'), 'r')
        for l in f.readlines():
            if '\n' in l:
                l = l[:l.find('\n')]
            k, v = l.split(' = ')
            if v == 'True':
                v = True
            elif v == 'False':
                v = False
            else:
                v = float(v)
            d[k] = v
    except:
        pass
    return d

project_config = check_config_indept_mpi()
try:
    independent_mpi = project_config['independent_mpi']
except:
    independent_mpi = False

try:
    if independent_mpi:
        raise Exception
    from mpi4py import MPI
except:
    from adorym.pseudo import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

SUMMARY_PRESET_PTYCHO = ['obj_size',
                         'probe_size',
                         'output_folder',
                         'theta_downsample',
                         'n_theta',
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
                         'probe_type',
                         'optimize_probe_defocusing',
                         'probe_defocusing_learning_rate',
                         'optimizer_probe_defocusing',
                         'optimize_all_probe_pos',
                         'all_probe_pos_learning_rate',
                         'optimizer_all_probe_pos',
                         'optimize_probe_pos_offset',
                         'probe_pos_offset_learning_rate',
                         'optimizer_probe_pos_offset',
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


def create_directory_multirank(path):
    """
    Due to the presence of comm.Barrier(), always use this function collectively!
    """

    if not os.path.exists(path):
        comm.Barrier()
        if rank == 0:
            os.makedirs(path)
        comm.Barrier()
    return


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
        if var_name == 'learning_rate' and locals_dict['optimizer'] is not None:
            continue
        elif 'learning_rate' in var_name and locals_dict['optimizer_' + var_name[:-14]] is not None:
            continue
        else:
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
    np.savetxt(os.path.join(path, 'checkpoint.txt'),
               np.array([i_epoch, i_batch]), fmt='%d')
    if distribution_mode is None:
        np.save(os.path.join(path, 'obj_checkpoint.npy'), obj_array)
        if optimizer is not None:
            optimizer.save_param_arrays_to_checkpoint()
    elif distribution_mode == 'distributed_object':
        if obj_array is not None:
            np.save(os.path.join(path, 'obj_checkpoint_rank_{}.npy'.format(rank)), obj_array)
        if optimizer is not None:
            optimizer.save_distributed_param_arrays_to_checkpoint()
    else:
        pass
    return


def restore_checkpoint(output_folder, distribution_mode=None, optimizer=None, dtype='float32'):

    path = os.path.join(output_folder, 'checkpoint')
    i_epoch, i_batch = [int(i) for i in np.loadtxt(os.path.join(path, 'checkpoint.txt'))]
    if distribution_mode is None:
        obj = np.load(os.path.join(path, 'obj_checkpoint.npy'))
        optimizer.restore_param_arrays_from_checkpoint()
        return i_epoch, i_batch, obj
    elif distribution_mode == 'distributed_object':
        obj = np.load(os.path.join(path, 'obj_checkpoint_rank_{}.npy'.format(rank)))
        optimizer.restore_distributed_param_arrays_from_checkpoint(use_numpy=True, dtype=dtype)
        return i_epoch, i_batch, obj
    elif distribution_mode == 'shared_file':
        return i_epoch, i_batch


def parse_source_folder(src_dir, prefix):
    flist = glob.glob(os.path.join(src_dir, prefix + '*.tif*'))
    raw_img = np.squeeze(dxchange.read_tiff(flist[0]))
    raw_img_shape = raw_img.shape
    theta_full_ls = []
    dist_ls = []
    for f in flist:
        i_theta = int(re.findall(r'\d+', f)[-2])
        i_dist = int(re.findall(r'\d+', f)[-1])
        theta_full_ls.append(i_theta)
        dist_ls.append(i_dist)
    theta_ls = np.unique(theta_full_ls)
    n_theta = len(theta_ls)
    n_dist = len(flist) // n_theta
    ind_ls = np.array(theta_full_ls) * n_dist + np.array(dist_ls)
    flist = np.array(flist)[np.argsort(ind_ls)]
    return flist, n_theta, n_dist, raw_img_shape


def print_flush(a, designate_rank=None, this_rank=None, save_stdout=False, same_line=None,
                output_folder='', timestamp='', **kwargs):
    if same_line is None:
        sameline = not global_settings.disable_sameline_output
    a = '[{}][{}] '.format(str(datetime.datetime.today())[:-3], this_rank) + a
    if designate_rank is not None:
        if this_rank == designate_rank:
            if same_line:
                print(a, '\r', end='')
            else:
                print(a)
    else:
        if same_line:
            print(a, '\r', end='')
        else:
            print(a)
    if (designate_rank is None or this_rank == designate_rank) and save_stdout:
        try:
            f = open(os.path.join(output_folder, 'stdout_{}.txt'.format(timestamp)), 'a')
        except:
            os.makedirs(output_folder)
            f = open(os.path.join(output_folder, 'stdout_{}.txt'.format(timestamp)), 'a')
        f.write(a)
        f.write('\n')
        f.close()
    sys.stdout.flush()
    return