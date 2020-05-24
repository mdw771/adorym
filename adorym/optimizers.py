import numpy as np
import os
import h5py
import pickle
try:
    from mpi4py import MPI
except:
    from adorym.pseudo import MPI

from adorym.util import *
from adorym.array_ops import ObjectFunction, Gradient
import adorym.wrappers as w
import adorym.global_settings as global_settings

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

class Optimizer(object):

    def __init__(self, name, whole_object_size, output_folder='.', params_list=(), distribution_mode=None, options_dict=None):
        """
        :param whole_object_size: List of int; 4-D vector for object function (including 2 channels),
                                  or a 3-D vector for probe, or a 1-D scalar for other variables.
                                  Channel must be the last domension. Parameter arrays will be created
                                  following exactly whole_object_size.
        :param params_list: List of str; a list of optimizer parameters provided in strings.
        """
        self.name = name
        self.whole_object_size = whole_object_size
        self.output_folder = output_folder
        self.params_list = params_list
        self.params_dset_dict = {}
        self.params_file_pointer_dict = {}
        self.params_whole_array_dict = {}
        self.params_chunk_array_dict = {}
        self.params_chunk_array_0_dict = {}
        self.i_batch = 0
        self.index_in_grad_returns = None
        self.slice_catalog = None
        self.distribution_mode = distribution_mode
        self.options_dict = options_dict
        self.grads = None # Object gradient should be saved in Gradient class, not here.
        if distribution_mode == 'distributed_object':
            self.slice_catalog = get_multiprocess_distribution_index(whole_object_size[0], n_ranks)
        return

    def create_container(self, use_checkpoint, device_obj):
        if self.distribution_mode == 'shared_file':
            self.create_file_objects(use_checkpoint=use_checkpoint)
        elif self.distribution_mode == 'distributed_object':
            self.create_distributed_param_arrays()
        elif self.distribution_mode is None:
            self.create_param_arrays(device=device_obj)

    def create_file_objects(self, use_checkpoint=False):

        if len(self.params_list) > 0:
            for param_name in self.params_list:
                fmode = 'a' if use_checkpoint else 'w'
                try:
                    self.params_file_pointer_dict[param_name] = h5py.File(os.path.join(self.output_folder, 'intermediate_{}.h5'.format(param_name)), fmode, driver='mpio', comm=comm)
                    print_flush('Created intermediate file: {}'.format(os.path.join(self.output_folder, 'intermediate_{}.h5'.format(param_name))), 0, rank)
                except:
                    self.params_file_pointer_dict[param_name] = h5py.File(os.path.join(self.output_folder, 'intermediate_{}.h5'.format(param_name)), fmode)
                try:
                    dset_p = self.params_file_pointer_dict[param_name].create_dataset('obj', shape=self.whole_object_size,
                                                                                      dtype='float64', data=np.zeros(self.whole_object_size))
                except:
                    dset_p = self.params_file_pointer_dict[param_name]['obj']
                # if rank == 0: dset_p[...] = 0
                self.params_dset_dict[param_name] = dset_p
        return

    def create_param_arrays(self, device=None):

        if len(self.params_list) > 0:
            for param_name in self.params_list:
                self.params_whole_array_dict[param_name] = w.zeros(self.whole_object_size, device=device)
        return

    def create_distributed_param_arrays(self):

        if len(self.params_list) > 0 and self.slice_catalog[rank] is not None:
            for param_name in self.params_list:
                self.params_whole_array_dict[param_name] = w.zeros([self.slice_catalog[rank][1] - self.slice_catalog[rank][0], *self.whole_object_size[1:]])
        return

    def restore_param_arrays_from_checkpoint(self, device=None):

        if len(self.params_list) > 0:
            arr = np.load(os.path.join(self.output_folder, 'checkpoint', 'opt_params_checkpoint.npy'))
            arr = w.create_variable(arr, device=device)
            if len(self.params_list) > 0:
                for i, param_name in enumerate(self.params_list):
                    self.params_whole_array_dict[param_name] = arr[i]
        return

    def restore_distributed_param_arrays_from_checkpoint(self, device=None):

        if len(self.params_list) > 0:
            arr = np.load(os.path.join(self.output_folder, 'checkpoint', 'opt_params_checkpoint_rank_{}.npy'.format(rank)))
            arr = w.create_variable(arr, device=device)
            if len(self.params_list) > 0:
                for i, param_name in enumerate(self.params_list):
                    self.params_whole_array_dict[param_name] = arr[i]
        return

    def save_param_arrays_to_checkpoint(self):

        path = os.path.join(self.output_folder, 'checkpoint')
        create_directory_multirank(path)
        if len(self.params_list) > 0:
            arr = []
            for i, param_name in enumerate(self.params_list):
                arr.append(self.params_whole_array_dict[param_name])
            arr = w.stack(arr)
            np.save(os.path.join(path, 'opt_params_checkpoint.npy'), w.to_numpy(arr))
        return

    def save_distributed_param_arrays_to_checkpoint(self):

        path = os.path.join(self.output_folder, 'checkpoint')
        if not os.path.exists(path):
            os.makedirs(path)
        if len(self.params_list) > 0:
            arr = []
            for i, param_name in enumerate(self.params_list):
                arr.append(self.params_whole_array_dict[param_name])
            arr = w.stack(arr)
            np.save(os.path.join(path, 'opt_params_checkpoint_rank_{}.npy'.format(rank)), w.to_numpy(arr))
        return

    def get_params_from_file(self, this_pos_batch=None, probe_size=None):

        for param_name, dset_p in self.params_dset_dict.items():
            p = get_rotated_subblocks(dset_p, this_pos_batch, probe_size, self.whole_object_size[:-1])
            self.params_chunk_array_dict[param_name] = p
            self.params_chunk_array_0_dict[param_name] = np.copy(p)
        return

    def write_params_to_file(self, this_pos_batch=None, probe_size=None, n_ranks=1):

        for param_name, p in self.params_chunk_array_dict.items():
            p = w.to_numpy(p)
            p = p - self.params_chunk_array_0_dict[param_name]
            p /= n_ranks
            dset_p = self.params_dset_dict[param_name]
            write_subblocks_to_file(dset_p, this_pos_batch, np.take(p, 0, axis=-1), np.take(p, 1, axis=-1),
                                    probe_size, self.whole_object_size[:-1], monochannel=False)
        return

    def rotate_files(self, coords, interpolation='bilinear'):

        for param_name, dset_p in self.params_dset_dict.items():
            apply_rotation_to_hdf5(dset_p, coords, rank, n_ranks, interpolation=interpolation, monochannel=False)

    def rotate_arrays(self, coords, interpolation='bilinear'):

        for param_name, arr in self.params_whole_array_dict.items():
            self.params_whole_array_dict[param_name] = apply_rotation(arr, coords, interpolation=interpolation)
        return

    def set_index_in_grad_return(self, ind):
        self.index_in_grad_returns = ind


class AdamOptimizer(Optimizer):

    def __init__(self, name, whole_object_size, output_folder='.', distribution_mode=None, options_dict=None):
        super(AdamOptimizer, self).__init__(name, whole_object_size, output_folder=output_folder, params_list=['m', 'v'],
                                            distribution_mode=distribution_mode, options_dict=options_dict)
        return

    def apply_gradient(self, x, g, i_batch, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, distribution_mode=False,
                       m=None, v=None, return_moments=False, update_batch_count=True, **kwargs):

        if m is None or v is None:
            if distribution_mode == 'shared_file':
                m = self.params_chunk_array_dict['m']
                v = self.params_chunk_array_dict['v']
            else:
                m = self.params_whole_array_dict['m']
                v = self.params_whole_array_dict['v']
        m = (1 - b1) * g + b1 * m  # First moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i_batch + 1))
        d = step_size * mhat / (w.sqrt(vhat) + eps)
        x = x - d
        if distribution_mode == 'shared_file':
            self.params_chunk_array_dict['m'] = m
            self.params_chunk_array_dict['v'] = v
        else:
            self.params_whole_array_dict['m'] = m
            self.params_whole_array_dict['v'] = v
        if update_batch_count:
            self.i_batch += 1
        del mhat, vhat
        if return_moments:
            return x, m, v
        else:
            return x

    def apply_gradient_to_file(self, obj, gradient, i_batch=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, **kwargs):

        assert isinstance(obj, ObjectFunction)
        assert isinstance(gradient, Gradient)
        s = obj.dset.shape
        slice_ls = range(rank, s[0], n_ranks)
        if i_batch is None: i_batch = self.i_batch

        backend_temp = global_settings.backend
        global_settings.backend = 'autograd'

        for i_slice in slice_ls:
            x = obj.dset[i_slice]
            g = gradient.dset[i_slice] / n_ranks
            m = self.params_dset_dict['m'][i_slice]
            v = self.params_dset_dict['v'][i_slice]
            x, m, v = self.apply_gradient(x, g, i_batch, step_size=step_size,
                                    b1=b1, b2=b2, eps=eps, shared_file_object=False,
                                    m=m, v=v, update_batch_count=False, return_moments=True)

            obj.dset[i_slice] = x
            self.params_dset_dict['m'][i_slice] = m
            self.params_dset_dict['v'][i_slice] = v
        self.i_batch += 1
        global_settings.backend = backend_temp

class GDOptimizer(Optimizer):

    def __init__(self, name, whole_object_size, output_folder='.', distribution_mode=None, options_dict=None):
        super(GDOptimizer, self).__init__(name, whole_object_size, output_folder=output_folder, params_list=[],
                                          distribution_mode=distribution_mode, options_dict=options_dict)
        return

    def apply_gradient(self, x, g, i_batch, step_size=0.001, dynamic_rate=True, first_downrate_iteration=92, **kwargs):
        if dynamic_rate:
            threshold_iteration = first_downrate_iteration
            i = 1
            while threshold_iteration < i_batch:
                threshold_iteration += first_downrate_iteration * 2 ** i
                i += 1
                step_size /= 2.
                print_flush('  -- Step size halved.', 0, comm.Get_rank(), save_stdout=False)
        x = x - step_size * g

        return x

    def apply_gradient_to_file(self, obj, gradient, i_batch=None, step_size=0.001, dynamic_rate=True, first_downrate_iteration=92, **kwargs):

        assert isinstance(obj, ObjectFunction)
        assert isinstance(gradient, Gradient)
        s = obj.dset.shape
        slice_ls = range(rank, s[0], n_ranks)
        if i_batch is None: i_batch = self.i_batch

        backend_temp = global_settings.backend
        global_settings.backend = 'autograd'
        for i_slice in slice_ls:
            x = obj.dset[i_slice]
            g = gradient.dset[i_slice]
            x = self.apply_gradient(x, g, i_batch, step_size=step_size,
                                    dynamic_rate=dynamic_rate, first_downrate_iteration=first_downrate_iteration)
            obj.dset[i_slice] = x
        self.i_batch += 1
        global_settings.backend = backend_temp


def apply_gradient_adam(x, g, i_batch, m=None, v=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, **kwargs):

    g = np.array(g)
    if m is None or v is None:
        m = np.zeros_like(x)
        v = np.zeros_like(v)
    m = (1 - b1) * g + b1 * m  # First moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i_batch + 1))
    d = step_size * mhat / (np.sqrt(vhat) + eps)
    x = x - d
    return x, m, v


def apply_gradient_gd(x, g, step_size=0.001, dynamic_rate=True, i_batch=0, first_downrate_iteration=92):

    g = np.array(g)
    if dynamic_rate:
        threshold_iteration = first_downrate_iteration
        i = 1
        while threshold_iteration < i_batch:
            threshold_iteration += first_downrate_iteration * 2 ** i
            i += 1
            step_size /= 2.
            print_flush('  -- Step size halved.', 0, comm.Get_rank(), save_stdout=False)
    x = x - step_size * g

    return x


def load_params_checkpoint(path):
    f_pcp = open(path, 'rb')
    a = pickle.load(f_pcp)
    f_pcp.close()
    return a


def save_params_checkpoint(path, params):
    f_pcp = open(path, 'wb')
    pickle.dump(params, f_pcp)
    f_pcp.close()
    return


def create_and_initialize_parameter_optimizers(optimizable_params, kwargs):

    opt_ls = kwargs['opt_ls']
    forward_model = kwargs['forward_model']
    output_folder = kwargs['output_folder']
    device_obj = kwargs['device_obj']
    n_probe_modes = kwargs['n_probe_modes']
    probe_size = kwargs['probe_size']

    opt_args_ls = [0]
    if kwargs['optimize_probe']:
        optimizer_options_probe = {'step_size': kwargs['probe_learning_rate']}
        opt_probe = AdamOptimizer('probe', [n_probe_modes, *probe_size, 2], output_folder=output_folder,
                                  options_dict=optimizer_options_probe)
        opt_probe.create_param_arrays(device=device_obj)
        opt_probe.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls = opt_args_ls + [forward_model.get_argument_index('probe_real'),
                                     forward_model.get_argument_index('probe_imag')]
        opt_ls.append(opt_probe)

    # Except probe, optimizer name must match the name of the variable to be optimized.
    if kwargs['optimize_probe_defocusing']:
        optimizer_options_probe_defocus = {'step_size': kwargs['probe_defocusing_learning_rate']}
        opt_probe_defocus = AdamOptimizer('probe_defocus_mm', [1], output_folder=output_folder, options_dict=optimizer_options_probe_defocus)
        opt_probe_defocus.create_param_arrays(device=device_obj)
        opt_probe_defocus.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('probe_defocus_mm'))
        opt_ls.append(opt_probe_defocus)

    if kwargs['optimize_probe_pos_offset']:
        assert kwargs['optimize_all_probe_pos'] == False
        optimizer_options_probe_pos_offset = {'step_size': kwargs['probe_pos_offset_learning_rate'],
                                              'dynamic_rate': False}
        opt_probe_pos_offset = GDOptimizer('probe_pos_offset', optimizable_params['probe_pos_offset'].shape, output_folder=output_folder,
                                           options_dict=optimizer_options_probe_pos_offset)
        opt_probe_pos_offset.create_param_arrays(device=device_obj)
        opt_probe_pos_offset.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('probe_pos_offset'))
        opt_ls.append(opt_probe_pos_offset)

    if kwargs['optimize_all_probe_pos']:
        assert kwargs['optimize_probe_pos_offset'] == False
        optimizer_options_probe_pos = {'step_size': kwargs['all_probe_pos_learning_rate']}
        opt_probe_pos = AdamOptimizer('probe_pos_correction', optimizable_params['probe_pos_correction'].shape, output_folder=output_folder,
                                      options_dict=optimizer_options_probe_pos)
        opt_probe_pos.create_param_arrays(device=device_obj)
        opt_probe_pos.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('probe_pos_correction'))
        opt_ls.append(opt_probe_pos)

    if kwargs['is_sparse_multislice']:
        if kwargs['optimize_slice_pos']:
            optimizer_options_slice_pos = {'step_size': kwargs['slice_pos_learning_rate']}
            opt_slice_pos = AdamOptimizer('slice_pos_cm_ls', optimizable_params['slice_pos_cm_ls'].shape, output_folder=output_folder,
                                          options_dict=optimizer_options_slice_pos)
            opt_slice_pos.create_param_arrays(device=device_obj)
            opt_slice_pos.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls.append(forward_model.get_argument_index('slice_pos_cm_ls'))
            opt_ls.append(opt_slice_pos)

    if kwargs['is_multi_dist']:
        if kwargs['optimize_free_prop']:
            optimizer_options_free_prop = {'step_size': kwargs['free_prop_learning_rate']}
            opt_free_prop = AdamOptimizer('free_prop_cm', optimizable_params['free_prop_cm'].shape, output_folder=output_folder,
                                          options_dict=optimizer_options_free_prop)
            opt_free_prop.create_param_arrays(device=device_obj)
            opt_free_prop.set_index_in_grad_return(len(opt_args_ls))
            opt_args_ls.append(forward_model.get_argument_index('free_prop_cm'))
            opt_ls.append(opt_free_prop)

    if kwargs['optimize_tilt']:
        optimizer_options_tilt = {'step_size': kwargs['tilt_learning_rate']}
        opt_tilt = AdamOptimizer('tilt_ls', optimizable_params['tilt_ls'].shape, output_folder=output_folder,
                                 options_dict=optimizer_options_tilt)
        opt_tilt.create_param_arrays(device=device_obj)
        opt_tilt.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('tilt_ls'))
        opt_ls.append(opt_tilt)

    if kwargs['optimize_prj_affine']:
        optimizer_options_prj_scale = {'step_size': kwargs['prj_affine_learning_rate']}
        opt_prj_affine = AdamOptimizer('prj_affine_ls', optimizable_params['prj_affine_ls'].shape, output_folder=output_folder,
                                       options_dict=optimizer_options_prj_scale)
        opt_prj_affine.create_param_arrays(device=device_obj)
        opt_prj_affine.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('prj_affine_ls'))
        opt_ls.append(opt_prj_affine)

    if kwargs['optimize_ctf_lg_kappa']:
        optimizer_options_ctf_lg_kappa = {'step_size': kwargs['ctf_lg_kappa_learning_rate']}
        opt_ctf_lg_kappa = AdamOptimizer('ctf_lg_kappa', optimizable_params['ctf_lg_kappa'].shape, output_folder=output_folder,
                                         options_dict=optimizer_options_ctf_lg_kappa)
        opt_ctf_lg_kappa.create_param_arrays(device=device_obj)
        opt_ctf_lg_kappa.set_index_in_grad_return(len(opt_args_ls))
        opt_args_ls.append(forward_model.get_argument_index('ctf_lg_kappa'))
        opt_ls.append(opt_ctf_lg_kappa)

    return opt_ls, opt_args_ls


def initialize_parameter_gradients(opt_ls, device=None):

    for opt in opt_ls:
        if opt.name =='obj':
            continue
        else:
            opt.grads = w.zeros(opt.whole_object_size, requires_grad=False, device=device)
    return opt_ls


def update_parameter_gradients(opt_ls, grads):

    for opt in opt_ls:
        if opt.name == 'obj':
            continue
        elif opt.name == 'probe':
            opt.grads += w.stack(grads[1:3], axis=-1)
        else:
            opt.grads += grads[opt.index_in_grad_returns]
    return opt_ls


def update_parameters(opt_ls, optimizable_params, kwargs):

    i_epoch = kwargs['i_epoch']
    i_batch = kwargs['i_batch']
    n_batch = kwargs['n_batch']
    other_params_update_delay = kwargs['other_params_update_delay']
    probe_update_delay = kwargs['probe_update_delay']
    probe_update_limit = kwargs['probe_update_limit']
    i_full_angle = kwargs['i_full_angle']
    stdout_options = kwargs['stdout_options']

    if probe_update_limit is None:
        probe_update_limit = np.inf

    for opt in opt_ls:
        if opt.name == 'obj':
            continue
        elif opt.name == 'probe':
            if i_batch + i_epoch * n_batch >= probe_update_delay and i_batch + i_epoch * n_batch < probe_update_limit:
                with w.no_grad():
                    opt.grads = comm.allreduce(opt.grads)
                    probe_temp = opt.apply_gradient(w.stack([optimizable_params['probe_real'], optimizable_params['probe_imag']], axis=-1), opt.grads,
                                                          i_full_angle, **opt.options_dict)
                    optimizable_params['probe_real'], optimizable_params['probe_imag'] = w.split_channel(probe_temp)
                    del opt.grads, probe_temp
                w.reattach(optimizable_params['probe_real'])
                w.reattach(optimizable_params['probe_imag'])
            else:
                print_flush('Probe is not updated because current batch is out of the specified range ({}, {}).'.format(
                    probe_update_delay, probe_update_limit), 0, rank, **stdout_options)

        elif i_batch + i_epoch * n_batch >= other_params_update_delay:

            if opt.name == 'probe_pos_correction':
                with w.no_grad():
                    opt.grads = comm.allreduce(opt.grads)
                    probe_pos_correction = optimizable_params['probe_pos_correction']
                    probe_pos_correction = opt.apply_gradient(probe_pos_correction, opt.grads, i_full_angle,
                                                                        **opt.options_dict)
                    # Prevent position drifting
                    slicer = tuple(range(len(probe_pos_correction.shape) - 1))
                    optimizable_params['probe_pos_correction'] = probe_pos_correction - w.mean(probe_pos_correction, axis=slicer)
                w.reattach(optimizable_params['probe_pos_correction'])

            elif opt.name == 'slice_pos_cm_ls':
                with w.no_grad():
                    opt.grads = comm.allreduce(opt.grads)
                    slice_pos_cm_ls = optimizable_params['slice_pos_cm_ls']
                    slice_pos_cm_ls = opt.apply_gradient(slice_pos_cm_ls, opt.grads, i_full_angle,
                                                                   **opt.options_dict)
                    # Prevent position drifting
                    optimizable_params['slice_pos_cm_ls'] = slice_pos_cm_ls - slice_pos_cm_ls[0]
                w.reattach(optimizable_params['slice_pos_cm_ls'])

            elif opt.name == 'prj_affine_ls':
                with w.no_grad():
                    opt.grads = comm.allreduce(opt.grads)
                    optimizable_params['prj_affine_ls'] = opt.apply_gradient(optimizable_params['prj_affine_ls'], opt.grads, i_full_angle,
                                                                  **opt.options_dict)
                    # Regularize transformation of image 0.
                    optimizable_params['prj_affine_ls'][0, 0, 0] = 1.
                    optimizable_params['prj_affine_ls'][0, 0, 1] = 0.
                    optimizable_params['prj_affine_ls'][0, 0, 2] = 0.
                    optimizable_params['prj_affine_ls'][0, 1, 0] = 0.
                    optimizable_params['prj_affine_ls'][0, 1, 1] = 1.
                    optimizable_params['prj_affine_ls'][0, 1, 2] = 0.
                w.reattach(optimizable_params['prj_affine_ls'])

            else:
                with w.no_grad():
                    opt.grads = comm.allreduce(opt.grads)
                    var = optimizable_params[opt.name]
                    optimizable_params[opt.name] = opt.apply_gradient(var, opt.grads, i_full_angle, **opt.options_dict)
                w.reattach(optimizable_params[opt.name])

        else:
            print_flush(
                'Params are not updated because current epoch is smaller than specified delay ({}).'.format(
                    other_params_update_delay), 0, rank, **stdout_options)
    return optimizable_params


def create_parameter_output_folders(opt_ls, output_folder):

    for opt in opt_ls:
        if opt.name == 'obj':
            continue

        elif opt.name == 'probe':
            create_directory_multirank(os.path.join(output_folder, 'intermediate', 'probe'))

        elif opt.name == 'probe_pos_offset':
            create_directory_multirank(os.path.join(output_folder, 'intermediate', 'probe_pos_offset'))

        elif opt.name == 'probe_pos_correction':
            create_directory_multirank(os.path.join(output_folder, 'intermediate', 'probe_pos'))

        elif opt.name == 'prj_affine_ls':
            create_directory_multirank(os.path.join(output_folder, 'intermediate', 'prj_affine'))

        else:
            create_directory_multirank(os.path.join(output_folder, 'intermediate', opt.name))


def output_intermediate_parameters(opt_ls, optimizable_params, kwargs):

    output_folder = kwargs['output_folder']
    i_epoch = kwargs['i_epoch']
    i_batch = kwargs['i_batch']
    save_history = kwargs['save_history']
    n_theta = kwargs['n_theta']
    is_multi_dist = kwargs['is_multi_dist']

    for opt in opt_ls:
        if opt.name == 'obj':
            continue

        elif opt.name == 'probe':
            output_probe(optimizable_params['probe_real'], optimizable_params['probe_imag'], os.path.join(output_folder, 'intermediate', 'probe'),
                         full_output=False, i_epoch=i_epoch, i_batch=i_batch,
                         save_history=save_history)

        elif opt.name == 'probe_pos_offset':
            f_offset = open(os.path.join(output_folder, 'intermediate', 'probe_pos_offset',
                                         'probe_pos_offset.txt'), 'a' if i_batch > 0 or i_epoch > 0 else 'w')
            f_offset.write('{:4d}, {:4d}, {}\n'.format(i_epoch, i_batch, list(w.to_numpy(optimizable_params['probe_pos_offset']).flatten())))
            f_offset.close()

        elif opt.name == 'probe_pos_correction':
            for i_theta_pos in range(n_theta):
                if is_multi_dist:
                    np.savetxt(os.path.join(output_folder, 'intermediate', 'probe_pos',
                                            'probe_pos_correction_{}_{}.txt'.format(i_epoch, i_batch)),
                               w.to_numpy(optimizable_params['probe_pos_correction']))
                else:
                    np.savetxt(os.path.join(output_folder, 'intermediate', 'probe_pos',
                                            'probe_pos_correction_{}_{}_{}.txt'.format(i_epoch, i_batch, i_theta_pos)),
                               w.to_numpy(optimizable_params['probe_pos_correction'][i_theta_pos]))

        elif opt.name == 'prj_affine_ls':
            np.savetxt(os.path.join(output_folder, 'intermediate', 'prj_affine',
                                    'prj_affine_{}.txt'.format(i_epoch)),
                       np.concatenate(w.to_numpy(optimizable_params['prj_affine_ls']), 0))

        else:
            np.savetxt(os.path.join(output_folder, 'intermediate', opt.name,
                                    '{}_{}.txt'.format(opt.name, i_epoch)),
                       w.to_numpy(optimizable_params[opt.name]))
