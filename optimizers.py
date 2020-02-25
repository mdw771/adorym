import autograd.numpy as np
import os
import h5py
from mpi4py import MPI

from util import get_rotated_subblocks, write_subblocks_to_file, print_flush, apply_rotation_to_hdf5, apply_rotation
from array_ops import ObjectFunction, Gradient

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

class Optimizer(object):

    def __init__(self, whole_object_size, output_folder='.', params_list=()):
        """
        :param whole_object_size: List of int; 4-D vector for object function (including 2 channels),
                                  or a 3-D vector for probe, or a 1-D scalar for other variables.
                                  Channel must be the last domension. Parameter arrays will be created
                                  following exactly whole_object_size.
        :param params_list: List of str; a list of optimizer parameters provided in strings.
        """
        self.whole_object_size = whole_object_size
        self.output_folder = output_folder
        self.params_list = params_list
        self.params_dset_dict = {}
        self.params_file_pointer_dict = {}
        self.params_whole_array_dict = {}
        self.params_chunk_array_dict = {}
        self.params_chunk_array_0_dict = {}
        self.i_batch = 0
        return

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

    def create_param_arrays(self):

        if len(self.params_list) > 0:
            for param_name in self.params_list:
                self.params_whole_array_dict[param_name] = np.zeros(self.whole_object_size)
        return

    def restore_param_arrays_from_checkpoint(self):

        arr = np.load(os.path.join(self.output_folder, 'opt_params_checkpoint.npy'))
        if len(self.params_list) > 0:
            for i, param_name in enumerate(self.params_list):
                self.params_whole_array_dict[param_name] = arr[i]
        return

    def save_param_arrays_to_checkpoint(self):

        if len(self.params_list) > 0:
            arr = []
            for i, param_name in enumerate(self.params_list):
                arr.append(self.params_whole_array_dict[param_name])
            arr = np.stack(arr)
            np.save(os.path.join(self.output_folder, 'opt_params_checkpoint.npy'), arr)
        return

    def get_params_from_file(self, this_pos_batch=None, probe_size_half=None):

        for param_name, dset_p in self.params_dset_dict.items():
            p = get_rotated_subblocks(dset_p, this_pos_batch, probe_size_half, self.whole_object_size[:-1])
            self.params_chunk_array_dict[param_name] = p
            self.params_chunk_array_0_dict[param_name] = np.copy(p)
        return

    def write_params_to_file(self, this_pos_batch=None, probe_size_half=None, n_ranks=1):

        for param_name, p in self.params_chunk_array_dict.items():
            p = p - self.params_chunk_array_0_dict[param_name]
            p /= n_ranks
            dset_p = self.params_dset_dict[param_name]
            write_subblocks_to_file(dset_p, this_pos_batch, np.take(p, 0, axis=-1), np.take(p, 1, axis=-1),
                                    probe_size_half, self.whole_object_size[:-1], monochannel=False)
        return

    def rotate_files(self, coords, interpolation='bilinear'):

        for param_name, dset_p in self.params_dset_dict.items():
            apply_rotation_to_hdf5(dset_p, coords, rank, n_ranks, interpolation=interpolation, monochannel=False)

    def rotate_arrays(self, coords, interpolation='bilinear'):

        for param_name, arr in self.params_whole_array_dict.items():
            self.params_whole_array_dict[param_name] = apply_rotation(arr, coords, None, interpolation=interpolation)
        return


class AdamOptimizer(Optimizer):

    def __init__(self, whole_object_size, n_channel=2, output_folder='.'):
        super(AdamOptimizer, self).__init__(whole_object_size, output_folder=output_folder, params_list=['m', 'v'])
        return

    def apply_gradient(self, x, g, i_batch, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, verbose=True, shared_file_object=False, m=None, v=None):

        if m is None or v is None:
            if shared_file_object:
                m = self.params_chunk_array_dict['m']
                v = self.params_chunk_array_dict['v']
            else:
                m = self.params_whole_array_dict['m']
                v = self.params_whole_array_dict['v']
        m = (1 - b1) * g + b1 * m  # First moment estimate.
        v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
        vhat = v / (1 - b2 ** (i_batch + 1))
        d = step_size * mhat / (np.sqrt(vhat) + eps)
        x = x - d
        if verbose:
            try:
                print_flush('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))), 0,
                            comm.Get_rank())
            except:
                print('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))))
        if shared_file_object:
            self.params_chunk_array_dict['m'] = m
            self.params_chunk_array_dict['v'] = v
        else:
            self.params_whole_array_dict['m'] = m
            self.params_whole_array_dict['v'] = v
        self.i_batch += 1
        return x

    def apply_gradient_to_file(self, obj, gradient, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, verbose=True):

        assert isinstance(obj, ObjectFunction)
        assert isinstance(gradient, Gradient)
        s = obj.dset.shape
        slice_ls = range(rank, s[0], n_ranks)

        for i_slice in slice_ls:
            x = obj.dset[i_slice]
            g = gradient.dset[i_slice]
            m = self.params_dset_dict['m'][i_slice]
            v = self.params_dset_dict['v'][i_slice]
            x = self.apply_gradient(x, g, self.i_batch, step_size=step_size,
                                    b1=b1, b2=b2, eps=eps, verbose=verbose, shared_file_object=False,
                                    m=m, v=v)
            obj.dset[i_slice] = x
        self.i_batch += 1

class GDOptimizer(Optimizer):

    def __init__(self, whole_object_size, output_folder='.'):
        super(GDOptimizer, self).__init__(whole_object_size, output_folder=output_folder, params_list=[])
        return

    def apply_gradient(self, x, g, i_batch, step_size=0.001, dynamic_rate=True, first_downrate_iteration=92):
        g = np.array(g)
        if dynamic_rate:
            threshold_iteration = first_downrate_iteration
            i = 1
            while threshold_iteration < i_batch:
                threshold_iteration += first_downrate_iteration * 2 ** i
                i += 1
                step_size /= 2.
                print_flush('  -- Step size halved.', 0, comm.Get_rank())
        x = x - step_size * g

        return x

    def apply_gradient_to_file(self, obj, gradient, step_size=0.001, dynamic_rate=True, first_downrate_iteration=92):

        assert isinstance(obj, ObjectFunction)
        assert isinstance(gradient, Gradient)
        s = obj.dset.shape
        slice_ls = range(rank, s[0], n_ranks)

        for i_slice in slice_ls:
            x = obj.dset[i_slice]
            g = gradient.dset[i_slice]
            x = self.apply_gradient(x, g, self.i_batch, step_size=step_size,
                                    dynamic_rate=dynamic_rate, first_downrate_iteration=first_downrate_iteration)
            obj.dset[i_slice] = x
        self.i_batch += 1


def apply_gradient_adam(x, g, i_batch, m=None, v=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, verbose=True):

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
    if verbose:
        try:
            print_flush('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))), 0, comm.Get_rank())
        except:
            print('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))))
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
            print_flush('  -- Step size halved.', 0, comm.Get_rank())
    x = x - step_size * g

    return x