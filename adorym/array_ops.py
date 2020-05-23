import numpy as np
import os
import h5py
import gc
from scipy.ndimage import rotate as sp_rotate
try:
    from mpi4py import MPI
except:
    from adorym.pseudo import MPI

from adorym.util import *
import adorym.wrappers as w
import adorym.conventional as c

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

class LargeArray(object):

    def __init__(self, full_size, distribution_mode=None, monochannel=False, output_folder=None, device=None):
        self.full_size = full_size
        self.distribution_mode=distribution_mode
        self.monochannel = monochannel
        self.output_folder = output_folder
        self.f = None
        self.dset = None
        self.arr = None
        self.arr_0 = None
        self.arr_rot = None
        self.device = device
        self.slice_catalog = None
        if distribution_mode == 'distributed_object':
            self.slice_catalog = get_multiprocess_distribution_index(full_size[0], n_ranks)

    def create_file_object(self, fname, use_checkpoint=False):
        fmode = 'a' if use_checkpoint else 'w'
        try:
            self.f = h5py.File(os.path.join(self.output_folder, fname), fmode, driver='mpio', comm=comm)
        except:
            self.f = h5py.File(os.path.join(self.output_folder, fname), fmode)
        try:
            # If dataset doesn't exist, create it.
            self.dset = self.f.create_dataset('obj', shape=self.full_size, dtype='float64')
        except:
            # If dataset exists, create a pointer to it.
            self.dset = self.f['obj']

    def read_chunks_from_file(self, this_pos_batch, probe_size, dset_2=None, device=None, unknown_type='delta_beta'):
        dset = self.dset if dset_2 is None else dset_2
        obj = get_rotated_subblocks(dset, this_pos_batch, probe_size,
                                    self.full_size, monochannel=self.monochannel, unknown_type=unknown_type)
        self.arr_0 = np.copy(obj)
        obj = w.create_variable(obj, device=device)
        return obj

    def read_chunks_from_distributed_object(self, probe_pos, this_ind_batch_allranks, minibatch_size,
                                            probe_size, device=None, unknown_type='delta_beta', apply_to_arr_rot=False, dtype='float32'):
        a = self.arr if not apply_to_arr_rot else self.arr_rot
        obj = get_subblocks_from_distributed_object_mpi(a, self.slice_catalog, probe_pos, this_ind_batch_allranks, minibatch_size,
                                                    probe_size, self.full_size, unknown_type, output_folder=self.output_folder, dtype=dtype)
        obj = w.create_variable(obj, device=device)
        return obj

    def rotate_data_in_file(self, coords, interpolation='bilinear', dset_2=None, precalculate_rotation_coords=True):
        apply_rotation_to_hdf5(self.dset, coords, rank, n_ranks, interpolation=interpolation,
                               monochannel=self.monochannel, dset_2=dset_2, precalculate_rotation_coords=precalculate_rotation_coords)

    def reverse_rotate_data_in_file(self, coords, interpolation='bilinear', precalculate_rotation_coords=True):
        revert_rotation_to_hdf5(self.dset, coords, rank, n_ranks, interpolation=interpolation,
                                monochannel=self.monochannel, precalculate_rotation_coords=precalculate_rotation_coords)

    def rotate_array(self, coords, interpolation='bilinear', precalculate_rotation_coords=True, apply_to_arr_rot=False,
                     overwrite_arr=False, override_backend=None, dtype=None, override_device=None):
        if self.arr is None:
            return
        a = self.arr if not apply_to_arr_rot else self.arr_rot
        if override_device is not None:
            if override_device == 'cpu':
                d = None
            else:
                d = override_device
        else:
            d = self.device
        if precalculate_rotation_coords:
            if overwrite_arr:
                self.arr = apply_rotation(a, coords, interpolation=interpolation, device=d, override_backend=override_backend)
            else:
                self.arr_rot = apply_rotation(a, coords, interpolation=interpolation, device=d, override_backend=override_backend)
        else:
            if overwrite_arr:
                self.arr = sp_rotate(a, -coords, axes=(1, 2), reshape=False, order=1, mode='nearest')
            else:
                self.arr_rot = sp_rotate(a, -coords, axes=(1, 2), reshape=False, order=1, mode='nearest')
        if dtype is not None:
            if overwrite_arr:
                self.arr = w.cast(self.arr, dtype, override_backend=override_backend)
            else:
                self.arr_rot = w.cast(self.arr_rot, dtype, override_backend=override_backend)

    def write_chunks_to_file(self, this_pos_batch, arr_channel_0, arr_channel_1, probe_size, write_difference=True, dset_2=None, dtype='float32'):
        dset = self.dset if dset_2 is None else dset_2
        arr_channel_0 = w.to_numpy(arr_channel_0)
        if arr_channel_1 is not None: arr_channel_1 = w.to_numpy(arr_channel_1)
        if write_difference:
            if self.monochannel:
                arr_channel_0 = arr_channel_0 - self.arr_0
                arr_channel_0 /= n_ranks
            else:
                arr_channel_0 = arr_channel_0 - np.take(self.arr_0, 0, axis=-1)
                arr_channel_1 = arr_channel_1 - np.take(self.arr_0, 1, axis=-1)
                arr_channel_0 /= n_ranks
                arr_channel_1 /= n_ranks
        write_subblocks_to_file(dset, this_pos_batch, arr_channel_0, arr_channel_1,
                                probe_size, self.full_size, monochannel=self.monochannel, dtype='float32')

    def sync_chunks_to_distributed_object(self, obj, probe_pos, this_ind_batch_allranks, minibatch_size,
                                          probe_size, dtype='float32'):
        obj = w.to_numpy(obj)
        self.arr = sync_subblocks_among_distributed_object_mpi(obj, self.arr, self.slice_catalog, probe_pos, this_ind_batch_allranks,
                                                       minibatch_size, probe_size, self.full_size,
                                                       output_folder=self.output_folder, dtype='float32')


class ObjectFunction(LargeArray):

    def __init__(self, full_size, distribution_mode=None, output_folder=None, ds_level=1,
                 object_type='normal', device=None):
        super(ObjectFunction, self).__init__(full_size, distribution_mode=distribution_mode,
                                             monochannel=False, output_folder=output_folder, device=device)
        self.chunks = None
        self.ds_level = ds_level
        self.object_type = object_type
        self.f_rot = None
        self.dset_rot = None

    def create_file_object(self, use_checkpoint=False):
        super(ObjectFunction, self).create_file_object('intermediate_obj.h5', use_checkpoint=use_checkpoint)

    def create_temporary_file_object(self):
        """
        This file is used to save rotated object.
        """
        try:
            self.f_rot = h5py.File(os.path.join(self.output_folder, 'intermediate_obj_rot.h5'), 'w', driver='mpio', comm=comm)
        except:
            self.f_rot = h5py.File(os.path.join(self.output_folder, 'intermediate_obj_rot.h5'), 'w')
        self.dset_rot = self.f_rot.create_dataset('obj', shape=self.full_size, dtype='float64')

    def initialize_array(self, save_stdout=None, timestr=None, not_first_level=False, initial_guess=None, device=None,
                         random_guess_means_sigmas=(8.7e-7, 5.1e-8, 1e-7, 1e-8), unknown_type='delta_beta', non_negativity=False):
        temp_delta, temp_beta = \
            initialize_object_for_dp(self.full_size[:-1], dset=None, ds_level=self.ds_level, object_type=self.object_type,
                              initial_guess=initial_guess, output_folder=self.output_folder,
                              save_stdout=save_stdout, timestr=timestr,
                              not_first_level=not_first_level,
                              random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type, non_negativity=non_negativity)
        self.arr = w.create_variable(np.stack([temp_delta, temp_beta], -1), device=device, requires_grad=True)
        del temp_delta
        del temp_beta
        gc.collect()

    def initialize_array_with_values(self, obj_delta, obj_beta, device=None):
        self.arr = w.create_variable(np.stack([obj_delta, obj_beta], -1), device=device, requires_grad=True)

    def initialize_distributed_array(self, save_stdout=None, timestr=None, not_first_level=False, initial_guess=None,
                         random_guess_means_sigmas=(8.7e-7, 5.1e-8, 1e-7, 1e-8), unknown_type='delta_beta', dtype='float32', non_negativity=False):
        if self.slice_catalog[rank] is not None:
            delta, beta = \
                initialize_object_for_do(self.full_size[:-1], slice_catalog=self.slice_catalog, ds_level=self.ds_level, object_type=self.object_type,
                                  initial_guess=initial_guess, output_folder=self.output_folder,
                                  save_stdout=save_stdout, timestr=timestr,
                                  not_first_level=not_first_level,
                                  random_guess_means_sigmas=random_guess_means_sigmas, unknown_type=unknown_type, dtype=dtype,
                                  non_negativity=non_negativity)
            self.arr = np.stack([delta, beta], -1)

    def initialize_distributed_array_with_values(self, obj_delta, obj_beta, dtype='float32'):
        if self.slice_catalog[rank] is not None:
            delta = obj_delta[slice(*self.slice_catalog[rank])]
            beta = obj_beta[slice(*self.slice_catalog[rank])]
            self.arr = np.stack([delta, beta], -1).astype(dtype)

    def initialize_distributed_array_with_zeros(self, dtype='float32'):
        if self.slice_catalog[rank] is not None:
            delta = np.zeros([self.slice_catalog[rank][1] - self.slice_catalog[rank][0], *self.full_size[1:-1]], dtype=dtype)
            beta = np.zeros([self.slice_catalog[rank][1] - self.slice_catalog[rank][0], *self.full_size[1:-1]], dtype=dtype)
            self.arr = np.stack([delta, beta], -1)

    def initialize_file_object(self, save_stdout=None, timestr=None, not_first_level=False, initial_guess=None,
                               random_guess_means_sigmas=(8.7e-7, 5.1e-8, 1e-7, 1e-8), unknown_type='delta_beta',
                               dtype='float32', non_negativity=False):
        initialize_object_for_sf(self.full_size[:-1], dset=self.dset, ds_level=self.ds_level, object_type=self.object_type,
                          initial_guess=initial_guess, output_folder=self.output_folder,
                          save_stdout=save_stdout, timestr=timestr,
                          not_first_level=not_first_level, random_guess_means_sigmas=random_guess_means_sigmas,
                          unknown_type=unknown_type, dtype=dtype, non_negativity=non_negativity)

    def apply_finite_support_mask_to_array(self, mask, unknown_type='delta_beta', device=None):
        assert isinstance(mask, Mask)
        with w.no_grad():
            if unknown_type == 'delta_beta':
                delta = self.arr[:, :, :, 0] * mask.mask
                beta = self.arr[:, :, :, 1] * mask.mask
            elif unknown_type == 'real_imag':
                ones_arr = w.ones(self.arr.shape[:-1], requires_grad=False, device=device)
                zeros_arr = w.zeros(self.arr.shape[:-1], requires_grad=False, device=device)
                delta = self.arr[:, :, :, 0] * mask.mask + ones_arr * (1 - mask.mask)
                beta = self.arr[:, :, :, 1] * mask.mask + zeros_arr * (1 - mask.mask)
            self.arr = w.stack([delta, beta], -1)
        w.reattach(self.arr)

    def apply_finite_support_mask_to_file(self, mask, unknown_type='delta_beta', device=None):
        assert isinstance(mask, Mask)
        slice_ls = range(rank, self.full_size[0], n_ranks)
        if unknown_type == 'real_imag':
            ones_arr = w.ones(mask.dset.shape[1:3], requires_grad=False, device=device)
            zeros_arr = w.zeros(mask.dset.shape[1:3], requires_grad=False, device=device)
        for i_slice in slice_ls:
            mask_arr = mask.dset[i_slice]
            obj_arr = self.dset[i_slice]
            if unknown_type == 'delta_beta':
                obj_arr[:, :, 0] *= mask_arr
                obj_arr[:, :, 1] *= mask_arr
            elif unknown_type == 'real_imag':
                obj_arr[:, :, 0] = obj_arr[:, :, 0] * mask_arr + ones_arr * (1 - mask_arr)
                obj_arr[:, :, 1] = obj_arr[:, :, 1] * mask_arr + zeros_arr * (1 - mask_arr)
            self.dset[i_slice] = obj_arr

    def update_object(self, obj):
        self.arr.detach()
        self.arr = obj

    def update_using_external_algorithm(self, algorithm, kwargs, device=None):
        if algorithm == 'ctf':
            this_prj_batch = kwargs['prj'][0]
            energy_ev = kwargs['energy_ev']
            psize_cm = kwargs['psize_cm']
            free_prop_cm = kwargs['free_prop_cm']
            ctf_lg_kappa = kwargs['ctf_lg_kappa']
            prj_affine_ls = kwargs['prj_affine_ls']
            # Safe zone width is default to 0 on purpose in order to produce self-vignetted result, so that
            # the forward propagation in the next iteration won't blow up.
            phase = c.multidistance_ctf_wrapped(this_prj_batch, free_prop_cm, energy_ev, psize_cm, 10 ** ctf_lg_kappa[0],
                                                safe_zone_width=0, prj_affine_ls=prj_affine_ls, device=device)
            self.arr[:, :, :, 0] = w.reshape(phase, [*phase.shape, 1])


class Gradient(ObjectFunction):

    def __init__(self, obj):
        assert isinstance(obj, ObjectFunction)
        super(Gradient, self).__init__(obj.full_size, obj.distribution_mode,
                                 obj.output_folder, obj.dset, obj.object_type)

    def create_file_object(self):
        super(ObjectFunction, self).create_file_object('intermediate_grad.h5', use_checkpoint=False)

    def initialize_gradient_file(self, dtype='float32'):
        initialize_hdf5_with_constant(self.dset, rank, n_ranks, dtype=dtype)


class Mask(LargeArray):

    def __init__(self, full_size, finite_support_mask_path, distribution_mode=None, output_folder=None, ds_level=1):
        super(Mask, self).__init__(full_size, distribution_mode,
                                   monochannel=True, output_folder=output_folder)
        self.mask = None
        self.ds_level = ds_level
        self.finite_support_mask_path = finite_support_mask_path

    def create_file_object(self, use_checkpoint=False):
        super(Mask, self).create_file_object('intermediate_mask.h5', use_checkpoint=use_checkpoint)

    def initialize_array_with_values(self, mask, device=None, dtype=None):
        args = {}
        if dtype is not None:
            args['dtype'] = dtype
        self.mask = w.create_variable(mask, requires_grad=False, device=device, **args)

    def initialize_distributed_array(self, mask, dtype='float32'):
        if self.slice_catalog[rank] is not None:
            self.mask = mask[slice(*self.slice_catalog[rank])].astype(dtype)

    def initialize_file_object(self, dtype='float32'):
        # arr is a memmap.
        arr = dxchange.read_tiff(self.finite_support_mask_path)
        initialize_hdf5_with_arrays(self.dset, rank, n_ranks, arr, None, dtype=dtype)

    def update_mask_array(self, obj, threshold=1e-9):
        assert isinstance(obj, ObjectFunction)
        if obj.arr is not None:
            obj_arr, _ = w.split_channel(obj.arr)
            self.mask[obj_arr < threshold] = 0

    def update_mask_file(self, obj, threshold=1e-9):
        assert isinstance(obj, ObjectFunction)
        if self.shared_file_object:
            slice_ls = range(rank, self.full_size[0], n_ranks)
            for i_slice in slice_ls:
                obj_arr = obj.dset[i_slice, :, :, 0]
                mask_arr = self.dset[i_slice, :, :]
                mask_arr[obj_arr < threshold] = 0
                self.dset[i_slice, :, :] = mask_arr
