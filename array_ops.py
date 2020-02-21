import autograd.numpy as np
import os
import h5py
from mpi4py import MPI

from util import *

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

class LargeArray(object):

    def __init__(self, full_size, shared_file_object=False, monochannel=False, output_folder=None):

        self.full_size = full_size
        self.shared_file_object=shared_file_object
        self.monochannel = monochannel
        self.output_folder = output_folder
        self.f = None
        self.dset = None
        self.arr_0 = None

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

    def read_chunks_from_file(self, this_pos_batch, probe_size_half):

        obj = get_rotated_subblocks(self.dset, this_pos_batch, probe_size_half,
                                    self.full_size, monochannel=self.monochannel)
        self.arr_0 = np.copy(obj)
        return obj

    def rotate_data_in_file(self, coords, interpolation='bilinear'):

        apply_rotation_to_hdf5(self.dset, coords, rank, n_ranks, interpolation=interpolation)

    def write_chunks_to_file(self, this_pos_batch, arr_channel_0, arr_channel_1, probe_size_half, write_difference=True):

        if write_difference:
            if self.monochannel:
                arr_channel_0 = arr_channel_0 - self.arr_0
                arr_channel_0 /= n_ranks
            else:
                arr_channel_0 = arr_channel_0 - np.take(self.arr_0, 0, axis=-1)
                arr_channel_1 = arr_channel_1 - np.take(self.arr_0, 1, axis=-1)
                arr_channel_0 /= n_ranks
                arr_channel_1 /= n_ranks
        write_subblocks_to_file(self.dset, this_pos_batch, arr_channel_0, arr_channel_1,
                                probe_size_half, self.full_size, monochannel=self.monochannel)


class ObjectFunction(LargeArray):

    def __init__(self, full_size, shared_file_object=False, output_folder=None, ds_level=1,
                 object_type='normal'):
        super(ObjectFunction, self).__init__(full_size, shared_file_object=shared_file_object,
                                             monochannel=False, output_folder=output_folder)
        self.delta = None
        self.beta = None
        self.ds_level = ds_level
        self.object_type = object_type

    def create_file_object(self, use_checkpoint=False):

        super(ObjectFunction, self).create_file_object('intermediate_obj.h5', use_checkpoint=use_checkpoint)

    def initialize_array(self, save_stdout=None, timestr=None, not_first_level=False, initial_guess=None):
        self.delta, self.beta = \
            initialize_object(self.full_size[:-1], dset=None, ds_level=self.ds_level, object_type=self.object_type,
                              initial_guess=initial_guess, output_folder=self.output_folder, rank=rank,
                              n_ranks=n_ranks, save_stdout=save_stdout, timestr=timestr,
                              shared_file_object=False, not_first_level=not_first_level)

    def initialize_array_with_values(self, obj_delta, obj_beta):

        self.delta, self.beta = obj_delta, obj_beta

    def initialize_file_object(self, save_stdout=None, timestr=None, not_first_level=False, initial_guess=None):

        initialize_object(self.full_size[:-1], dset=self.dset, ds_level=self.ds_level, object_type=self.object_type,
                          initial_guess=initial_guess, output_folder=self.output_folder, rank=rank,
                          n_ranks=n_ranks, save_stdout=save_stdout, timestr=timestr,
                          shared_file_object=True, not_first_level=not_first_level)
