import gc
import inspect
import time

import numpy as np
from scipy.ndimage import rotate as sp_rotate

import adorym.wrappers as w
from adorym.propagate import get_kernel, multislice_propagate_batch, multislice_propagate_batch_complex
from adorym.regularizers import *
from adorym.util import *


class ForwardModel(object):
    """
    The parent forward model class.

    :param loss_function_type: String. Can be ``'lsq'`` or ``'poisson'``. Type of loss function.
    :param distribution_mode: ``None``, ``'distributed_object'``, or ``'shared_file'``. Mode of parallelization.
    :param device: ``None`` or device object. Use ``None`` for computations on CPU only.
    :param common_vars_dict: Dict. A dictionary of variables that are static (i.e., won't be optimized and won't
        change with different minibatches). When the ForwardModel class is declared in ``reconstruct_ptycho``, it
        can simply be the local namespace of the main module, i.e., ``locals()``.
    :param raw_data_type: String. Can be ``'magnitude'`` or ``'intensity'``. Type of raw data in HDF5. For line-projection
        tomography reconstruction where raw data have already been minus-logged, always use ``'magnitude'``.
    """
    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        self.loss_function_type = loss_function_type
        self.argument_ls = []
        self.regularizer_dict = {}
        self.distribution_mode = distribution_mode
        self.device = device
        self.simulation_mode = simulation_mode
        self.current_loss = 0
        self.raw_data_type = raw_data_type
        self.i_call = 0
        self.common_vars = common_vars_dict
        if common_vars_dict is not None:
            self.unknown_type = common_vars_dict['unknown_type']
            self.normalize_fft = common_vars_dict['normalize_fft']
            self.sign_convention = common_vars_dict['sign_convention']
            self.rotate_out_of_loop = common_vars_dict['rotate_out_of_loop']
            self.scale_ri_by_k = common_vars_dict['scale_ri_by_k']
            self.is_minus_logged = common_vars_dict['is_minus_logged']
            self.forward_algorithm = common_vars_dict['forward_algorithm']
            self.stdout_options = common_vars_dict['stdout_options']
            self.poisson_multiplier = common_vars_dict['poisson_multiplier']
            self.common_probe_pos = common_vars_dict['common_probe_pos']
            self.binning = common_vars_dict['binning']
            self.prj = common_vars_dict['prj'] # HDF5 dataset pointer
        self.loss_args = {}
        self.reg_list = []

    def update_loss_args(self, kwargs):
        self.loss_args = kwargs

    def add_regularizers(self, reg_list):
        self.reg_list = reg_list

    def get_regularization_value(self, obj, device=None):
        reg = w.create_variable(0., device=self.device)
        for r in self.reg_list:
            reg = reg + r.get_value(obj, distribution_mode=self.distribution_mode, device=device)
        print_flush('  Reg term = {}.'.format(w.to_numpy(reg)), 0, rank, **self.stdout_options)
        return reg

    def get_argument_index(self, arg):
        for i, a in enumerate(self.argument_ls):
            if a == arg:
                return i
        raise ValueError('{} is not in the argument list.'.format(arg))

    def get_mismatch_loss(self, this_pred_batch, this_prj_batch):
        if self.loss_function_type == 'lsq':
            if self.raw_data_type == 'magnitude':
                loss = w.mean((this_pred_batch - w.abs(this_prj_batch)) ** 2)
            elif self.raw_data_type == 'intensity':
                loss = w.mean((this_pred_batch - w.sqrt(w.abs(this_prj_batch))) ** 2)
        elif self.loss_function_type == 'poisson':
            if self.raw_data_type == 'magnitude':
                loss = w.mean(this_pred_batch ** 2 * self.poisson_multiplier -
                              w.abs(this_prj_batch) ** 2 * self.poisson_multiplier * w.log(
                    this_pred_batch ** 2 * self.poisson_multiplier))
            elif self.raw_data_type == 'intensity':
                loss = w.mean(this_pred_batch ** 2 * self.poisson_multiplier -
                              w.abs(this_prj_batch) * self.poisson_multiplier * w.log(
                    this_pred_batch ** 2 * self.poisson_multiplier))
        return loss

    def predict(self, *args, **kwargs):
        """
        Override and expand this method for each specific class.
        Generally the method should return the detected magnitude. If this is not the case, you should also
        override method `loss` (the last-layer loss) or `get_loss_function` (the full loss function constructor).
        """
        pass

    def get_data(self, this_i_theta, this_ind_batch, theta_downsample=None, ds_level=1):
        if theta_downsample is None: theta_downsample = 1
        this_prj_batch = self.prj[this_i_theta * theta_downsample, this_ind_batch]
        this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
        if ds_level > 1:
            this_prj_batch = this_prj_batch[:, ::ds_level, ::ds_level]
        return this_prj_batch

    def loss(self, this_pred_batch, this_prj_batch, obj):
        """
        The last-layer loss function that computes (regularized) loss from predicted magnitude.
        This function is combined with `predict` to give the full loss function, as done in `get_loss_function`.
        Argument this_pred_batch is assumed to be detected **magnitude** (square root of intensity).
        If this is not the case for your specific ForwardModel class, override this method.
        """
        beamstop = self.common_vars['beamstop']
        if beamstop is not None:
            beamstop[beamstop >= 1e-5] = 1
            beamstop[beamstop < 1e-5] = 0
            beamstop = w.cast(beamstop, 'bool')
            beamstop_mask_stack = w.tile(beamstop, [len(this_pred_batch), 1, 1])
            this_pred_batch = w.reshape(this_pred_batch[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
            this_prj_batch = w.reshape(this_prj_batch[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
            print_flush('  {} valid pixels remain after applying beamstop mask.'.format(this_pred_batch.shape[1]), 0, rank)
        loss = self.get_mismatch_loss(this_pred_batch, this_prj_batch)
        if len(self.reg_list) > 0:
            loss = loss + self.get_regularization_value(obj, device=self.device)
        loss_val = w.to_numpy(loss)
        try:
            loss_val = float(loss_val)
        except:
            # Variable loss might be a nested ArrayBox when using second order optimizers with Autograd
            loss_val = float(loss_val._value)
        self.current_loss = float(loss_val)
        return loss

    def get_loss_function(self):
        """
        Overwrite this function for each sub-class, but make sure it contains the explicitly listed arguments.
        Do not keep differentiable arguments as **kwargs since it will not be recognized by Autograd.
        """
        def calculate_loss(obj, this_i_theta, this_ind_batch, *args, **kwargs):
            theta_downsample = self.common_vars['theta_downsample']
            ds_level = self.common_vars['ds_level']
            this_pred_batch = self.predict(*args, **kwargs)
            this_prj_batch = self.get_data(this_i_theta, this_ind_batch, theta_downsample=theta_downsample, ds_level=ds_level)
            loss = self.loss(this_pred_batch, this_prj_batch, obj)
            return loss
        return calculate_loss


class PtychographyModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        super().__init__(loss_function_type, distribution_mode, device, common_vars_dict,
                                                raw_data_type, simulation_mode=simulation_mode)
        # ==========================================================================================
        # argument_ls must be in the same order as arguments in get_loss_function's function call!
        # ==========================================================================================
        args = inspect.getfullargspec(self.predict).args
        args.pop(0)
        self.argument_ls = args

    def predict(self, obj, probe, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, tilt_ls, prj_pos_offset):
        """
        Calculated predicted measurment. The signature of this function exactly matches the function returned
        by the ``get_loss_function`` method.

        :param obj: Array with shape [obj_size_y, obj_size_x, obj_size_z, 2]. Object function.
        :param probe: Array with shape [n_probe_modes, len_probe_y, len_probe_x].
        :param probe_defocus_mm: Probe defocus in mm.
        :param probe_pos_offset: Array with shape [n_theta, 2]. Probe position offset for each angle, in pixels.
        :param this_i_theta: Int. Current angle index.
        :param this_pos_batch: Array with shape [minibatch_size, 2]. Current batch of probe positions.
        :param prj: HDF5 measurement dataset handler.
        :param probe_pos_correction: Array with shape [n_probes, 2]. Additive correction values of probe positions.
        :param this_ind_batch: List of Int. Current batch of tile indices.
        :param tilt_ls: Array with shape [3, n_theta]. 3-axis tilt to be applied before propagation.
        :param prj_pos_offset: Array with shape [n_theta, 2]. Projection position offset for each angle, in pixels. Note
            that this is different from probe_pos_offset - shifting is applied after passing through the object.
        :return: Array with shape [minibatch_size, len_probe_y, len_probe_x]. Magnitude of detected wavefields.
        """
        device_obj = self.common_vars['device_obj']
        lmbda_nm = self.common_vars['lmbda_nm']
        voxel_nm = self.common_vars['voxel_nm']
        probe_size = self.common_vars['probe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        minibatch_size = self.common_vars['minibatch_size']
        ds_level = self.common_vars['ds_level']
        this_obj_size = self.common_vars['this_obj_size']
        energy_ev = self.common_vars['energy_ev']
        psize_cm = self.common_vars['psize_cm']
        h = self.common_vars['h']
        pure_projection = self.common_vars['pure_projection']
        n_dp_batch = self.common_vars['n_dp_batch']
        free_prop_cm = self.common_vars['free_prop_cm']
        optimize_probe_defocusing = self.common_vars['optimize_probe_defocusing']
        optimize_probe_pos_offset = self.common_vars['optimize_probe_pos_offset']
        optimize_prj_pos_offset = self.common_vars['optimize_prj_pos_offset']
        optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
        optimize_tilt = self.common_vars['optimize_tilt']
        debug = self.common_vars['debug']
        output_folder = self.common_vars['output_folder']
        unknown_type = self.common_vars['unknown_type']
        n_probe_modes = self.common_vars['n_probe_modes']
        n_theta = self.common_vars['n_theta']
        precalculate_rotation_coords = self.common_vars['precalculate_rotation_coords']
        theta_ls = self.common_vars['theta_ls']

        if precalculate_rotation_coords:
            coord_ls = read_origin_coords(f'arrsize_{this_obj_size[0]}_{this_obj_size[1]}_{this_obj_size[2]}_ntheta_{n_theta}',
                                          theta_ls[this_i_theta], reverse=False)

        flag_pp_sqrt = True
        if self.raw_data_type == 'magnitude':
            flag_pp_sqrt = False

        # Allocate subbatches.
        probe_pos_batch_ls = []
        i_dp = 0
        while i_dp < minibatch_size:
            probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
            i_dp += n_dp_batch

        this_pos_batch = np.round(this_pos_batch).astype(int)
        if optimize_probe_defocusing:
            h_probe = get_kernel(probe_defocus_mm * 1e6, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)
            probe = w.convolve_with_transfer_function_complex(probe, h_probe)

        if optimize_prj_pos_offset:
            this_prj_offset = prj_pos_offset[this_i_theta]
        else:
            this_prj_offset = None

        if optimize_probe_pos_offset:
            this_offset = probe_pos_offset[this_i_theta]
            probe = realign_image_fourier_complex(probe, this_offset, axes=(1, 2), device=device_obj)

        if not two_d_mode and not self.distribution_mode:
            if not optimize_tilt and self.common_vars['initial_tilt'] is None:
                if not self.rotate_out_of_loop:
                    if precalculate_rotation_coords:
                        obj_rot = apply_rotation(obj, coord_ls, device=device_obj)
                    else:
                        obj_rot = rotate_no_grad(obj, theta_ls[this_i_theta], axis=0)
                else:
                    obj_rot = obj
            else:
                obj_rot = w.rotate(obj, tilt_ls[0, this_i_theta], axis=0, device=device_obj)
                obj_rot = w.rotate(obj_rot, tilt_ls[1, this_i_theta], axis=1, device=device_obj)
                obj_rot = w.rotate(obj_rot, tilt_ls[2, this_i_theta], axis=2, device=device_obj)

        else:
            if optimize_tilt:
                raise NotImplementedError('Tilt optimization is not impemented for distributed mode or when two_d_mode is on.')
            obj_rot = obj

        ex_mag_ls = []

        # Pad if needed
        if self.distribution_mode is None:
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch, probe_size, unknown_type=unknown_type)

        pos_ind = 0
        for k, pos_batch in enumerate(probe_pos_batch_ls):
            subobj_ls = []
            probe_ls = []

            # Get shifted probe list.
            for j in range(len(pos_batch)):
                if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                    this_shift = probe_pos_correction[this_i_theta, this_ind_batch[k * n_dp_batch + j]]
                    probe_shifted = realign_image_fourier_complex(probe, this_shift, axes=(1, 2), device=device_obj)
                    probe_ls.append(probe_shifted)
            if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                # Shape of probe_xxx_ls.shape is [n_dp_batch, n_probe_modes, y, x].
                probe_ls = w.stack(probe_ls)
            else:
                # Shape of probe_xxx_ls.shape is [n_probe_modes, y, x].
                probe_ls = probe

            # Get object list.
            if self.distribution_mode is None:
                if len(pos_batch) == 1:
                    pos = pos_batch[0]
                    pos_y = pos[0] + pad_arr[0, 0]
                    pos_x = pos[1] + pad_arr[1, 0]
                    if pos_y == 0 and pos_x == 0 and probe_size[0] == this_obj_size[0] and probe_size[1] == this_obj_size[1]:
                        subobj = obj_rot
                    else:
                        subobj = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :]
                    subobj_ls = w.reshape(subobj, [1, *subobj.shape])
                else:
                    for pos in pos_batch:
                        pos_y = pos[0] + pad_arr[0, 0]
                        pos_x = pos[1] + pad_arr[1, 0]
                        subobj = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :]
                        subobj_ls.append(subobj)
                    subobj_ls = w.stack(subobj_ls)
            else:
                subobj_ls = obj_rot[pos_ind:pos_ind + len(pos_batch), :, :, :]
                pos_ind += len(pos_batch)

            gc.collect()
            if n_probe_modes == 1:
                if len(probe_ls.shape) == 3:
                    this_probe_ls = probe_ls[0, :, :]
                else:
                    this_probe_ls = probe_ls[:, 0, :, :]
                ex = multislice_propagate_batch_complex(
                                subobj_ls,
                                this_probe_ls,
                                energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm, binning=self.binning,
                                obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
                                unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
                                scale_ri_by_k=self.scale_ri_by_k, is_minus_logged=self.is_minus_logged,
                                pure_projection_return_sqrt=flag_pp_sqrt, shift_exit_wave=this_prj_offset)
                ex_mag_ls.append(w.norm(ex))
            else:
                for i_mode in range(n_probe_modes):
                    if len(probe_ls.shape) == 3:
                        this_probe_ls = probe_ls[i_mode, :, :]
                    else:
                        this_probe_ls = probe_ls[:, i_mode, :, :]
                    temp = multislice_propagate_batch_complex(
                                subobj_ls, this_probe_ls,
                                energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm, binning=self.binning,
                                obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
                                unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
                                scale_ri_by_k=self.scale_ri_by_k, is_minus_logged=self.is_minus_logged,
                                pure_projection_return_sqrt=flag_pp_sqrt, shift_exit_wave=this_prj_offset)
                    if i_mode == 0:
                        ex_int = w.abs(temp)**2
                    else:
                        ex_int = ex_int + w.abs(temp)**2
                ex_mag_ls.append(w.sqrt(ex_int))
        del subobj_ls, probe_ls

        # Output shape is [minibatch_size, y, x].
        if len(ex_mag_ls) > 1:
            ex_mag_ls = w.concatenate(ex_mag_ls, 0)
        else:
            ex_mag_ls = ex_mag_ls[0]
        if rank == 0 and debug and self.i_call % 10 == 0:
            ex_mag_val = w.to_numpy(ex_mag_ls)
            dxchange.write_tiff(ex_mag_val, os.path.join(output_folder, 'intermediate', 'detected_mag'), dtype='float32', overwrite=True)
        self.i_call += 1
        return ex_mag_ls

    def get_loss_function(self):
        def calculate_loss(obj, probe, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch, tilt_ls, prj_pos_offset):
            theta_downsample = self.common_vars['theta_downsample']
            ds_level = self.common_vars['ds_level']
            this_pred_batch = self.predict(obj, probe, probe_defocus_mm,
                                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                                           probe_pos_correction, this_ind_batch, tilt_ls, prj_pos_offset)
            this_prj_batch = self.get_data(this_i_theta, this_ind_batch, theta_downsample=theta_downsample, ds_level=ds_level)
            loss = self.loss(this_pred_batch, this_prj_batch, obj)
            return loss
        return calculate_loss


class SingleBatchFullfieldModel(PtychographyModel):
    # Created to avoid unnecessary stacking and concatenation.

    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        super(SingleBatchFullfieldModel, self).__init__(loss_function_type, distribution_mode, device, common_vars_dict,
                                                raw_data_type, simulation_mode=simulation_mode)

    def predict(self, obj, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, tilt_ls, prj_pos_offset):
        """
        Calculated predicted measurment. The signature of this function exactly matches the function returned
        by the ``get_loss_function`` method.

        :param obj: Array with shape [obj_size_y, obj_size_x, obj_size_z, 2]. Object function.
        :param probe_real: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Preal part of the probe.
        :param probe_imag: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Imaginary part of the probe.
        :param probe_defocus_mm: Probe defocus in mm.
        :param probe_pos_offset: Array with shape [n_theta, 2]. Probe position offset for each angle, in pixels.
        :param this_i_theta: Int. Current angle index.
        :param this_pos_batch: Array with shape [minibatch_size, 2]. Current batch of probe positions.
        :param prj: HDF5 measurement dataset handler.
        :param probe_pos_correction: Array with shape [n_probes, 2]. Additive correction values of probe positions.
        :param this_ind_batch: List of Int. Current batch of tile indices.
        :param tilt_ls: Array with shape [3, n_theta]. 3-axis tilt to be applied before propagation.
        :return: Array with shape [minibatch_size, len_probe_y, len_probe_x]. Magnitude of detected wavefields.
        """
        device_obj = self.common_vars['device_obj']
        probe_size = self.common_vars['probe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        ds_level = self.common_vars['ds_level']
        this_obj_size = self.common_vars['this_obj_size']
        energy_ev = self.common_vars['energy_ev']
        psize_cm = self.common_vars['psize_cm']
        h = self.common_vars['h']
        optimize_prj_pos_offset = self.common_vars['optimize_prj_pos_offset']
        pure_projection = self.common_vars['pure_projection']
        free_prop_cm = self.common_vars['free_prop_cm']
        unknown_type = self.common_vars['unknown_type']
        n_theta = self.common_vars['n_theta']
        precalculate_rotation_coords = self.common_vars['precalculate_rotation_coords']
        theta_ls = self.common_vars['theta_ls']

        if precalculate_rotation_coords:
            coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                          theta_ls[this_i_theta], reverse=False)

        flag_pp_sqrt = True
        if self.raw_data_type == 'magnitude':
            flag_pp_sqrt = False

        if not two_d_mode and not self.distribution_mode:
            if not self.rotate_out_of_loop:
                if precalculate_rotation_coords:
                    obj_rot = apply_rotation(obj, coord_ls, device=device_obj)
                else:
                    obj_rot = rotate_no_grad(obj, theta_ls[this_i_theta], axis=0, device=device_obj)
            else:
                obj_rot = obj
        else:
            obj_rot = obj

        if self.distribution_mode is None:
            obj_rot = w.reshape(obj_rot, [1, *obj_rot.shape])

        if optimize_prj_pos_offset:
            this_prj_offset = prj_pos_offset[this_i_theta]
        else:
            this_prj_offset = None

        ex_real, ex_imag = multislice_propagate_batch(
            obj_rot,
            probe_real, probe_imag,
            energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm,
            obj_batch_shape=[1, *probe_size, this_obj_size[-1]], binning=self.binning,
            fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
            unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
            scale_ri_by_k=self.scale_ri_by_k, is_minus_logged=self.is_minus_logged,
            pure_projection_return_sqrt=flag_pp_sqrt, shift_exit_wave=this_prj_offset)
        ex_mag_ls = w.norm(ex_real, ex_imag)

        if self.simulation_mode:
            return ex_real, ex_imag
        else:
            return ex_mag_ls


class SingleBatchPtychographyModel(PtychographyModel):
    # Created to avoid unnecessary stacking and concatenation.

    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        super(SingleBatchPtychographyModel, self).__init__(loss_function_type, distribution_mode, device, common_vars_dict,
                                                raw_data_type, simulation_mode=simulation_mode)

    def predict(self, obj, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, tilt_ls, prj_pos_offset):
        """
        Calculated predicted measurment. The signature of this function exactly matches the function returned
        by the ``get_loss_function`` method.

        :param obj: Array with shape [obj_size_y, obj_size_x, obj_size_z, 2]. Object function.
        :param probe_real: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Preal part of the probe.
        :param probe_imag: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Imaginary part of the probe.
        :param probe_defocus_mm: Probe defocus in mm.
        :param probe_pos_offset: Array with shape [n_theta, 2]. Probe position offset for each angle, in pixels.
        :param this_i_theta: Int. Current angle index.
        :param this_pos_batch: Array with shape [minibatch_size, 2]. Current batch of probe positions.
        :param prj: HDF5 measurement dataset handler.
        :param probe_pos_correction: Array with shape [n_probes, 2]. Additive correction values of probe positions.
        :param this_ind_batch: List of Int. Current batch of tile indices.
        :param tilt_ls: Array with shape [3, n_theta]. 3-axis tilt to be applied before propagation.
        :return: Array with shape [minibatch_size, len_probe_y, len_probe_x]. Magnitude of detected wavefields.
        """

        device_obj = self.common_vars['device_obj']
        probe_size = self.common_vars['probe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        ds_level = self.common_vars['ds_level']
        this_obj_size = self.common_vars['this_obj_size']
        energy_ev = self.common_vars['energy_ev']
        psize_cm = self.common_vars['psize_cm']
        h = self.common_vars['h']
        pure_projection = self.common_vars['pure_projection']
        free_prop_cm = self.common_vars['free_prop_cm']
        unknown_type = self.common_vars['unknown_type']
        n_theta = self.common_vars['n_theta']
        precalculate_rotation_coords = self.common_vars['precalculate_rotation_coords']
        theta_ls = self.common_vars['theta_ls']
        optimize_prj_pos_offset = self.common_vars['optimize_prj_pos_offset']

        if precalculate_rotation_coords:
            coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                          theta_ls[this_i_theta], reverse=False)

        flag_pp_sqrt = True
        if self.raw_data_type == 'magnitude':
            flag_pp_sqrt = False

        if not two_d_mode and not self.distribution_mode:
            if not self.rotate_out_of_loop:
                if precalculate_rotation_coords:
                    obj_rot = apply_rotation(obj, coord_ls, device=device_obj)
                else:
                    obj_rot = rotate_no_grad(obj, theta_ls[this_i_theta], axis=0, device=device_obj)
            else:
                obj_rot = obj
        else:
            obj_rot = obj

        if self.distribution_mode is None:
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch, probe_size, unknown_type=unknown_type)
        else:
            pad_arr = np.array([[0, 0], [0, 0]])

        if optimize_prj_pos_offset:
            this_prj_offset = prj_pos_offset[this_i_theta]
        else:
            this_prj_offset = None

        if self.distribution_mode is None:
            pos = this_pos_batch[0]
            pos_y = pos[0] + pad_arr[0, 0]
            pos_x = pos[1] + pad_arr[1, 0]
            obj_rot = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :, :]
            obj_rot = w.reshape(obj_rot, [1, *obj_rot.shape])

        ex_real, ex_imag = multislice_propagate_batch(
            obj_rot,
            probe_real, probe_imag,
            energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm,
            obj_batch_shape=[1, *probe_size, this_obj_size[-1]], binning=self.binning,
            fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
            unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
            scale_ri_by_k=self.scale_ri_by_k, is_minus_logged=self.is_minus_logged,
            pure_projection_return_sqrt=flag_pp_sqrt, shift_exit_wave=this_prj_offset)
        ex_mag_ls = w.norm(ex_real, ex_imag)

        return ex_mag_ls


class SparseMultisliceModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        super(SparseMultisliceModel, self).__init__(loss_function_type, distribution_mode, device, common_vars_dict,
                                                    raw_data_type, simulation_mode=simulation_mode)
        # ==========================================================================================
        # argument_ls must be in the same order as arguments in get_loss_function's function call!
        # ==========================================================================================
        args = inspect.getfullargspec(self.predict).args
        args.pop(0)
        self.argument_ls = args

    def predict(self, obj, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, slice_pos_cm_ls,
                prj_pos_offset):
        """
        Calculated predicted measurment. The signature of this function exactly matches the function returned
        by the ``get_loss_function`` method.

        :param obj: Array with shape [obj_size_y, obj_size_x, obj_size_z, 2]. Object function.
        :param probe_real: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Preal part of the probe.
        :param probe_imag: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Imaginary part of the probe.
        :param probe_defocus_mm: Probe defocus in mm.
        :param probe_pos_offset: Array with shape [n_theta, 2]. Probe position offset for each angle, in pixels.
        :param this_i_theta: Int. Current angle index.
        :param this_pos_batch: Array with shape [minibatch_size, 2]. Current batch of probe positions.
        :param prj: HDF5 measurement dataset handler.
        :param probe_pos_correction: Array with shape [n_probes, 2]. Additive correction values of probe positions.
        :param this_ind_batch: List of Int. Current batch of tile indices.
        :param slice_pos_cm_ls: Array of Float. Positions of object slices in cm.
        :param prj_pos_offset: Array with shape [n_theta, 2]. Projection position offset for each angle, in pixels. Note
            that this is different from probe_pos_offset - shifting is applied after passing through the object.
        :return: Array with shape [minibatch_size, len_probe_y, len_probe_x]. Magnitude of detected wavefields.
        """

        device_obj = self.common_vars['device_obj']
        lmbda_nm = self.common_vars['lmbda_nm']
        voxel_nm = self.common_vars['voxel_nm']
        probe_size = self.common_vars['probe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        minibatch_size = self.common_vars['minibatch_size']
        ds_level = self.common_vars['ds_level']
        this_obj_size = self.common_vars['this_obj_size']
        energy_ev = self.common_vars['energy_ev']
        psize_cm = self.common_vars['psize_cm']
        h = self.common_vars['h']
        pure_projection = self.common_vars['pure_projection']
        n_dp_batch = self.common_vars['n_dp_batch']
        free_prop_cm = self.common_vars['free_prop_cm']
        optimize_probe_defocusing = self.common_vars['optimize_probe_defocusing']
        optimize_probe_pos_offset = self.common_vars['optimize_probe_pos_offset']
        optimize_prj_pos_offset = self.common_vars['optimize_prj_pos_offset']
        optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
        debug = self.common_vars['debug']
        output_folder = self.common_vars['output_folder']
        unknown_type = self.common_vars['unknown_type']
        n_probe_modes = self.common_vars['n_probe_modes']
        n_theta = self.common_vars['n_theta']
        precalculate_rotation_coords = self.common_vars['precalculate_rotation_coords']
        theta_ls = self.common_vars['theta_ls']
        u = self.common_vars['u']
        v = self.common_vars['v']

        if precalculate_rotation_coords:
            coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                          theta_ls[this_i_theta], reverse=False)

        # Allocate subbatches.
        probe_pos_batch_ls = []
        i_dp = 0
        while i_dp < minibatch_size:
            probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
            i_dp += n_dp_batch

        this_pos_batch = np.round(this_pos_batch).astype(int)
        if optimize_probe_defocusing:
            h_probe = get_kernel(probe_defocus_mm * 1e6, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)
            h_probe_real, h_probe_imag = w.real(h_probe), w.imag(h_probe)
            probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_probe_real,
                                                                       h_probe_imag)

        if optimize_prj_pos_offset:
            this_prj_offset = prj_pos_offset[this_i_theta]
        else:
            this_prj_offset = None

        if optimize_probe_pos_offset:
            this_offset = probe_pos_offset[this_i_theta]
            probe_real, probe_imag = realign_image_fourier(probe_real, probe_imag, this_offset, axes=(1, 2), device=device_obj)

        if not two_d_mode and not self.distribution_mode:
            if not self.rotate_out_of_loop:
                if precalculate_rotation_coords:
                    obj_rot = apply_rotation(obj, coord_ls, device=device_obj)
                else:
                    obj_rot = rotate_no_grad(obj, theta_ls[this_i_theta], device=device_obj)
            else:
                obj_rot = obj
        else:
            obj_rot = obj

        ex_mag_ls = []

        # Pad if needed
        if not self.distribution_mode:
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch, probe_size, unknown_type=unknown_type)

        pos_ind = 0
        for k, pos_batch in enumerate(probe_pos_batch_ls):
            subobj_ls = []
            probe_real_ls = []
            probe_imag_ls = []

            # Get shifted probe list.
            for j in range(len(pos_batch)):
                if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                    this_shift = probe_pos_correction[this_i_theta, this_ind_batch[k * n_dp_batch + j]]
                    probe_real_shifted, probe_imag_shifted = realign_image_fourier(probe_real, probe_imag,
                                                                                   this_shift, axes=(1, 2),
                                                                                   device=device_obj)
                    probe_real_ls.append(probe_real_shifted)
                    probe_imag_ls.append(probe_imag_shifted)
            if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                # Shape of probe_xxx_ls.shape is [n_dp_batch, n_probe_modes, y, x].
                probe_real_ls = w.stack(probe_real_ls)
                probe_imag_ls = w.stack(probe_imag_ls)
            else:
                # Shape of probe_xxx_ls.shape is [n_probe_modes, y, x].
                probe_real_ls = probe_real
                probe_imag_ls = probe_imag

            # Get object list.
            if not self.distribution_mode:
                if len(pos_batch) == 1:
                    pos = pos_batch[0]
                    pos_y = pos[0] + pad_arr[0, 0]
                    pos_x = pos[1] + pad_arr[1, 0]
                    subobj = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :, :]
                    subobj_ls = w.reshape(subobj, [1, *subobj.shape])
                else:
                    for j in range(len(pos_batch)):
                        pos = pos_batch[j]
                        pos_y = pos[0] + pad_arr[0, 0]
                        pos_x = pos[1] + pad_arr[1, 0]
                        subobj = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :, :]
                        subobj_ls.append(subobj)
                    subobj_ls = w.stack(subobj_ls)
            else:
                subobj_ls = obj[pos_ind:pos_ind + len(pos_batch), :, :, :, :]
                pos_ind += len(pos_batch)

            gc.collect()
            if n_probe_modes == 1:
                if len(probe_real_ls.shape) == 3:
                    this_probe_real_ls = probe_real_ls[0, :, :]
                    this_probe_imag_ls = probe_imag_ls[0, :, :]
                else:
                    this_probe_real_ls = probe_real_ls[:, 0, :, :]
                    this_probe_imag_ls = probe_imag_ls[:, 0, :, :]
                ex_real, ex_imag = sparse_multislice_propagate_batch(u, v,
                                subobj_ls, this_probe_real_ls,
                                this_probe_imag_ls, energy_ev, psize_cm * ds_level, slice_pos_cm_ls, free_prop_cm=free_prop_cm,
                                obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                fresnel_approx=fresnel_approx, device=device_obj,
                                unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
                                scale_ri_by_k=self.scale_ri_by_k, shift_exit_wave=this_prj_offset)
                ex_mag_ls.append(w.norm(ex_real, ex_imag))
            else:
                for i_mode in range(n_probe_modes):
                    if len(probe_real_ls.shape) == 3:
                        this_probe_real_ls = probe_real_ls[i_mode, :, :]
                        this_probe_imag_ls = probe_imag_ls[i_mode, :, :]
                    else:
                        this_probe_real_ls = probe_real_ls[:, i_mode, :, :]
                        this_probe_imag_ls = probe_imag_ls[:, i_mode, :, :]
                    temp_real, temp_imag = sparse_multislice_propagate_batch(
                                u, v, subobj_ls,
                                this_probe_real_ls, this_probe_imag_ls,
                                energy_ev, psize_cm * ds_level, slice_pos_cm_ls, free_prop_cm=free_prop_cm,
                                obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                                fresnel_approx=fresnel_approx, device=device_obj,
                                unknown_type=unknown_type, normalize_fft=self.normalize_fft, sign_convention=self.sign_convention,
                                scale_ri_by_k=self.scale_ri_by_k, shift_exit_wave=this_prj_offset)
                    if i_mode == 0:
                        ex_int = temp_real ** 2 + temp_imag ** 2
                    else:
                        ex_int = ex_int + temp_real ** 2 + temp_imag ** 2
                ex_mag_ls.append(w.sqrt(ex_int))
        del subobj_ls, probe_real_ls, probe_imag_ls

        # Output shape is [minibatch_size, y, x].
        if len(ex_mag_ls) > 1:
            ex_mag_ls = w.concatenate(ex_mag_ls, 0)
        else:
            ex_mag_ls = ex_mag_ls[0]
        if rank == 0 and debug and self.i_call % 10 == 0:
            ex_mag_val = w.to_numpy(ex_mag_ls)
            dxchange.write_tiff(ex_mag_val, os.path.join(output_folder, 'intermediate', 'detected_mag'),
                                dtype='float32', overwrite=True)
        self.i_call += 1
        return ex_mag_ls

    def get_loss_function(self):
        def calculate_loss(obj, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch, slice_pos_cm_ls, prj_pos_offset):
            theta_downsample = self.common_vars['theta_downsample']
            ds_level = self.common_vars['ds_level']
            this_pred_batch = self.predict(obj, probe_real, probe_imag, probe_defocus_mm,
                                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                                           probe_pos_correction, this_ind_batch, slice_pos_cm_ls, prj_pos_offset)
            this_prj_batch = self.get_data(this_i_theta, this_ind_batch, theta_downsample=theta_downsample, ds_level=ds_level)
            loss = self.loss(this_pred_batch, this_prj_batch, obj)
            return loss
        return calculate_loss


class MultiDistModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', distribution_mode=None, device=None, common_vars_dict=None,
                 raw_data_type='magnitude', simulation_mode=False):
        super(MultiDistModel, self).__init__(loss_function_type, distribution_mode, device, common_vars_dict,
                                             raw_data_type, simulation_mode=simulation_mode)
        args = inspect.getfullargspec(self.predict).args
        args.pop(0)
        self.argument_ls = args

    def predict(self, obj, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width, prj_affine_ls, ctf_lg_kappa,
                prj_pos_offset):
        """
        Calculated predicted measurment. The signature of this function exactly matches the function returned
        by the ``get_loss_function`` method.

        :param obj: Array with shape [obj_size_y, obj_size_x, obj_size_z, 2]. Object function.
        :param probe_real: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Preal part of the probe.
        :param probe_imag: Array with shape [n_probe_modes, len_probe_y, len_probe_x]. Imaginary part of the probe.
        :param probe_defocus_mm: Probe defocus in mm.
        :param probe_pos_offset: Array with shape [n_theta, 2]. Probe position offset for each angle, in pixels.
        :param this_i_theta: Int. Current angle index.
        :param this_pos_batch: Array with shape [minibatch_size, 2]. Current batch of probe positions.
        :param prj: HDF5 measurement dataset handler.
        :param probe_pos_correction: Array with shape [n_probes, 2]. Additive correction values of probe positions.
        :param this_ind_batch: List of Int. Current batch of tile indices.
        :param free_prop_cm: Array of Float. Propagation distances for all holograms.
        :param safe_zone_width: Int. Width of safe zone to be padded to probes before propagation to prevent fringe wrapping.
        :param prj_affine_ls: Array with shape [n_dists, 2, 3]. Affine transform matrices for each hologram.
        :param ctf_lg_kappa: Log10 of kappa, the relation coefficient between delta and beta. If homogeneous assumption is not used, set as None.
        :param prj_pos_offset: Array with shape [n_theta, 2]. Projection position offset for each angle, in pixels. Note
            that this is different from probe_pos_offset - shifting is applied after passing through the object.
        :return: Array with shape [minibatch_size, len_probe_y, len_probe_x]. Magnitude of detected wavefields.
        """

        device_obj = self.common_vars['device_obj']
        lmbda_nm = self.common_vars['lmbda_nm']
        voxel_nm = self.common_vars['voxel_nm']
        probe_size = self.common_vars['probe_size']
        subprobe_size = self.common_vars['subprobe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        minibatch_size = self.common_vars['minibatch_size']
        ds_level = self.common_vars['ds_level']
        this_obj_size = self.common_vars['this_obj_size']
        energy_ev = self.common_vars['energy_ev']
        psize_cm = self.common_vars['psize_cm']
        h = self.common_vars['h']
        pure_projection = self.common_vars['pure_projection']
        n_dp_batch = self.common_vars['n_dp_batch']
        optimize_probe_defocusing = self.common_vars['optimize_probe_defocusing']
        optimize_probe_pos_offset = self.common_vars['optimize_probe_pos_offset']
        optimize_prj_pos_offset = self.common_vars['optimize_prj_pos_offset']
        optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
        optimize_free_prop = self.common_vars['optimize_free_prop']
        debug = self.common_vars['debug']
        output_folder = self.common_vars['output_folder']
        unknown_type = self.common_vars['unknown_type']
        beamstop = self.common_vars['beamstop']
        n_probe_modes = self.common_vars['n_probe_modes']
        n_theta = self.common_vars['n_theta']
        precalculate_rotation_coords = self.common_vars['precalculate_rotation_coords']
        theta_ls = self.common_vars['theta_ls']
        u_free = self.common_vars['u_free']
        v_free = self.common_vars['v_free']
        optimize_ctf_lg_kappa = self.common_vars['optimize_ctf_lg_kappa']

        kappa = 10 ** ctf_lg_kappa[0] if optimize_ctf_lg_kappa else None
        if precalculate_rotation_coords:
            coord_ls = read_origin_coords('arrsize_{}_{}_{}_ntheta_{}'.format(*this_obj_size, n_theta),
                                          theta_ls[this_i_theta], reverse=False)

        n_dists = len(free_prop_cm)
        n_blocks = prj.shape[1] // n_dists

        this_pos_batch = np.round(this_pos_batch).astype(int)
        if optimize_probe_defocusing:
            h_probe = get_kernel(probe_defocus_mm * 1e6, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)
            h_probe_real, h_probe_imag = w.real(h_probe), w.imag(h_probe)
            probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_probe_real,
                                                                       h_probe_imag)

        if optimize_prj_pos_offset:
            this_prj_offset = prj_pos_offset[this_i_theta]
        else:
            this_prj_offset = None

        # Allocate subbatches.
        probe_pos_batch_ls = []
        i_dp = 0
        while i_dp < minibatch_size:
            probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
            i_dp += n_dp_batch

        if not two_d_mode:
            if not self.rotate_out_of_loop:
                if precalculate_rotation_coords:
                    obj_rot = apply_rotation(obj, coord_ls, device=device_obj)
                else:
                    obj_rot = rotate_no_grad(obj, theta_ls[this_i_theta], device=device_obj)
            else:
                obj_rot = obj
        else:
            obj_rot = obj

        # Pad object with safe zone width if not using low-mem mode (chunks will be read padded otherwise).
        szw_arr = np.array([safe_zone_width] * 2)
        if not self.distribution_mode:
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch - szw_arr, subprobe_size + 2 * szw_arr, unknown_type=unknown_type)

        # Pad probe with safe zone width.
        if safe_zone_width > 0:
            pad_arr = calculate_pad_len(probe_size, this_pos_batch - szw_arr, subprobe_size + 2 * szw_arr)
            probe_real_sz = w.pad(probe_real, [[0, 0]] + pad_arr, mode='constant', constant_values=1)
            probe_imag_sz = w.pad(probe_imag, [[0, 0]] + pad_arr, mode='constant', constant_values=0)
        else:
            probe_real_sz = probe_real
            probe_imag_sz = probe_imag
            pad_arr = np.array([[0, 0], [0, 0]])

        subobj_ls_ls = []
        subprobe_real_ls_ls = []
        subprobe_imag_ls_ls = []
        pos_ind = 0
        for k, pos_batch in enumerate(probe_pos_batch_ls):
            if self.distribution_mode is None:
                if n_blocks > 1:
                    subobj_subbatch_ls = []
                    subprobe_subbatch_real_ls = []
                    subprobe_subbatch_imag_ls = []
                    if len(pos_batch) == 1:
                        pos = pos_batch[0]
                        pos_y = pos[0] + pad_arr[0, 0]
                        pos_x = pos[1] + pad_arr[1, 0]
                        subobj = obj_rot[pos_y:pos_y + probe_size[0], pos_x:pos_x + probe_size[1], :, :]
                        subobj_subbatch_ls = w.reshape(subobj, [1, *subobj.shape])
                        sub_probe_real = probe_real_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                                        pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                        sub_probe_imag = probe_imag_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                                        pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                        subprobe_subbatch_real_ls = w.reshape(sub_probe_real, [1, *sub_probe_real.shape])
                        subprobe_subbatch_imag_ls = w.reshape(sub_probe_imag, [1, *sub_probe_imag.shape])

                    else:
                        for j in range(len(pos_batch)):
                            pos = pos_batch[j]
                            pos_y = pos[0] + pad_arr[0, 0] - safe_zone_width
                            pos_x = pos[1] + pad_arr[1, 0] - safe_zone_width
                            subobj = obj_rot[pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                             pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2, :, :]
                            sub_probe_real = probe_real_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                                              pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                            sub_probe_imag = probe_imag_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                                              pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                            subobj_subbatch_ls.append(subobj)
                            subprobe_subbatch_real_ls.append(sub_probe_real)
                            subprobe_subbatch_imag_ls.append(sub_probe_imag)
                        subobj_subbatch_ls = w.stack(subobj_subbatch_ls)
                        subprobe_subbatch_real_ls = w.stack(subprobe_subbatch_real_ls)
                        subprobe_subbatch_imag_ls = w.stack(subprobe_subbatch_imag_ls)
                else:
                    subobj_subbatch_ls = w.reshape(obj_rot, [1, *obj_rot.shape])
                    subprobe_subbatch_real_ls = w.reshape(probe_real_sz, [1, *probe_real_sz.shape])
                    subprobe_subbatch_imag_ls = w.reshape(probe_imag_sz, [1, *probe_imag_sz.shape])
                # Shape of subprobe_real_ls_ls is [n_subbatches, len(pos_batch), y, x, z].
                subobj_ls_ls.append(subobj_subbatch_ls)
            else:
                subobj_ls_ls.append(obj[pos_ind:pos_ind + len(pos_batch)])
                for j in range(len(pos_batch)):
                    pos = pos_batch[j]
                    pos_y = pos[0] + pad_arr[0, 0] - safe_zone_width
                    pos_x = pos[1] + pad_arr[1, 0] - safe_zone_width
                    sub_probe_real = probe_real_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                     pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                    sub_probe_imag = probe_imag_sz[:, pos_y:pos_y + subprobe_size[0] + safe_zone_width * 2,
                                     pos_x:pos_x + subprobe_size[1] + safe_zone_width * 2]
                    subprobe_subbatch_real_ls.append(sub_probe_real)
                    subprobe_subbatch_imag_ls.append(sub_probe_imag)
                subprobe_subbatch_real_ls = w.stack(subprobe_subbatch_real_ls)
                subprobe_subbatch_imag_ls = w.stack(subprobe_subbatch_imag_ls)

            # Shape of subprobe_real_ls_ls is [n_subbatches, len(pos_batch), n_probe_modes, y, x].
            subprobe_real_ls_ls.append(subprobe_subbatch_real_ls)
            subprobe_imag_ls_ls.append(subprobe_subbatch_imag_ls)
            pos_ind += len(pos_batch)

        ex_mag_ls = []

        for i_dist, this_dist in enumerate(free_prop_cm):
            for k, pos_batch in enumerate(probe_pos_batch_ls):
                for i_mode in range(n_probe_modes):
                    if self.forward_algorithm == 'fresnel':
                        temp_real, temp_imag = multislice_propagate_batch(
                            subobj_ls_ls[k],
                            subprobe_real_ls_ls[k][:, i_mode, :, :], subprobe_imag_ls_ls[k][i_mode, :, :],
                            energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=this_dist, binning=self.binning,
                            obj_batch_shape=[len(pos_batch), subprobe_size[0] + 2 * safe_zone_width, subprobe_size[1] + 2 * safe_zone_width, this_obj_size[-1]],
                            fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
                            unknown_type=unknown_type, sign_convention=self.sign_convention, optimize_free_prop=optimize_free_prop,
                            u_free=u_free, v_free=v_free, scale_ri_by_k=self.scale_ri_by_k, kappa=kappa, shift_exit_wave=this_prj_offset)
                    elif self.forward_algorithm == 'ctf':
                        temp_real, temp_imag = modulate_and_get_ctf(subobj_ls_ls[k], energy_ev, this_dist, u_free, v_free, kappa=10 ** ctf_lg_kappa[0])
                    else:
                        raise ValueError('Invalid value for "forward_algorithm". ')
                    if i_mode == 0:
                        ex_int = temp_real ** 2 + temp_imag ** 2
                    else:
                        ex_int = ex_int + temp_real ** 2 + temp_imag ** 2
                ex_mag_ls.append(w.sqrt(ex_int))

        # Output shape is [minibatch_size, y, x].
        if len(ex_mag_ls) > 1:
            ex_mag_ls = w.concatenate(ex_mag_ls, 0)
        else:
            ex_mag_ls = ex_mag_ls[0]

        if safe_zone_width > 0:
            ex_mag_ls = ex_mag_ls[:, safe_zone_width:safe_zone_width + subprobe_size[0],
                                     safe_zone_width:safe_zone_width + subprobe_size[1]]

        if rank == 0 and debug:
            ex_mag_val = w.to_numpy(ex_mag_ls)
            dxchange.write_tiff(ex_mag_val, os.path.join(output_folder, 'intermediate', 'detected_mag'), dtype='float32', overwrite=True)

        del subobj_ls_ls, subprobe_real_ls_ls, subprobe_imag_ls_ls
        self.i_call += 1
        return ex_mag_ls

    def get_loss_function(self):
        def calculate_loss(obj, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width,
                           prj_affine_ls, ctf_lg_kappa, prj_pos_offset):
            theta_downsample = self.common_vars['theta_downsample']
            ds_level = self.common_vars['ds_level']
            optimize_probe_pos_offset = self.common_vars['optimize_probe_pos_offset']
            optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
            optimize_prj_affine = self.common_vars['optimize_prj_affine']

            this_pred_batch = self.predict(obj, probe_real, probe_imag, probe_defocus_mm,
                                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                                           probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width,
                                           prj_affine_ls, ctf_lg_kappa, prj_pos_offset)

            if theta_downsample is None: theta_downsample = 1 
            n_dists = len(free_prop_cm)
            n_blocks = prj.shape[1] // n_dists
            this_ind_batch_full = this_ind_batch
            for i in range(1, n_dists):
                this_ind_batch_full = np.concatenate([this_ind_batch_full, this_ind_batch + i * n_blocks])
            this_prj_batch = prj[this_i_theta * theta_downsample, this_ind_batch_full]
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
            if ds_level > 1:
                this_prj_batch = this_prj_batch[:, :ds_level, ::ds_level]

            if optimize_prj_affine:
                scaled_prj_ls = []
                for i in range(n_dists):
                    this_prj_batch_idist = this_prj_batch[len(this_ind_batch) * i:len(this_ind_batch) * (i + 1)]
                    this_prj_batch_idist = w.affine_transform(this_prj_batch_idist, prj_affine_ls[i])
                    scaled_prj_ls.append(this_prj_batch_idist)
                this_prj_batch = w.concatenate(scaled_prj_ls)

            if optimize_probe_pos_offset:
                this_offset = probe_pos_offset[this_i_theta]
                this_prj_batch, _ = realign_image_fourier(this_prj_batch, w.zeros_like(this_prj_batch), this_offset, axes=(0, 1),
                                                               device=self.device)

            if optimize_all_probe_pos:
                shifted_prj_ls = []
                for i in range(n_dists):
                    this_shift = probe_pos_correction[i]
                    this_prj_batch_idist = this_prj_batch[len(this_ind_batch) * i:len(this_ind_batch) * (i + 1)]
                    this_prj_batch_idist, _ = realign_image_fourier(this_prj_batch_idist, w.zeros_like(this_prj_batch_idist),
                                                                  this_shift, axes=(1, 2),
                                                                  device=self.device)
                    shifted_prj_ls.append(this_prj_batch_idist)
                this_prj_batch = w.concatenate(shifted_prj_ls)

            loss = self.loss(this_pred_batch, this_prj_batch, obj)
            return loss
        return calculate_loss
