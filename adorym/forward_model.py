import numpy as np

import gc

import adorym.wrappers as w
from adorym.util import *
from adorym.propagate import multislice_propagate_batch, get_kernel

class ForwardModel(object):

    def __init__(self, loss_function_type='lsq', shared_file_object=False, device=None, common_vars_dict=None, raw_data_type='magnitude'):
        self.loss_function_type = loss_function_type
        self.argument_ls = []
        self.regularizer_dict = {}
        self.shared_file_object = shared_file_object
        self.device = device
        self.current_loss = 0
        self.common_vars = common_vars_dict
        self.raw_data_type = raw_data_type
        self.unknown_type = common_vars_dict['unknown_type']

    def add_regularizer(self, name, reg_dict):
        self.regularizer_dict[name] = reg_dict

    def add_l1_norm(self, alpha_d, alpha_b):
        d = {'alpha_d': alpha_d,
             'alpha_b': alpha_b}
        self.add_regularizer('l1_norm', d)

    def add_reweighted_l1_norm(self, alpha_d, alpha_b, weight_l1):
        d = {'alpha_d': alpha_d,
             'alpha_b': alpha_b,
             'weight_l1': weight_l1}
        self.add_regularizer('reweighted_l1', d)

    def add_tv(self, gamma):
        d = {'gamma': gamma}
        self.add_regularizer('tv', d)

    def update_l1_weight(self, weight_l1):
        self.regularizer_dict['reweighted_l1']['weight_l1'] = weight_l1

    def get_regularization_value(self, obj_delta, obj_beta):
        reg = w.create_variable(0., device=self.device)
        for name in list(self.regularizer_dict):
            if name == 'l1_norm':
                reg += l1_norm_term(obj_delta, obj_beta,
                                    self.regularizer_dict[name]['alpha_d'],
                                    self.regularizer_dict[name]['alpha_b'],
                                    device=self.device)
            elif name == 'reweighted_l1_norm':
                reg += reweighted_l1_norm_term(obj_delta, obj_beta,
                                               self.regularizer_dict[name]['alpha_d'],
                                               self.regularizer_dict[name]['alpha_b'],
                                               self.regularizer_dict[name]['weight_l1'],
                                               device=self.device)
            elif name == 'tv':
                if self.unknown_type == 'delta_beta':
                    reg += tv(obj_delta, obj_beta,
                              self.regularizer_dict[name]['gamma'],
                              self.shared_file_object, device=self.device)
                elif self.unknown_type == 'real_imag':
                    reg += tv(w.arctan2(obj_beta, obj_delta), None,
                              self.regularizer_dict[name]['gamma'],
                              self.shared_file_object, device=self.device)
        return reg

    def get_argument_index(self, arg):
        for i, a in enumerate(self.argument_ls):
            if a == arg:
                return i
        raise ValueError('{} is not in the argument list.'.format(arg))


class PtychographyModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', shared_file_object=False, device=None, common_vars_dict=None, raw_data_type='magnitude'):
        super(PtychographyModel, self).__init__(loss_function_type, shared_file_object, device, common_vars_dict, raw_data_type)
        # ==========================================================================================
        # argument_ls must be in the same order as arguments in get_loss_function's function call!
        # ==========================================================================================
        self.argument_ls = ['obj_delta', 'obj_beta', 'probe_real', 'probe_imag', 'probe_defocus_mm',
                            'probe_pos_offset', 'this_i_theta', 'this_pos_batch', 'prj',
                            'probe_pos_correction', 'this_ind_batch']

    def predict(self, obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch):

        device_obj = self.common_vars['device_obj']
        lmbda_nm = self.common_vars['lmbda_nm']
        voxel_nm = self.common_vars['voxel_nm']
        probe_size = self.common_vars['probe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        coord_ls = self.common_vars['coord_ls']
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
        optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
        debug = self.common_vars['debug']
        output_folder = self.common_vars['output_folder']
        unknown_type = self.common_vars['unknown_type']

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

        if optimize_probe_pos_offset:
            this_offset = probe_pos_offset[this_i_theta]
            probe_real, probe_imag = realign_image_fourier(probe_real, probe_imag, this_offset, axes=(0, 1), device=device_obj)

        if not self.shared_file_object:
            obj_stack = w.stack([obj_delta, obj_beta], axis=3)
            if not two_d_mode:
                obj_rot = apply_rotation(obj_stack, coord_ls[this_i_theta], device=device_obj)
                # obj_rot = sp_rotate(obj_stack, theta, axes=(1, 2), reshape=False)
            else:
                obj_rot = obj_stack
            ex_real_ls = []
            ex_imag_ls = []

            # Pad if needed
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch, probe_size, unknown_type=unknown_type)

            for k, pos_batch in enumerate(probe_pos_batch_ls):
                subobj_ls = []
                probe_real_ls = []
                probe_imag_ls = []
                for j in range(len(pos_batch)):
                    pos = pos_batch[j]
                    # pos = [int(x) for x in pos]
                    pos[0] = pos[0] + pad_arr[0, 0]
                    pos[1] = pos[1] + pad_arr[1, 0]
                    subobj = obj_rot[pos[0]:pos[0] + probe_size[0], pos[1]:pos[1] + probe_size[1], :, :]
                    subobj_ls.append(subobj)
                    if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                        this_shift = probe_pos_correction[this_i_theta, this_ind_batch[k * n_dp_batch + j]]
                        probe_real_shifted, probe_imag_shifted = realign_image_fourier(probe_real, probe_imag,
                                                                                       this_shift, axes=(0, 1),
                                                                                       device=device_obj)
                        probe_real_ls.append(probe_real_shifted)
                        probe_imag_ls.append(probe_imag_shifted)

                subobj_ls = w.stack(subobj_ls)
                if optimize_all_probe_pos or len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                    probe_real_ls = w.stack(probe_real_ls)
                    probe_imag_ls = w.stack(probe_imag_ls)
                else:
                    probe_real_ls = probe_real
                    probe_imag_ls = probe_imag
                gc.collect()
                ex_real, ex_imag = multislice_propagate_batch(
                    subobj_ls[:, :, :, :, 0], subobj_ls[:, :, :, :, 1], probe_real_ls,
                    probe_imag_ls, energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=free_prop_cm,
                    obj_batch_shape=[len(pos_batch), *probe_size, this_obj_size[-1]],
                    fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
                    type=unknown_type)
                ex_real_ls.append(ex_real)
                ex_imag_ls.append(ex_imag)
            del subobj_ls, probe_real_ls, probe_imag_ls
        else:
            ex_real_ls = []
            ex_imag_ls = []

            pos_ind = 0
            for k, pos_batch in enumerate(probe_pos_batch_ls):
                subobj_ls_delta = obj_delta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                subobj_ls_beta = obj_beta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                ex_real, ex_imag = multislice_propagate_batch(subobj_ls_delta, subobj_ls_beta, probe_real,
                                                              probe_imag, energy_ev, psize_cm * ds_level, kernel=h,
                                                              free_prop_cm=free_prop_cm,
                                                              obj_batch_shape=[len(pos_batch), *probe_size,
                                                                               this_obj_size[-1]],
                                                              fresnel_approx=fresnel_approx,
                                                              pure_projection=pure_projection,
                                                              device=device_obj,
                                                              type=unknown_type)
                ex_real_ls.append(ex_real)
                ex_imag_ls.append(ex_imag)
                pos_ind += len(pos_batch)
        ex_real_ls = w.concatenate(ex_real_ls, 0)
        ex_imag_ls = w.concatenate(ex_imag_ls, 0)

        # if beamstop is not None:
        #     beamstop_mask, beamstop_value = beamstop
        #     beamstop_mask[beamstop_mask >= 1e-5] = 1
        #     beamstop_mask[beamstop_mask < 1e-5] = 0
            # ex_real_ls = ex_real_ls * (1 - beamstop_mask) + beamstop_value * beamstop_mask
            # ex_imag_ls = ex_imag_ls * (1 - beamstop_mask) + beamstop_value * beamstop_mask

        if rank == 0 and debug:
            ex_real_val = w.to_numpy(ex_real_ls)
            ex_imag_val = w.to_numpy(ex_imag_ls)
            dxchange.write_tiff(np.sqrt(ex_real_val ** 2 + ex_imag_val ** 2), os.path.join(output_folder, 'intermediate', 'detected_mag'), dtype='float32', overwrite=True)
            dxchange.write_tiff(np.arctan2(ex_real_val, ex_imag_val), os.path.join(output_folder, 'intermediate', 'detected_phase'), dtype='float32', overwrite=True)
        return ex_real_ls, ex_imag_ls

    def get_loss_function(self):
        def calculate_loss(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch):
            ex_real_ls, ex_imag_ls = self.predict(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch)

            beamstop = self.common_vars['beamstop']
            ds_level = self.common_vars['ds_level']

            this_prj_batch = prj[this_i_theta, this_ind_batch]
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
            if ds_level > 1:
                this_prj_batch = this_prj_batch[:, ::ds_level, ::ds_level]

            if beamstop is not None:
                beamstop_mask, beamstop_value = beamstop
                beamstop_mask[beamstop_mask >= 1e-5] = 1
                beamstop_mask[beamstop_mask < 1e-5] = 0
                beamstop_mask = w.cast(beamstop_mask, 'bool')
                beamstop_mask_stack = w.tile(beamstop_mask, [len(ex_real_ls), 1, 1])
                ex_real_ls = w.reshape(ex_real_ls[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
                ex_imag_ls = w.reshape(ex_imag_ls[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
                this_prj_batch = w.reshape(this_prj_batch[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])

            if self.loss_function_type == 'lsq':
                if self.raw_data_type == 'magnitude':
                    loss = w.mean((w.norm(ex_real_ls, ex_imag_ls) - w.abs(this_prj_batch)) ** 2)
                elif self.raw_data_type == 'intensity':
                    loss = w.mean((w.norm(ex_real_ls, ex_imag_ls) - w.sqrt(w.abs(this_prj_batch))) ** 2)
            elif self.loss_function_type == 'poisson':
                if self.raw_data_type == 'magnitude':
                    loss = w.mean(w.norm(ex_real_ls, ex_imag_ls) ** 2 - w.abs(this_prj_batch) ** 2 * w.log(w.norm(ex_real_ls, ex_imag_ls) ** 2))
                elif self.raw_data_type == 'intensity':
                    loss = w.mean(w.norm(ex_real_ls, ex_imag_ls) ** 2 - w.abs(this_prj_batch) * w.log(w.norm(ex_real_ls, ex_imag_ls) ** 2))
            loss = loss + self.get_regularization_value(obj_delta, obj_beta)
            self.current_loss = float(w.to_numpy(loss))
            del ex_real_ls, ex_imag_ls
            del this_prj_batch
            return loss
        return calculate_loss


class MultiDistModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', shared_file_object=False, device=None, common_vars_dict=None, raw_data_type='magnitude'):
        super(MultiDistModel, self).__init__(loss_function_type, shared_file_object, device, common_vars_dict, raw_data_type)
        # ==========================================================================================
        # argument_ls must be in the same order as arguments in get_loss_function's function call!
        # ==========================================================================================
        self.argument_ls = ['obj_delta', 'obj_beta', 'probe_real', 'probe_imag', 'probe_defocus_mm',
                            'probe_pos_offset', 'this_i_theta', 'this_pos_batch', 'prj',
                            'probe_pos_correction', 'this_ind_batch', 'free_prop_cm', 'safe_zone_width']

    def predict(self, obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, prj,
                probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width):

        device_obj = self.common_vars['device_obj']
        lmbda_nm = self.common_vars['lmbda_nm']
        voxel_nm = self.common_vars['voxel_nm']
        probe_size = self.common_vars['probe_size']
        subprobe_size = self.common_vars['subprobe_size']
        fresnel_approx = self.common_vars['fresnel_approx']
        two_d_mode = self.common_vars['two_d_mode']
        coord_ls = self.common_vars['coord_ls']
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
        optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
        debug = self.common_vars['debug']
        output_folder = self.common_vars['output_folder']
        unknown_type = self.common_vars['unknown_type']
        beamstop = self.common_vars['beamstop']

        n_dists = len(free_prop_cm)
        n_blocks = prj.shape[1] // n_dists

        this_pos_batch = np.round(this_pos_batch).astype(int)
        if optimize_probe_defocusing:
            h_probe = get_kernel(probe_defocus_mm * 1e6, lmbda_nm, voxel_nm, probe_size, fresnel_approx=fresnel_approx)
            h_probe_real, h_probe_imag = w.real(h_probe), w.imag(h_probe)
            probe_real, probe_imag = w.convolve_with_transfer_function(probe_real, probe_imag, h_probe_real,
                                                                       h_probe_imag)

        # Allocate subbatches.
        probe_pos_batch_ls = []
        i_dp = 0
        while i_dp < minibatch_size:
            probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
            i_dp += n_dp_batch

        if not self.shared_file_object:
            obj_stack = w.stack([obj_delta, obj_beta], axis=3)
            if not two_d_mode:
                obj_rot = apply_rotation(obj_stack, coord_ls[this_i_theta], device=device_obj)
                # obj_rot = sp_rotate(obj_stack, theta, axes=(1, 2), reshape=False)
            else:
                obj_rot = obj_stack

            # Pad object and probe with safe zone width
            if safe_zone_width > 0:
                szw_arr = np.array([safe_zone_width] * 2)
                obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch - szw_arr, subprobe_size + 2 * szw_arr, unknown_type=unknown_type)
                probe_real_sz = w.pad(probe_real, pad_arr, mode='constant', constant_values=1)
                probe_imag_sz = w.pad(probe_imag, pad_arr, mode='constant', constant_values=0)
            else:
                probe_real_sz = probe_real
                probe_imag_sz = probe_imag
                pad_arr = np.array([[0, 0], [0, 0]])

            # Take out object and probe patches.
            subobj_ls_ls = []
            subprobe_real_ls_ls = []
            subprobe_imag_ls_ls = []
            for k, pos_batch in enumerate(probe_pos_batch_ls):
                if n_blocks > 1:
                    subobj_subbatch_ls = []
                    subprobe_subbatch_real_ls = []
                    subprobe_subbatch_imag_ls = []
                    for j in range(len(pos_batch)):
                        pos = pos_batch[j]
                        pos[0] = pos[0] + pad_arr[0, 0] - safe_zone_width
                        pos[1] = pos[1] + pad_arr[1, 0] - safe_zone_width
                        subobj = obj_rot[pos[0]:pos[0] + subprobe_size[0] + safe_zone_width * 2,
                                         pos[1]:pos[1] + subprobe_size[1] + safe_zone_width * 2, :, :]
                        sub_probe_real = probe_real_sz[pos[0]:pos[0] + subprobe_size[0] + safe_zone_width * 2,
                                                       pos[1]:pos[1] + subprobe_size[1] + safe_zone_width * 2]
                        sub_probe_imag = probe_imag_sz[pos[0]:pos[0] + subprobe_size[0] + safe_zone_width * 2,
                                                       pos[1]:pos[1] + subprobe_size[1] + safe_zone_width * 2]
                        subobj_subbatch_ls.append(subobj)
                        subprobe_subbatch_real_ls.append(sub_probe_real)
                        subprobe_subbatch_imag_ls.append(sub_probe_imag)
                    subobj_subbatch_ls = w.stack(subobj_subbatch_ls)
                    subprobe_subbatch_real_ls = w.stack(subprobe_subbatch_real_ls)
                    subprobe_subbatch_imag_ls = w.stack(subprobe_subbatch_imag_ls)
                else:
                    subobj_subbatch_ls = w.stack([obj_rot])
                    subprobe_subbatch_real_ls = w.stack([probe_real_sz])
                    subprobe_subbatch_imag_ls = w.stack([probe_imag_sz])
                subobj_ls_ls.append(subobj_subbatch_ls)
                subprobe_real_ls_ls.append(subprobe_subbatch_real_ls)
                subprobe_imag_ls_ls.append(subprobe_subbatch_imag_ls)

            ex_real_ls = []
            ex_imag_ls = []
            for i_dist, this_dist in enumerate(free_prop_cm):
                for k, pos_batch in enumerate(probe_pos_batch_ls):
                    ex_real, ex_imag = multislice_propagate_batch(
                        subobj_ls_ls[k][:, :, :, :, 0], subobj_ls_ls[k][:, :, :, :, 1], subprobe_real_ls_ls[k],
                        subprobe_imag_ls_ls[k], energy_ev, psize_cm * ds_level, kernel=h, free_prop_cm=this_dist,
                        obj_batch_shape=[len(pos_batch), subprobe_size[0] + 2 * safe_zone_width, subprobe_size[1] + 2 * safe_zone_width, this_obj_size[-1]],
                        fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj,
                        type=unknown_type)

                    if safe_zone_width > 0:
                        ex_real = ex_real[:, safe_zone_width:safe_zone_width + subprobe_size[0],
                                             safe_zone_width:safe_zone_width + subprobe_size[1]]
                        ex_imag = ex_imag[:, safe_zone_width:safe_zone_width + subprobe_size[0],
                                  safe_zone_width:safe_zone_width + subprobe_size[1]]
                    ex_real_ls.append(ex_real)
                    ex_imag_ls.append(ex_imag)
            ex_real_ls = w.concatenate(ex_real_ls)
            ex_imag_ls = w.concatenate(ex_imag_ls)
            del subobj_ls_ls, subprobe_real_ls_ls, subprobe_imag_ls_ls
        else:
            # Pad object and probe with safe zone width
            if safe_zone_width > 0:
                szw_arr = np.array([safe_zone_width] * 2)
                probe_real_sz, pad_arr = pad_object(probe_real, probe_size, this_pos_batch - szw_arr, subprobe_size + 2 * szw_arr)
                probe_imag_sz, pad_arr = pad_object(probe_imag, probe_size, this_pos_batch - szw_arr, subprobe_size + 2 * szw_arr)
            else:
                probe_real_sz = probe_real
                probe_imag_sz = probe_imag

            subprobe_real_ls_ls = []
            subprobe_imag_ls_ls = []
            for k, pos_batch in enumerate(probe_pos_batch_ls):
                if n_blocks > 1:
                    subprobe_subbatch_real_ls = []
                    subprobe_subbatch_imag_ls = []
                    for j in range(len(pos_batch)):
                        pos = pos_batch[j]
                        pos[0] = pos[0] + pad_arr[0, 0] - safe_zone_width
                        pos[1] = pos[1] + pad_arr[1, 0] - safe_zone_width
                        sub_probe_real = probe_real_sz[pos[0]:pos[0] + probe_size[0] + safe_zone_width * 2,
                                         pos[1]:pos[1] + probe_size[1] + safe_zone_width * 2]
                        sub_probe_imag = probe_imag_sz[pos[0]:pos[0] + probe_size[0] + safe_zone_width * 2,
                                         pos[1]:pos[1] + probe_size[1] + safe_zone_width * 2]
                        subprobe_subbatch_real_ls.append(sub_probe_real)
                        subprobe_subbatch_imag_ls.append(sub_probe_imag)
                    subprobe_subbatch_real_ls = w.stack(subprobe_subbatch_real_ls)
                    subprobe_subbatch_imag_ls = w.stack(subprobe_subbatch_imag_ls)
                else:
                    subprobe_subbatch_real_ls = w.stack([probe_real_sz])
                    subprobe_subbatch_imag_ls = w.stack([probe_imag_sz])
                subprobe_real_ls_ls.append(subprobe_subbatch_real_ls)
                subprobe_imag_ls_ls.append(subprobe_subbatch_imag_ls)

            ex_real_ls = []
            ex_imag_ls = []
            pos_ind = 0
            for i_dist in range(n_dists):
                for k, pos_batch in enumerate(probe_pos_batch_ls):
                    subobj_ls_delta = obj_delta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                    subobj_ls_beta = obj_beta[pos_ind:pos_ind + len(pos_batch), :, :, :]
                    subprobe_real_ls = subprobe_real_ls_ls[k]
                    subprobe_imag_ls = subprobe_imag_ls_ls[k]
                    ex_real, ex_imag = multislice_propagate_batch(subobj_ls_delta, subobj_ls_beta, subprobe_real_ls,
                                                                  subprobe_imag_ls, energy_ev, psize_cm * ds_level, kernel=h,
                                                                  free_prop_cm=free_prop_cm,
                                                                  obj_batch_shape=[len(pos_batch), *probe_size,
                                                                                   this_obj_size[-1]],
                                                                  fresnel_approx=fresnel_approx,
                                                                  pure_projection=pure_projection,
                                                                  device=device_obj,
                                                                  type=unknown_type)
                    ex_real_ls.append(ex_real)
                    ex_imag_ls.append(ex_imag)
                    pos_ind += len(pos_batch)
            ex_real_ls = w.concatenate(ex_real_ls)
            ex_imag_ls = w.concatenate(ex_imag_ls)

        if rank == 0 and debug:
            ex_real_val = w.to_numpy(ex_real_ls)
            ex_imag_val = w.to_numpy(ex_imag_ls)
            dxchange.write_tiff(np.sqrt(ex_real_val ** 2 + ex_imag_val ** 2)[:, :, :, np.newaxis], os.path.join(output_folder, 'intermediate', 'detected_mag'), dtype='float32', overwrite=True)
            dxchange.write_tiff(np.arctan2(ex_imag_val, ex_real_val)[:, :, :, np.newaxis], os.path.join(output_folder, 'intermediate', 'detected_phase'), dtype='float32', overwrite=True)
        return ex_real_ls, ex_imag_ls

    def get_loss_function(self):
        def calculate_loss(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width):
            ex_real_ls, ex_imag_ls = self.predict(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, prj,
                           probe_pos_correction, this_ind_batch, free_prop_cm, safe_zone_width)

            beamstop = self.common_vars['beamstop']
            ds_level = self.common_vars['ds_level']
            optimize_probe_pos_offset = self.common_vars['optimize_probe_pos_offset']
            optimize_all_probe_pos = self.common_vars['optimize_all_probe_pos']
            device_obj = self.common_vars['device_obj']
            minibatch_size =self.common_vars['minibatch_size']

            n_dists = len(free_prop_cm)
            n_blocks = prj.shape[1] // n_dists
            this_ind_batch_full = this_ind_batch
            for i in range(1, n_dists):
                this_ind_batch_full = np.concatenate([this_ind_batch_full, this_ind_batch + i * n_blocks])
            this_prj_batch = prj[this_i_theta, this_ind_batch_full]
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
            if ds_level > 1:
                this_prj_batch = this_prj_batch[:, :ds_level, ::ds_level]

            if optimize_probe_pos_offset:
                this_offset = probe_pos_offset[this_i_theta]
                this_prj_batch, _ = realign_image_fourier(this_prj_batch, w.zeros_like(this_prj_batch), this_offset, axes=(0, 1),
                                                               device=device_obj)

            if optimize_all_probe_pos:
                shifted_prj_ls = []
                for i in range(n_dists):
                    this_shift = probe_pos_correction[i]
                    this_prj_batch_idist = this_prj_batch[len(this_ind_batch) * i:len(this_ind_batch) * (i + 1)]
                    this_prj_batch_idist, _ = realign_image_fourier(this_prj_batch_idist, w.zeros_like(this_prj_batch_idist),
                                                                  this_shift, axes=(1, 2),
                                                                  device=device_obj)
                    shifted_prj_ls.append(this_prj_batch_idist)
                this_prj_batch = w.concatenate(shifted_prj_ls)

            if beamstop is not None:
                beamstop_mask, beamstop_value = beamstop
                beamstop_mask[beamstop_mask >= 1e-5] = 1
                beamstop_mask[beamstop_mask < 1e-5] = 0
                beamstop_mask = w.cast(beamstop_mask, 'bool')
                beamstop_mask_stack = w.tile(beamstop_mask, [len(ex_real_ls), 1, 1])
                ex_real_ls = w.reshape(ex_real_ls[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
                ex_imag_ls = w.reshape(ex_imag_ls[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])
                this_prj_batch = w.reshape(this_prj_batch[beamstop_mask_stack], [beamstop_mask_stack.shape[0], -1])

            if self.loss_function_type == 'lsq':
                if self.raw_data_type == 'magnitude':
                    loss = w.mean((w.norm(ex_real_ls, ex_imag_ls) - w.abs(this_prj_batch)) ** 2)
                elif self.raw_data_type == 'intensity':
                    loss = w.mean((w.norm(ex_real_ls, ex_imag_ls) - w.sqrt(w.abs(this_prj_batch))) ** 2)
            elif self.loss_function_type == 'poisson':
                if self.raw_data_type == 'magnitude':
                    loss = w.mean(w.norm(ex_real_ls, ex_imag_ls) ** 2 - w.abs(this_prj_batch) ** 2 * w.log(w.norm(ex_real_ls, ex_imag_ls) ** 2))
                elif self.raw_data_type == 'intensity':
                    loss = w.mean(w.norm(ex_real_ls, ex_imag_ls) ** 2 - w.abs(this_prj_batch) * w.log(w.norm(ex_real_ls, ex_imag_ls) ** 2))
            loss = loss + self.get_regularization_value(obj_delta, obj_beta)
            self.current_loss = float(w.to_numpy(loss))

            del ex_real_ls, ex_imag_ls
            del this_prj_batch
            return loss
        return calculate_loss


def l1_norm_term(obj_delta, obj_beta, alpha_d, alpha_b, device=None):
    reg = w.create_variable(0., device=device)
    if alpha_d not in [None, 0]:
        reg += alpha_d * w.mean(w.abs(obj_delta))
    if alpha_b not in [None, 0]:
        reg += alpha_b * w.mean(w.abs(obj_beta))
    return reg

def reweighted_l1_norm_term(obj_delta, obj_beta, alpha_d, alpha_b, weight_l1, device=None):
    reg = w.create_variable(0., device=device)
    if alpha_d not in [None, 0]:
        reg += alpha_d * w.mean(weight_l1 * w.abs(obj_delta))
    if alpha_b not in [None, 0]:
        reg += alpha_b * w.mean(weight_l1 * w.abs(obj_beta))
    return reg

def tv(obj_delta, obj_beta, gamma, shared_file_object, device=None):
    reg = w.create_variable(0., device=device)
    if shared_file_object:
        reg += gamma * total_variation_3d(obj_delta, axis_offset=1)
    else:
        reg += gamma * total_variation_3d(obj_delta, axis_offset=0)
    return reg
