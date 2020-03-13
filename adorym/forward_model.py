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
                reg += tv(obj_delta, obj_beta,
                          self.regularizer_dict[name]['gamma'],
                          self.shared_file_object, device=self.device)
        return reg


class PtychographyModel(ForwardModel):

    def __init__(self, loss_function_type='lsq', shared_file_object=False, device=None, common_vars_dict=None, raw_data_type='magnitude'):
        super(PtychographyModel, self).__init__(loss_function_type, shared_file_object, device, common_vars_dict, raw_data_type)
        self.argument_ls = ['obj_delta', 'obj_beta', 'probe_real', 'probe_imag', 'probe_defocus_mm',
                            'probe_pos_offset', 'this_i_theta', 'this_pos_batch', 'this_prj_batch',
                            'probe_pos_correction', 'this_ind_batch']

    def predict(self, obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                probe_pos_offset, this_i_theta, this_pos_batch, this_prj_batch,
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
            probe_pos_batch_ls = []
            ex_real_ls = []
            ex_imag_ls = []
            i_dp = 0
            while i_dp < minibatch_size:
                probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
                i_dp += n_dp_batch

            # Pad if needed
            obj_rot, pad_arr = pad_object(obj_rot, this_obj_size, this_pos_batch, probe_size)

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
                if optimize_all_probe_pos:
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
                    fresnel_approx=fresnel_approx, pure_projection=pure_projection, device=device_obj)
                ex_real_ls.append(ex_real)
                ex_imag_ls.append(ex_imag)
            del subobj_ls, probe_real_ls, probe_imag_ls
        else:
            probe_pos_batch_ls = []
            ex_real_ls = []
            ex_imag_ls = []
            i_dp = 0
            while i_dp < minibatch_size:
                probe_pos_batch_ls.append(this_pos_batch[i_dp:min([i_dp + n_dp_batch, minibatch_size])])
                i_dp += n_dp_batch

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
                                                              device=device_obj)
                ex_real_ls.append(ex_real)
                ex_imag_ls.append(ex_imag)
                pos_ind += len(pos_batch)
        ex_real_ls = w.concatenate(ex_real_ls, 0)
        ex_imag_ls = w.concatenate(ex_imag_ls, 0)
        return ex_real_ls, ex_imag_ls

    def get_loss_function(self):
        def calculate_loss(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, this_prj_batch,
                           probe_pos_correction, this_ind_batch):
            ex_real_ls, ex_imag_ls = self.predict(obj_delta, obj_beta, probe_real, probe_imag, probe_defocus_mm,
                           probe_pos_offset, this_i_theta, this_pos_batch, this_prj_batch,
                           probe_pos_correction, this_ind_batch)
            this_prj_batch = w.create_variable(abs(this_prj_batch), requires_grad=False, device=self.device)
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
