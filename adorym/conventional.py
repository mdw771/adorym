import dxchange

from adorym.util import *
import adorym.wrappers as w


def alt_reconstruction_epie(obj_real, obj_imag, probe_real, probe_imag, probe_pos, probe_pos_correction,
                            prj, device_obj=None, minibatch_size=1, alpha=1., n_epochs=100, **kwargs):
    """
    Reconstruct a 2D object and probe function using ePIE.
    """
    with w.no_grad():
        probe_real = probe_real[0]
        probe_imag = probe_imag[0]
        probe_pos = probe_pos.astype(int)
        energy_ev = kwargs['energy_ev']
        psize_cm = kwargs['psize_cm']
        output_folder = kwargs['output_folder']
        raw_data_type = kwargs['raw_data_type']
        this_obj_size = obj_real.shape
        if len(probe_real) == 2:
            probe_size = probe_real.shape
        else:
            probe_size = probe_imag.shape
        obj_stack = w.stack([obj_real, obj_imag], axis=3)

        # Pad if needed
        obj_stack, pad_arr = pad_object(obj_stack, this_obj_size, probe_pos, probe_size, unknown_type='real_imag')

        i_batch = 0
        subobj_ls = []
        probe_real_ls = []
        probe_imag_ls = []
        for i_epoch in range(n_epochs):
            for j in range(len(probe_pos)):
                print('Batch {}/{}; Epoch {}/{}.'.format(j, len(probe_pos), i_epoch, n_epochs))
                pos = probe_pos[j]
                pos[0] = pos[0] + pad_arr[0, 0]
                pos[1] = pos[1] + pad_arr[1, 0]
                subobj = obj_stack[pos[0]:pos[0] + probe_size[0], pos[1]:pos[1] + probe_size[1], :, :]
                subobj_ls.append(subobj)
                if len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                    this_shift = probe_pos_correction[0, j]
                    probe_real_shifted, probe_imag_shifted = realign_image_fourier(probe_real, probe_imag,
                                                                                   this_shift, axes=(0, 1),
                                                                                   device=device_obj)
                else:
                    probe_real_shifted = probe_real
                    probe_imag_shifted = probe_imag
                probe_real_ls.append(probe_real_shifted)
                probe_imag_ls.append(probe_imag_shifted)
                i_batch += 1
                if i_batch < minibatch_size and i_batch < prj.shape[1]:
                    continue
                else:
                    this_prj_batch = prj[0, j * minibatch_size:j * minibatch_size + i_batch, :, :]
                    this_prj_batch = w.create_variable(this_prj_batch, requires_grad=False, device=device_obj)
                    if raw_data_type == 'intensity':
                        this_prj_batch = w.sqrt(this_prj_batch)
                    subobj_ls = w.stack(subobj_ls)
                    probe_real_ls = w.stack(probe_real_ls)
                    probe_imag_ls = w.stack(probe_imag_ls)
                    c_real, c_imag = subobj_ls[:, :, :, 0, 0], subobj_ls[:, :, :, 0, 1]
                    ex_real_ls, ex_imag_ls = (probe_real_ls * c_real - probe_imag_ls * c_imag, probe_real_ls * c_imag + probe_imag_ls * c_real)
                    dp_real_ls, dp_imag_ls = w.fft2_and_shift(ex_real_ls, ex_imag_ls)
                    mag_replace_factor = this_prj_batch / w.sqrt(dp_real_ls ** 2 + dp_imag_ls ** 2)
                    dp_real_ls = dp_real_ls * mag_replace_factor
                    dp_imag_ls = dp_imag_ls * mag_replace_factor
                    phi_real_ls, phi_imag_ls = w.ishift_and_ifft2(dp_real_ls, dp_imag_ls)
                    d_real_ls = phi_real_ls - ex_real_ls
                    d_imag_ls = phi_imag_ls - ex_imag_ls

                    norm = w.max(probe_real_ls ** 2 + probe_imag_ls ** 2)
                    o_up_real = (probe_real_ls * d_real_ls + probe_imag_ls * d_imag_ls) / norm
                    o_up_imag = (probe_real_ls * d_imag_ls - probe_imag_ls * d_real_ls) / norm
                    o_up = w.stack([o_up_real, o_up_imag], axis=-1)
                    o_up = w.reshape(o_up, [i_batch, probe_size[0], probe_size[1], 1, 2])
                    subobj_ls = subobj_ls + alpha * o_up

                    norm = w.max(subobj_ls[:, :, :, 0, 0] ** 2 + subobj_ls[:, :, :, 0, 1] ** 2)
                    p_up_real = (subobj_ls[:, :, :, 0, 0] * d_real_ls + subobj_ls[:, :, :, 0, 1] * d_imag_ls) / norm
                    p_up_imag = (subobj_ls[:, :, :, 0, 0] * d_imag_ls - subobj_ls[:, :, :, 0, 1] * d_real_ls) / norm
                    p_up = w.stack([p_up_real, p_up_imag], axis=-1)
                    p_up = w.reshape(p_up, [i_batch, probe_size[0], probe_size[1], 1, 2])
                    p_up = w.mean(p_up, axis=0)
                    probe_real = probe_real + alpha * p_up
                    probe_imag = probe_imag + alpha * p_up

                    # Put back.
                    for i, k in enumerate(range(j * minibatch_size, j * minibatch_size + i_batch)):
                        pos = probe_pos[k]
                        pos[0] = pos[0] + pad_arr[0, 0]
                        pos[1] = pos[1] + pad_arr[1, 0]
                        obj_stack[pos[0]:pos[0] + probe_size[0], pos[1]:pos[1] + probe_size[1], :, :] = subobj_ls[i]

                    i_batch = 0
                    subobj_ls = []
                    probe_real_ls = []
                    probe_imag_ls = []

            fname0 = 'obj_mag_{}_{}'.format(i_epoch, i_batch)
            fname1 = 'obj_phase_{}_{}'.format(i_epoch, i_batch)
            obj0, obj1 = w.split_channel(obj_stack)
            obj0 = w.to_numpy(obj0)
            obj1 = w.to_numpy(obj1)
            dxchange.write_tiff(np.sqrt(obj0 ** 2 + obj1 ** 2), os.path.join(output_folder, fname0), dtype='float32',
                                overwrite=True)
            dxchange.write_tiff(np.arctan2(obj1, obj0), os.path.join(output_folder, fname1), dtype='float32',
                                overwrite=True)


def multidistance_ctf_wrapped(this_prj_batch, free_prop_cm, energy_ev, psize_cm, kappa=50, safe_zone_width=0,
                              prj_affine_ls=None, device=None):

    u_free, v_free = gen_freq_mesh(np.array([psize_cm * 1e7] * 3),
                                          [this_prj_batch.shape[i + 1] + 2 * safe_zone_width for i in range(2)])
    u_free = w.create_variable(u_free, requires_grad=False, device=device)
    v_free = w.create_variable(v_free, requires_grad=False, device=device)

    this_prj_batch = w.create_variable(this_prj_batch, requires_grad=False, device=device)
    if prj_affine_ls is not None:
        for i in range(len(prj_affine_ls)):
            this_prj_batch[i] = w.affine_transform(this_prj_batch[i:i + 1], prj_affine_ls[i])
    if safe_zone_width > 0:
        this_prj_batch = w.pad(this_prj_batch, [(0, 0), (safe_zone_width, safe_zone_width),
                                                (safe_zone_width, safe_zone_width)], mode='edge')
    this_prj_batch_ft_r, this_prj_batch_ft_i = w.fft2(this_prj_batch - 1,
                                                      w.zeros_like(this_prj_batch, requires_grad=False, device=device),
                                                      normalize=True)
    dist_nm_ls = free_prop_cm * 1e7
    prj_real_ls = []
    prj_imag_ls = []
    lmbda_nm = 1240. / energy_ev

    for i in range(len(dist_nm_ls)):
        xi = PI * lmbda_nm * dist_nm_ls[i] * (u_free ** 2 + v_free ** 2)
        prj_real_ls.append((w.sin(xi) + 1. / kappa * w.cos(xi)) * this_prj_batch_ft_r[i])
        prj_imag_ls.append((w.sin(xi) + 1. / kappa * w.cos(xi)) * this_prj_batch_ft_i[i])
    this_prj_batch_ft_r = w.sum(w.stack(prj_real_ls), axis=0)
    this_prj_batch_ft_i = w.sum(w.stack(prj_imag_ls), axis=0)

    osc_ls = []
    for i in range(len(dist_nm_ls)):
        xi = PI * lmbda_nm * dist_nm_ls[i] * (u_free ** 2 + v_free ** 2)
        osc_ls.append(2 * (w.sin(xi) + 1. / kappa * w.cos(xi)) ** 2)
    osc = w.sum(w.stack(osc_ls), axis=0) + 1e-10

    a_real = this_prj_batch_ft_r / osc
    a_imag = this_prj_batch_ft_i / osc
    phase, _ = w.ifft2(a_real, a_imag, normalize=True)

    return phase[safe_zone_width:phase.shape[0]-safe_zone_width, safe_zone_width:phase.shape[1]-safe_zone_width]