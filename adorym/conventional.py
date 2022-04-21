import dxchange

from adorym.util import *
import adorym.wrappers as w


def alt_reconstruction_epie(obj, probe, probe_pos, probe_pos_correction,
                            prj, device_obj=None, minibatch_size=1, alpha=1., n_epochs=100, **kwargs):
    """
    Reconstruct a 2D object and probe function using ePIE.
    """
    with w.no_grad():
        probe_pos = probe_pos.astype(int)
        energy_ev = kwargs['energy_ev']
        psize_cm = kwargs['psize_cm']
        output_folder = kwargs['output_folder']
        raw_data_type = kwargs['raw_data_type']
        this_obj_size = obj.shape
        probe_size = probe.shape

        # Pad if needed
        obj, pad_arr = pad_object(obj, this_obj_size, probe_pos, probe_size, unknown_type='real_imag')

        obj = .0001 + 1j * obj.imag
        i_batch = 0
        subobj_ls = []
        probe_ls = []
        pos_ls = []
        for i_epoch in range(n_epochs):
            print(f'Epoch {i_epoch}/{n_epochs}')
            for j, pos in enumerate(probe_pos):
                # print(f'Batch {j}/{len(probe_pos)}; Epoch {i_epoch}/{n_epochs}.')
                pos = pos.copy()
                pos[0] = pos[0] + pad_arr[0, 0]
                pos[1] = pos[1] + pad_arr[1, 0]
                pos_ls.append(pos)
                subobj = obj[pos[0]:pos[0] + probe_size[0], pos[1]:pos[1] + probe_size[1], 0]
                if len(subobj_ls) > 0:
                    assert subobj_ls[i_batch-1].shape == subobj.shape
                subobj_ls.append(subobj)
                if len(w.nonzero(probe_pos_correction > 1e-3)) > 0:
                    this_shift = probe_pos_correction[0, j]
                    probe_real_shifted, probe_imag_shifted = realign_image_fourier(np.real(probe), np.imag(probe),
                                                                                   this_shift, axes=(0, 1),
                                                                                   device=device_obj)
                    probe_shifted = probe_real_shifted + 1j * probe_imag_shifted
                else:
                    probe_shifted = probe
                probe_ls.append(probe_shifted)
                i_batch += 1
                if i_batch < minibatch_size and i_batch < prj.shape[1]:
                    continue
                else:
                    this_prj_batch = prj[0, (j-i_batch+1):j+1, :, :]
                    this_prj_batch = w.create_variable(this_prj_batch, requires_grad=False, device=device_obj, dtype='complex64')
                    if raw_data_type == 'intensity':
                        this_prj_batch = w.sqrt(this_prj_batch)
                    subobj_ls = w.stack(subobj_ls)
                    probe_ls = w.stack(probe_ls)

                    ex_ls = probe_ls * subobj_ls
                    ex_ls[np.abs(ex_ls)<1e-10] = 0
                    dp_ls = w.fft2_and_shift_complex(ex_ls)
                    mag_replace_factor = this_prj_batch / (np.abs(dp_ls) + np.finfo(float).eps)
                    dp_ls = dp_ls * mag_replace_factor
                    phi_ls = w.ishift_and_ifft2_complex(dp_ls)

                    d_ls = phi_ls - ex_ls

                    norm_probe = w.max(w.abs(probe_ls)**2)
                    norm_subobj = w.max(w.abs(subobj_ls)**2)
                    # subobj_ls = subobj_ls + alpha * np.conj(probe_ls) * d_ls / (norm_probe + np.finfo('float').eps)
                    subobj_ls_diff =  alpha * np.conj(probe_ls) * d_ls / (norm_probe + np.finfo('float').eps) 
                    subobj_ls_diff *= np.abs(subobj_ls_diff)
                    # probe update
                    probe_ls = probe_ls + alpha/4 * np.conj(subobj_ls) * d_ls  / (norm_subobj + np.finfo('float').eps)



                    # Put back.
                    # TODO this fails because I think successive spots overwrite eachother
                    for i, pos_batch in enumerate(pos_ls):
                        obj[pos_batch[0]:pos_batch[0] + probe_size[0], pos_batch[1]:pos_batch[1] + probe_size[1], 0] += subobj_ls_diff[i]

                    i_batch = 0
                    subobj_ls = []
                    probe_ls = []
                    pos_ls = []


            fname0 = 'obj_mag_{}_{}'.format(i_epoch, i_batch)
            fname1 = 'obj_phase_{}_{}'.format(i_epoch, i_batch)
            # obj0, obj1 = w.split_channel(obj)
            # obj0 = w.to_numpy(obj0)
            # obj1 = w.to_numpy(obj1)
            dxchange.write_tiff(np.abs(obj), os.path.join(output_folder, fname0), dtype='float32',
                                overwrite=True)
            dxchange.write_tiff(np.angle(obj), os.path.join(output_folder, fname1), dtype='float32',
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