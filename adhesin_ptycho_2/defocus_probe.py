import numpy as np
import xommons
import dxchange


psize_cm = 0.67e-7
dist_cm = 10e-7
energy_kev = 0.8

probe_mag = dxchange.read_tiff('probe_mag.tiff').astype('float64')
probe_phase = dxchange.read_tiff('probe_phase.tiff').astype('float64')

a = probe_mag * np.exp(1j * probe_phase)
a = np.reshape(a, [1, *a.shape])

a = xommons.fresnel_propagate(a, energy_kev * 1e3, [psize_cm] * 2, dist_cm, fresnel_approx=True)
a = np.squeeze(a)

dxchange.write_tiff(np.abs(a), 'probe_mag_defocus_10nm.tiff', dtype='float64', overwrite=True)
dxchange.write_tiff(np.angle(a), 'probe_phase_defocus_10nm.tiff', dtype='float64', overwrite=True)
dxchange.write_tiff(np.abs(a), 'probe_mag_defocus_10nm_f32.tiff', dtype='float32', overwrite=True)
dxchange.write_tiff(np.angle(a), 'probe_phase_defocus_10nm_f32.tiff', dtype='float32', overwrite=True)
# dxchange.write_tiff(a.real, 'probe_real_defocus_10nm.tiff', dtype='float32', overwrite=True)
# dxchange.write_tiff(a.imag, 'probe_imag_defocus_10nm.tiff', dtype='float32', overwrite=True)
#
a = np.reshape(a, [1, *a.shape])
a = xommons.fresnel_propagate(a, energy_kev * 1e3, [psize_cm] * 2, -dist_cm, fresnel_approx=True)
a = np.squeeze(a)

# dxchange.write_tiff(np.abs(a), 'probe_mag_defocus_10nm_back.tiff', dtype='float32', overwrite=True)
# dxchange.write_tiff(np.angle(a), 'probe_phase_defocus_10nm_back.tiff', dtype='float32', overwrite=True)
dxchange.write_tiff(np.abs(a), 'probe_mag_defocus_10nm_back_f32.tiff', dtype='float32', overwrite=True)
dxchange.write_tiff(np.angle(a), 'probe_phase_defocus_10nm_back_f32.tiff', dtype='float32', overwrite=True)