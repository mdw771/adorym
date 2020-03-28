import argparse
import numpy as np
import dxchange
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='None')
parser.add_argument('--output', default='data.h5')
parser.add_argument('--free_prop_cm', default='175.')
parser.add_argument('--detector_psize_cm', default='75e-4')
args = parser.parse_args()

fname = args.filename
fname_new = args.output
free_prop_cm = float(args.free_prop_cm)
detector_psize_cm = float(args.detector_psize_cm)

f_old = h5py.File(fname, 'r')
f_new = h5py.File(fname_new, 'w')

dset_old = f_old['dp']
n_pos = dset_old.shape[0]
probe_size = dset_old.shape[1:]

grp_new = f_new.create_group('exchange')
dset_new = grp_new.create_dataset('data', shape=[1, n_pos, probe_size[0], probe_size[1]], dtype=dset_old.dtype)

print('Old dataset shape: ', dset_old.shape)
print('New dataset shape: ', dset_new.shape)
print('Data type: ', dset_old.dtype)
dp = dset_old[...]
dset_new[0, :, :, :] = dp

dxchange.write_tiff(dp, 'diffraction_dat.tiff', dtype='float32', overwrite=True)

grp_meta_new = f_new.create_group('metadata')

# Write metadata
f_meta = open('parameters.txt', 'w')

# Wavelength
lmbda_nm = f_old['lambda'][0] * 1e9
grp_meta_new.create_dataset('energy_ev', data=1240. / lmbda_nm)
f_meta.write('wavelength_nm:     {}\n'.format(lmbda_nm))
f_meta.write('energy_ev:        {}\n'.format(1240. / lmbda_nm))

# Sample to detector distance
f_meta.write('free_prop_cm:      {}\n'.format(free_prop_cm))
f_meta.write('detector_psize_cm: {}\n'.format(detector_psize_cm))
psize_cm = f_old['dx'][0] * 1e2
f_meta.write('psize_cm:          {}\n'.format(psize_cm))
grp_meta_new.create_dataset('psize_cm', data=psize_cm)
f_meta.close()

# Probe position
probe_pos_x = f_old['ppX'][...]
probe_pos_y = f_old['ppY'][...]
probe_pos = np.stack([probe_pos_y, probe_pos_x], axis=1)
probe_pos = probe_pos * 1e2 / psize_cm
probe_pos -= np.min(probe_pos, axis=0)
probe_pos += 50
grp_meta_new.create_dataset('probe_pos_px', data=probe_pos)
np.savetxt('probe_pos_px.txt', probe_pos, fmt='%f')
    
f_old.close()
f_new.close()
