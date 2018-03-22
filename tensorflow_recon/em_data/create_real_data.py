import dxchange
import numpy as np
from scipy.ndimage import gaussian_filter
import xdesign


original_stack = dxchange.read_tiff_stack('adhesin/emd_0000.tif', range(64))
# new_stack_delta = np.zeros_like(original_stack, dtype='float32')
# new_stack_beta = np.zeros_like(original_stack, dtype='float32')

original_stack = original_stack.astype('float32')

original_stack = np.roll(original_stack, -10, 0)

alpha = (original_stack != 83)
alpha = alpha.astype('float')
alpha = gaussian_filter(alpha, 5)
original_stack = original_stack * alpha
dxchange.write_tiff(original_stack, 'delta', dtype='float32', overwrite=True)

new_stack_delta = original_stack * 1e-7
new_stack_beta = original_stack * 0.85e-9

dxchange.write_tiff_stack(new_stack_delta, 'delta/delta', overwrite=True, dtype='float32')
dxchange.write_tiff_stack(new_stack_beta, 'beta/beta', overwrite=True, dtype='float32')

np.save('adhesin_delta', new_stack_delta)
np.save('adhesin_beta', new_stack_beta)

#
# mat_base = xdesign.XraylibMaterial('H48.6C32.9N8.9O8.9S0.6', 1.35)
# delta1 = mat_base.delta(energy=5)
# beta1 = mat_base.beta(energy=5)
# print(delta1, beta1)
# delta2 = delta1 + 0.5e-5
# beta2 = beta1 + 0.9e-8
#
# xx, yy, zz = np.meshgrid(range(64), range(64), range(64))
#
# new_stack_delta[original_stack > 160] = delta2
# new_stack_beta[original_stack > 160] = beta2
#
# new_stack_delta[(original_stack > 84) * (original_stack < 160) * (yy > 42)] = delta1
# new_stack_beta[(original_stack > 84) * (original_stack < 160) * (yy > 42)] = beta1
#

#
# new_stack_delta = gaussian_filter(new_stack_delta, .6)
# new_stack_beta = gaussian_filter(new_stack_beta, .6)
#
# dxchange.write_tiff_stack(new_stack_delta, 'delta/delta', overwrite=True, dtype='float32')
# dxchange.write_tiff_stack(new_stack_beta, 'beta/beta', overwrite=True, dtype='float32')
#
# np.save('adhesin_delta', new_stack_delta)
# np.save('adhesin_beta', new_stack_beta)
