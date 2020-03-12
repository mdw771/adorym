import dxchange
import numpy as np
from scipy.ndimage import gaussian_filter
import tomopy
import os

save_path = 'cone_256_filled'
dym_x = dim_y = 256

obj_pr = dxchange.read_tiff_stack(os.path.join(save_path, 'paganin_obj/recon_00000.tiff'), range(dim_y), 5)
# obj_pr = gaussian_filter(np.abs(obj_pr), sigma=1, mode='constant')
mask = np.zeros_like(obj_pr)
mask[obj_pr < 0] = 1
for i in range(dim_y):
    mask[i] = mask[i] * tomopy.circ_mask(mask[i:i+1, :, :], 0, ratio=0.00112 * i + 0.5832)
mask[230:] = 0
mask = gaussian_filter(np.abs(mask), sigma=1, mode='constant')
mask[mask > 1e-8] = 1

dxchange.write_tiff_stack(mask, os.path.join(save_path, 'fin_sup_mask/mask'), dtype='float32', overwrite=True)
