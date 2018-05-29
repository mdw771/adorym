import dxchange
import numpy as np
from scipy.misc import imsave

# a = dxchange.read_tiff('delta_ds_1.tiff')
a = dxchange.read_tiff_stack('recon/recon_00000.tiff', range(256), 5)
dxchange.write_tiff(np.sum(a, axis=0), 'cone_xray_top_conv')
dxchange.write_tiff(np.sum(a, axis=2), 'cone_xray_left_conv')
dxchange.write_tiff(np.sum(a, axis=1), 'cone_xray_front_conv')