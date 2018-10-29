from util import *
import dxchange


# =========================================
path = 'cone_256_filled/new/180'
save_path = os.path.join(path, 'fsc')
ref_path = 'cone_256_filled/phantom/grid_delta.npy'
# =========================================

obj = dxchange.read_tiff(os.path.join(path, 'delta_ds_1.tiff'))
ref = np.load(ref_path)

fourier_shell_correlation(obj, ref, save_path=save_path)
