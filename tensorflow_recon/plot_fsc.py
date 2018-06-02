from util import *
import dxchange


# =========================================
path = 'adhesin/800ev/recon_360_minibatch_20_mskrls_200_iter_auto_alphad_1e-09_alphab_1e-10_rate1e-07_energy_800_size_64_cpu_False'
save_path = os.path.join(path, 'fsc')
ref_path = 'adhesin/phantom/grid_delta.npy'
# =========================================

obj = dxchange.read_tiff(os.path.join(path, 'delta_ds_1.tiff'))
ref = np.load(ref_path)

fourier_shell_correlation(obj, ref, save_path=save_path)