from util import *
import dxchange


# =========================================
path = 'cone_256_filled/recon_360_minibatch_10_mskrls_10_shrink_1_iter_auto_alphad_1.5e-06_alphab_1.5000000000000002e-07_gamma_5e-07_rate_1e-07_energy_5000_size_256_ntheta_500_prop_0.0001_ms_3_cpu_True'
save_path = os.path.join(path, 'fsc')
ref_path = 'cone_256_filled/phantom/grid_delta.npy'
# =========================================

obj = dxchange.read_tiff(os.path.join(path, 'delta_ds_1.tiff'))
ref = np.load(ref_path)

fourier_shell_correlation(obj, ref, save_path=save_path)