# Adorym: Automatic Differentiation-based Object Reconstruction with DynaMical Scattering

## Installation
Get this repository to your hard drive using 
```
git clone https://github.com/mdw771/adorym
```
and then use PIP to build and install:
```
pip install -e ./adorym
```
After installation, type `python` to open a Python console, and check
the installation status using `import adorym`. If an `ImportError` occurs,
you may need to manually install the missing dependencies. Most
dependencies are available on PIP and can be acquired with
```
pip install <package_name>
```
In order to run Adorym using PyTorch backend with GPU support, please
make sure the right version of PyTorch that matches your CUDA version
is installed.

## Quick start guide
Adorym does 2D/3D ptychography, CDI, holography, and tomography all
using the `reconstruct_ptychography` function in `ptychography.py`.
You can make use of the template scripts in `demos` or `tests` to start
your reconstruction job.

### Running a demo script
Adorym comes with a few datasets and scripts for demonstration and testing,
but the raw data files of some of them are stored elsewhere due to the size limit
on GitHub. If the folder in `demos` or `tests` corresponding to a
certain demo dataset
contains only a text file named `raw_data_url.txt`, please download the
dataset at the URL indicated in the file.

On your workstation, open a terminal in the `demos` folder in Adorym's
root directory, and run the desired script -- say, `multislice_ptycho_256_theta.py`,
which will start a multislice ptychotomography reconstruction job that
solves for the 256x256x256 "cone" object demonstrated in the paper
(see *Publications*), with
```
python multislice_ptycho_256_theta.py
```
To run the script with multiple processes, use
```
mpirun -n <num_procs> python multislice_ptycho_256_theta.py
```

### Dataset format
Adorym reads raw data contained an HDF5 file. The diffraction images should be
stored in the `exchange/data` dataset as a 4D array, with a shape of
`[n_rotation_angles, n_diffraction_spots, image_size_y, image_size_x]`.
In a large part, Adorym is blind to the type of experiment, which means
there no need to explicitly tell it the imaging technique used to generate
the dataset. For imaging data collected from only one angle, `n_rotation_angles = 1`.
For full-field imaging without scanning, `n_diffraction_spots = 1`. For
2D imaging, set the last dimension of the object size to 1 (this will be
introduced further below).

Experimental metadata including beam energy, probe position, and pixel
size, may also be stored in the HDF5, but they can also be provided individually
as arguments to the function `reconstruct_ptychography`. When these arguments
are provided, Adorym uses the arguments rather than reads the metadata from
the HDF5.

The following is the full structure of the HDf5:
```
data.h5
  |___ exchange
  |       |___ data: float, 
  |                  [n_rotation_angles, n_diffraction_spots, image_size_y, image_size_x]
  |
  |___ metadata
          |___ energy_ev: scalar, float. Beam energy in eV
          |___ probe_pos_px: float, [n_diffraction_spots, 2]. 
          |                  Probe positions (y, x) in pixel.
          |___ psize_cm: scalar, float. Sample-plane pixel size in cm.
```

### Parameter settings
The scripts in `demos` and `tests` supply the `reconstruct_ptychography`
with parameters listed as a Python dictionary. You may find the docstrings
of the function helpful, but here lists a collection of the most crucial
parameters:

#### Backend and GPU
- `backend`: Select `'pytorch'` or `'autograd'`. Both can be used as the automatic
differentiation engine, but only the PyTorch backend supports GPU computation.
- `cpu_only`: Set to `False` to enable GPU. This option is ineffective when
`backend` is `pytorch`.

#### I/O
- `fname`: Name of the HDF5 containing raw data. Put only the basename here; any
path predix should go to `save_path`.
- `save_path`: Directory that contains the raw data HDF5. If it is in the same
folder as the execution script, put `'.'`.
- `output_folder`: Name of the folder to place output data. The folder will be
assumed to be under `save_path`, *i.e.*, the actual output directory will be
`<save_path>/<output_folder>`.
- `finite_support_mask_path`: The path to the TIFF file storing the finite
support mask. Default to `None`. In general, this is needed only for single-shot
CDI and holography.
- `save_intermediate`: Bool. Whether to save the intermediate object
  (and probe when `optimize_probe` is `True`) after each minibatch.
- `save_history`: Bool. Useful only if `save_intermediate` is on, If
  `True`, the intermediate output will be saved with a different file
  name characterized by the current epoch and minibatch number.
  Otherwise, the intermediate output will be overwritten.
- `store_checkpoint`: Bool. Whether to save a checkpoint of the
  optimizable variables before each minibatch.
- `use_checkpoint`: Bool. If set to `True`, the program initializes the
  object and/or probe using the checkpoint stored in previous runs. If
  `False` or if checkpoint file is not found, start the reconstruction
  from scratch.

#### Experimental parameters
- `theta_st`: Starting rotation angle in radian. Default to 0.
- `theta_end`: End rotation angle in radian. For single angle data, set this
               the same as `theta_st`.
- `theta_downsample`: Int. By how many times the raw data should be downsampled
                      in rotation angles. Default to `None`.
- `probe_pos`: Float, `[n_diffraction_spots, 2]`. Probe positions in a
  scanning-type experiment in pixel in the object frame (*i.e.*,
  real-unit probe positions divided by sample plane pixel size). Default
  to `None`. If `None`, the program will attempt to get the value from
  HDF5. The positions will be interpreted as the **top-left corner of
  the probe array in object frame**. For single-shot experiments, set
  `probe_pos` to `[[0, 0]]`.
- `energy_ev`: Float. X-ray beam energy in eV. Default to `None`. If
  `None`, the program will attempt to get the value from HDF5.
- `psize_cm`: Float. Pixel size at sample plane in cm. Default to
  `None`. If `None`, the program will attempt to get the value from
  HDF5.
- `free_prop_cm`: Float. The distance between sample and detector in cm.
  For far-field imaging, set it to `None` or `'inf'`, so that the
  programs uses Fraunhofer approximation. **For near-field imaging, this
  value is assumed to be the propagation distance in a plane-wave
  illuminated experiment; if the illumination is a spherical wave
  generated by a point source, use the effective distance given by
  Fresnel scaling theorem: `z_eff = z1 * z2 / (z1 + z2)`**.
- `raw_data_type`: Choose from `'intensity'` or `'magnitude'`. This
  informs the optimizer the type of raw data contained in the HDF5.

#### Reconstruction parameters
- `obj_size`: Int, `[L_y, L_x, L_z]`. The size of the object function
  (*i.e.*, the unknowns) in pixels. `L_y` is the size in the vertical
  direction, while `L_x` and `L_z` refer to sizes on the horizontal
  plane. For 2D reconstruction, set `L_z` to 1. For 3D reconstruction,
  it is strongly recommended to keep `L_x == L_z`. For doing sparsely
  spaced multislice tomography (*i.e.*, when the number of slices along
  beam axis is much less than the number of lateral pixels), the best
  practice is to set `binning` to a larger value, instead of using a
  small `L_z`.
- `unknown_type`: Choose from `delta_beta` and `real_imag`. If set to
  `delta_beta`, the program treats the unknowns as the delta and beta
  parts in the complex refractive indices of the object, `n =
  1-delta-i*beta`. In this case, modulation to the wavefield by each
  slice of the object will be done as `wavefield * exp(-i*k*n*z)`. If
  set to `real_imag`, the unknowns are treated as the real and imaginary
  part of a multiplicative object function, where the modulation is done
  as `wavefield * (obj_real + i * obj_imag)`. Using `delta_beta` can
  help overcome mild phase wrapping, while using `real_imag` generally
  leads to better numerical robustness.
- `n_epochs`: Int. Number of epochs to run. An epoch refers to a cycle
  during which all diffraction data are processed. Set it to `'auto'` to
  automatically stops the reconstruction when the reduction rate of loss
  falls below `crit_conv_rate`. **This option is not recommended
  especially for noisy data due to the possibility of fake positives.**
  The best practice so far is to set `n_epochs` to a sufficiently large
  value and observe the loss curve and reconstruction output until
  satisfactory results are obtained.
- `crit_conv_rate`: Float. If the reduction rate of loss at the current
  epoch in regards to the previous one is below this value, convergence
  is assumed to be reached and the reconstruction process stops.
- `max_epochs`: Int. When `n_epochs` is set to `'auto'`, the program
  will stop regardless of the loss reduction rate once this number of
  epochs have been run.
- `minibatch_size`: Int. The number of diffraction spots to be processed
  at a time. When multi-processing, this is the number of diffraction
  spots processed by each rank.
- `alpha_d`: Float. Weight applied to l1-norm of the delta (or real)
  part of the object function. The full loss function is in the form of
  `L = D(f(x), y0) + alpha_d * |x_d|_1 + alpha_b * |x_b|_1 + gamma *
  TV(x)`.
- `alpha_b`: Float. Weight applied to l1-norm of the beta (or imaginary)
  part of the object function.
- `gamma`: Float. Weight applied to total variation of the object
  function.
- `reweighted_l1`: Bool. If `True` and `alpha_d != 0`, the program uses
  reweighted l1-norm to regularize the object (see Candès, E. J., Wakin,
  M. B. & Boyd, S. P. Enhancing Sparsity by Reweighted ℓ1 Minimization.
  *Journal of Fourier Analysis and Applications* **14**, (2008). )
- `object_type`: Choose from `'normal'`, `'phase_only'`, or
  `'absorption_only'`. If `'absorption_only'`, the delta part of the
  phase of the object will be forced to be 0 after each update. Vice
  versa for `'phase_only'`.
- `non_negativity`: Bool. Whether to enforce non-negative constraint.
  Useful only when `unknown_type` is `delta_beta`.
- `shrink_cycle`: Int. For every how many minibatches should the finite
  support mask be shrink-wrapped. Useful only when
  `finite_support_mask_path` is not None.
- `'shrink_threshold'`: Float. Shreshold for shrink-wrapping. Useful only when
  `finite_support_mask_path` is not None.
- `initial_guess`: List of Arrays. Default to None. The initial guess of
  the object function in the form of `[obj_delta, obj_beta]` when
  `unknown_type` is `delta_beta`, or `[obj_mag, obj_phase]` when
  `unknown_type` is `real_imag`. The arrays must have the same size as
  specified by `obj_size`.
- `random_guess_means_sigmas`: List of Floats. When `initial_guess` is
  `None`, the object function will be initialized usin Gaussian randoms.
  This argument provides the Gaussian parameters in the format of
  `(mean_delta, mean_beta, sigma_delta, sigma_beta)` or `(mean_mag,
  mean_phase, sigma_mag, sigma_phase)`, depending on the setting of
  `unknwon_type`.
- `update_scheme`: Choose from `'immediate'` or `'per angle'`. If
  `'immediate'`, the object function is updated immedaitely after each
  minibatch is done. If `'per angle'`, updated is performed only after
  all diffraction patterns from the current rotation angle are
  processed. If `shared_file_object` is on, the `'per angle'` mode is
  used regardless of this setting.
- `randomize_probe_pos`: Bool. Whether to randomize diffraction spots on
  each viewing angle when there are more than 1 of them. Recommended to
  be `True` for 2D ptychography.

#### Forward model
- `binning`: Int. The number of axial slices to be binned (*i.e.*, to be
  treated as line integrals) during multislice propagation.
- `pure_projection`: Bool. Set to `True` to model the propagation
  through the entire object as a simple line projection, not using
  multislice at all.
- `two_d_mode`: Bool. If the HDF5 dataset contains multiple viewing
  angles (*i.e.*, the length of the first dimension is larger than 1),
  setting `two_d_mode` to `True` will let the program to treat it as a
  single-angle dataset, with the only angle being the first one. Set to
  `True` automatically if the last dimension of `obj_size` is 1.
- `probe_type`: Choose from `'gaussian'`, `'plane'`, `'ifft'`,
  `'aperture_defocus'`, and `'supplied'`. The method of initializing the
  probe function. Some options requires additional inputs from user:
  - `'gaussian'`: Supply `probe_mag_sigma`, `probe_phase_sigma`, and
    `probe_phase_max`. The Gaussian spreads, or the `*sigma` values, are
    in pixel. Magnitude max is 1 by default.
  - `'aperture_defocus'`: Supply `aperture_radius`, `beamstop_radius`,
    and `probe_defocus_cm`. All radii are in pixels (on the object
    frame). A circular aperture (if `beamstop_radius == 0`) or a ring
    aperture (if `0 < beamstop_radius < aperture_radius`) is generated
    and then Fresnel defocused to created the initial probe.
  - `'supplied'`: Supply `probe_initial` as a List of Arrays:
    `[probe_mag, probe_phase]`.
- `rescale_probe_intensity`: Bool. Scale the probe function so that its
  integrated power spectrum (related to the total number of photons)
  matches that of the raw data.
- `loss_function_type`: Choose from `'lsq'` or `'poisson'`. Whether to
  use a least square term or a Poisson maximum likelihood term to
  measure the mismatch of predicted intensity.

#### Optimizers
- `optimizer`: Choosen from `'adam'` or `'gd'` (steepest gradient
  descent). Optimizer type for updating the object function.
- `learning_rate`: Float. Learning rate, or step size of the chosen
  optimizer for the object function.
- `optimize_probe`: Bool. Whether to optimize the probe function.
- `probe_learning_rate`: Float.
- `optimize_probe_defocuing`: Bool. Whether to optimize the defocusing
  distance of the probe.
- `probe_defocusing_learning_rate`: Float.
- `optimize_probe_pos_offset`: Bool. Whether to optimize the offset to
  probe positions. This is intended to correct for the x-y drifting of
  the sample stage at different angles. When turned on, the program
  creates an array with shape `[n_rotation_angles, 2]`. When processing
  data from a certain viewing angle, the positions of all diffraction
  spots are shifted by the value corresponding to that angle. The offset
  array is optimized by the optimizer along with the object function.
- `optimize_all_probe_pos`: Bool. Whether to optimize the probe
  positions at all angles. When turned on, the optimizer tries to
  optimize an array with shape `[n_rotation_angles, n_diffraction_spots,
  2]`, which stores the correction values applied to each probe position
  at all viewing angles. Not recommended for ptychotomography with many
  viewing angles as it significantly increases the unknwon space to be
  searched, making the problem less well constrained.
- `all_probe_pos_learning_rate`: Float.

#### Low-memory mode
- `shared_file_object`: Switch of hard-drive mediated low-memory node. When set
to `True`, the 3D object function will be stored as a parallel-HDF5 on hard drive
instead of in the RAM or GPU memory. The major advantage of using this working
mode is that when reconstructing 2D/3D ptychography data with GPU,
Adorym will only send GPU a small part of the object array that is relevant to
the minibatch being processed, which minimizes memory usage. The low-memory mode
is also useful when the machine's RAM is very limited, or when the object being
reconstructed is very large. I/O overhead might be observed when using a large
number of processes on a non-parallel file system.
**Note: using the low-memory node requires H5Py built against MPIO-enabled HDF5.**

### Output
During runtime, Adorym may create a folder named
`arrsize_?_?_?_ntheta_?` in the current working directory, which saves
the precalculated coordinates for rotation transformation. Other than
that, all outputs will be written in `<save_path>/<output_folder>`,
which is organized as shown in the chart below:
```
output_folder
     |___ convergence
     |         |___ loss_rank_0.txt // Record of the loss value after 
     |         |___ loss_rank_1.txt // each update coming from each process.
     |         |___ ...
     |___ intermediate
     |         |___ object
     |         |       |___ obj_mag(delta)_0_0.tiff
     |         |       |___ obj_phase(beta)_0_0.tiff
     |         |       |___ ...
     |         |___ probe
     |         |       |___ probe_mag_0_0.tiff
     |         |       |___ probe_phase_0_0.tiff
     |         |       |___ ...
     |         |___ probe_pos (if optimize_all_probe_pos is True)
     |                 |___ probe_pos_correction_0_0_0.txt
     |                 |___ ...
     |___ obj_delta_ds_1.tiff (or obj_mag_ds_1.tiff)
     |___ obj_beta_ds_1.tiff (or obj_phase_ds_1.tiff)
     |___ probe_mag_ds_1.tiff
     |___ probe_phase_ds_1.tiff
     |___ summary.txt // Summary of parameter settings.
     |___ checkpoint.txt // Exists if store_checkpoint is True.
     |___ obj_checkpoint.npy // Exists if store_checkpoint is True.
     |___ opt_params_checkpoint.npy // Exists if store_checkpoint is True and optimizer has parameters.
```
By default, all image outputs are in 32-bit floating points which can be
opened and viewed with ImageJ.

## Publications
The early version of Adorym, which was used to demonstrate 3D reconstruction of continuous object beyond the depth of focus, is published as

Du, M., Nashed, Y. S. G., Kandel, S., Gürsoy, D. & Jacobsen, C. Three
dimensions, two microscopes, one code: Automatic differentiation for
x-ray nanotomography beyond the depth of focus limit. *Sci Adv* **6**,
eaay3700 (2020).
