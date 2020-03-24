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

#### I/O paths
- `fname`: Name of the HDF5 containing raw data. Put only the basename here; any
path predix should go to `save_folder`.
- `save_folder`: Directory that contains the raw data HDF5. If it is in the same
folder as the execution script, put `'.'`.
- `output_folder`: Name of the folder to place output data. The folder will be
assumed to be under `save_folder`, *i.e.*, the actual output directory will be
`<save_folder>/<output_folder>`.
- `finite_support_mask_path`: The path to the TIFF file storing the finite
support mask. Default to `None`. In general, this is needed only for single-shot
CDI and holography.

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

## Publications
The early version of Adorym, which was used to demonstrate 3D reconstruction of continuous object beyond the depth of focus, is published as

Du, M., Nashed, Y. S. G., Kandel, S., Gursoy, D. & Jacobsen, C. Three dimensions, two microscopes, one code: automatic differentiation for x-ray nanotomography beyond the depth of focus limit. *arXiv.org* **eess.IV**, arXiv:1905.10433 (2019).
  
