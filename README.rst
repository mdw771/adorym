Adorym: Automatic Differentiation-based Object Reconstruction with DynaMical Scattering
=======================================================================================

- This repository contains the stable version of Adorym. For the most updated version, see `https://github.com/mdw771/adorym_dev <https://github.com/mdw771/adorym_dev>`_.
- For a more comprehensive (and more detailed) documentation, please visit the official `documentation <https://adorym.readthedocs.io>`_.

Table of contents
-----------------

#. `Installation <#installation>`__
#. `Quick start guide <#quick-start-guide>`__
#. `Running a demo script <#running-a-demo-script>`__
#. `Running your own jobs <#running-your-own-jobs>`__
#. `Data format <#dataset-format>`__
#. `Customization <#customization>`__
#. `Publications <#publications>`__

Installation
------------

Get this repository to your hard drive using

::

    git clone https://github.com/mdw771/adorym

and then use PIP to build and install:

::

    pip install ./adorym

If you will modify internal functions of Adorym, *e.g.*, add new
forward
models or refinable parameters, it is suggested to use the ``-e`` flag
to
enable editable mode so that you don't need to rebuild Adorym each
time
you make changes to the source code:

::

    pip install -e ./adorym

After installation, type ``python`` to open a Python console, and
check
the installation status using ``import adorym``. If an ``ImportError``
occurs,
you may need to manually install the missing dependencies. Most
dependencies are available on PIP and can be acquired with

::

    pip install <package_name>

or through Conda if you use the Anaconda or Miniconda distribution of
Python:

::

    conda install <package_name>

In order to run Adorym using PyTorch backend with GPU support, please
make sure the right version of PyTorch that matches your CUDA version
is installed. The latter can be checked through ``nvidia-smi``.

Quick start guide
-----------------

Adorym does 2D/3D ptychography, CDI, holography, and tomography all
using the ``reconstruct_ptychography`` function in
``ptychography.py``.
You can make use of the template scripts in ``demos`` or ``tests`` to
start
your reconstruction job.

Running a demo script
~~~~~~~~~~~~~~~~~~~~~

Adorym comes with a few datasets and scripts for demonstration and
testing,
but the raw data files of some of them are stored elsewhere due to the
size limit
on GitHub. If the folder in ``demos`` or ``tests`` corresponding to a
certain demo dataset
contains only a text file named ``raw_data_url.txt``, please download
the
dataset at the URL indicated in the file.

On your workstation, open a terminal in the ``demos`` folder in
Adorym's
root directory, and run the desired script -- say,
``multislice_ptycho_256_theta.py``,
which will start a multislice ptychotomography reconstruction job that
solves for the 256x256x256 "cone" object demonstrated in the paper
(see *Publications*), with

::

    python multislice_ptycho_256_theta.py

To run the script with multiple processes, use

::

    mpirun -n <num_procs> python multislice_ptycho_256_theta.py

Running your own jobs
~~~~~~~~~~~~~~~~~~~~~

You can use the scripts in ``demos`` and ``tests`` as templates to create the
scripts for your own jobs. While the major API is the function ``reconstruct_ptychography``
itself, you may also explicitly declare optimizers to be used for the object, the
probe, and any other refinable parameters. Below is an example script used
for 2D fly-scan ptychography reconstruction with probe position refinement:

.. code-block::

    import adorym
    from adorym.ptychography import reconstruct_ptychography

    output_folder = "recon"
    distribution_mode = None
    optimizer_obj = adorym.AdamOptimizer("obj", output_folder=output_folder,
                                         distribution_mode=distribution_mode,
                                         options_dict={"step_size": 1e-3})
    optimizer_probe = adorym.AdamOptimizer("probe", output_folder=output_folder,
                                           distribution_mode=distribution_mode,
                                           options_dict={"step_size": 1e-3, "eps": 1e-7})
    optimizer_all_probe_pos = adorym.AdamOptimizer("probe_pos_correction",
                                                   output_folder=output_folder,
                                                   distribution_mode=distribution_mode,
                                                   options_dict={"step_size": 1e-2})

    params_ptych = {"fname": "data.h5",
                    "theta_st": 0,
                    "theta_end": 0,
                    "n_epochs": 1000,
                    "obj_size": (618, 606, 1),
                    "two_d_mode": True,
                    "energy_ev": 8801.121930115722,
                    "psize_cm": 1.32789376566526e-06,
                    "minibatch_size": 35,
                    "output_folder": output_folder,
                    "cpu_only": False,
                    "save_path": ".",
                    "initial_guess": None,
                    "random_guess_means_sigmas": (1., 0., 0.001, 0.002),
                    "probe_type": "aperture_defocus",
                    "forward_model": adorym.PtychographyModel,
                    "n_probe_modes": 5,
                    "aperture_radius": 10,
                    "beamstop_radius": 5,
                    "probe_defocus_cm": 0.0069,
                    "rescale_probe_intensity": True,
                    "free_prop_cm": "inf",
                    "backend": "pytorch",
                    "raw_data_type": "intensity",
                    "optimizer": optimizer_obj,
                    "optimize_probe": True,
                    "optimizer_probe": optimizer_probe,
                    "optimize_all_probe_pos": True,
                    "optimizer_all_probe_pos": optimizer_all_probe_pos,
                    "save_history": True,
                    "unknown_type": "real_imag",
                    "loss_function_type": "lsq",
                    }

    reconstruct_ptychography(**params_ptych)

To learn the settings of the ``reconstruct_ptychography`` function, please visit
the `documentation <https://adorym.readthedocs.io>`_.

Dataset format
~~~~~~~~~~~~~~

Adorym reads raw data contained an HDF5 file. The diffraction images
should be
stored in the ``exchange/data`` dataset as a 4D array, with a shape of
``[n_rotation_angles, n_diffraction_spots, image_size_y, image_size_x]``.
In a large part, Adorym is blind to the type of experiment, which
means
there no need to explicitly tell it the imaging technique used to
generate
the dataset. For imaging data collected from only one angle,
``n_rotation_angles = 1``.
For full-field imaging without scanning, ``n_diffraction_spots = 1``.
For
2D imaging, set the last dimension of the object size to 1 (this will
be
introduced further below).

Experimental metadata including beam energy, probe position, and pixel
size, may also be stored in the HDF5, but they can also be provided
individually
as arguments to the function ``reconstruct_ptychography``. When these
arguments
are provided, Adorym uses the arguments rather than reads the metadata
from
the HDF5.

The following is the full structure of the HDf5:

::

    data.h5
      |___ exchange
      |       |___ data: float, 4D array
      |                  [n_rotation_angles, n_diffraction_spots, image_size_y, image_size_x]
      |
      |___ metadata
              |___ energy_ev: scalar, float. Beam energy in eV
              |___ probe_pos_px: float, [n_diffraction_spots, 2]. 
              |                  Probe positions (y, x) in pixel.
              |___ psize_cm: scalar, float. Sample-plane pixel size in cm.
              |___ free_prop_cm: (optional) scalar or array 
              |                  Distance between sample exiting plane and detector.
              |                  For far-field propagation, do not include this item. 
              |___ slice_pos_cm: (optional) float, 1D array
                                 Position of each slice in sparse multislice ptychography. Starts from 0.

Customization
-------------

Adding your own forward model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create additional forward models beyond the existing ones. To
begin with, in ``adorym/forward_model.py``,
create a class inheriting ``ForwardModel`` (*i.e.*,
``class MyNovelModel(ForwardModel)``). Each forward model class
should contain 4 essential methods: ``predict``, ``get_data``,
``loss``, and ``get_loss_function``. ``predict`` maps input variables
to predicted quantities (usually the real-numbered magnitude of the
detected wavefield). ``get_data`` reads from
the HDF5 file the raw data corresponding to the minibatch currently
being processed. ``loss`` is the last-layer
loss node that computes the (regularized)
loss values from the predicted data and the experimental measurement
for the current minibatch. ``get_loss_function``
concatenates the above methods and return the end-to-end loss
function. If your ``predict`` returns the real-numbered
magnitude of the detected wavefield, you can use ``loss`` inherented
from the parent class, although you still need to
make a copy of ``get_loss_function`` and explicitly change its
arguments to match those of ``predict`` (do not use
implicit argument tuples or dictionaries like ``*args`` and
``**kwargs``, as that won't work with Autograd!). If your ``predict``
returns something else, you may also need to override ``loss``. Also
make sure your new forward model class contains
a ``self.argument_ls`` attribute, which should be a list of argument
strings that exactly matches the signature of ``predict``.

To use your forward model, pass your forward model class to the
``forward_model`` argument of ``reconstruct_ptychography``.
For example, in the script that you execute with Python, do the
following:

::

    import adorym
    from adorym.ptychography import reconstruct_ptychography

    params = {'fname': 'data.h5', 
              ...
              'forward_model': adorym.MyNovelModel,
              ...}

Adding refinable parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever possible, users who want to create new forward models with
new refinable parameters are always
recommended to make use of parameter variables existing in the
program, because they all have optimizers
already linked to them. These include the following:

+----------------------------+-----------------------------------------+
| **Var name**               | **Shape**                               |
+============================+=========================================+
| ``probe_real``             | ``[n_modes, tile_len_y, tile_len_x]``   |
+----------------------------+-----------------------------------------+
| ``probe_imag``             | ``[n_modes, tile_len_y, tile_len_x]``   |
+----------------------------+-----------------------------------------+
| ``probe_defocus_mm``       | ``[1]``                                 |
+----------------------------+-----------------------------------------+
| ``probe_pos_offset``       | ``[n_theta, 2]``                        |
+----------------------------+-----------------------------------------+
| ``probe_pos_correction``   | ``[n_theta, n_tiles_per_angle]``        |
+----------------------------+-----------------------------------------+
| ``slice_pos_cm_ls``        | ``[n_slices]``                          |
+----------------------------+-----------------------------------------+
| ``free_prop_cm``           | ``[1] or [n_distances]``                |
+----------------------------+-----------------------------------------+
| ``tilt_ls``                | ``[3, n_theta]``                        |
+----------------------------+-----------------------------------------+
| ``prj_affine_ls``          | ``[n_distances, 2, 3]``                 |
+----------------------------+-----------------------------------------+
| ``ctf_lg_kappa``           | ``[1]``                                 |
+----------------------------+-----------------------------------------+

Adding new refinable parameters (at the current stage) involves some
hard coding. To do that, take the following
steps:

#. in ``ptychography.py``, find the code block labeled by
   ``"Create variables and optimizers for other parameters (probe, probe defocus, probe positions, etc.)."``
   In this block, declare the variable use
   ``adorym.wrapper.create_variable``, and add it to the dictionary
   ``optimizable_params``. The name of the variable must match the name
   of the argument defined in your ``ForwardModel`` class.

#. In the argument list of ``ptychography.reconstruct_ptychography``,
   add an optimization switch for the new variable. Optionally, also add
   an variable to hold pre-declared optimizer for this variable, and set
   the default to ``None``.

#. In function ``create_and_initialize_parameter_optimizers`` within
   ``adorym/optimizers.py``, define how the optimizer of the parameter
   variable should be defined. You can use the existing optimizer
   declaration codes for other parameters as a template.

#. If the parameter requires a special rule when it is defined, updated,
   or outputted, you will also need to explicitly modify
   ``create_and_initialize_parameter_optimizers``,
   ``update_parameters``, ``create_parameter_output_folders``, and
   ``output_intermediate_parameters``.

Publications
------------

- \M. Du, S. Kandel, J. Deng, X. Huang, A. Demortiere, T. T. Nguyen, R. Tucoulou, V. D. Andrade, Q. Jin, C. Jacobsen, Adorym: A multi-platform generic x-ray image reconstruction framework based on automatic differentiation. *Arxiv*, arXiv:2012.12686 (2020).

The early version of Adorym, which was used to demonstrate 3D
reconstruction of continuous object beyond the depth of focus, is
published as

- \M. Du, Y. S. G. Nashed, S. Kandel, D. Gürsoy, C. Jacobsen, Three dimensions, two microscopes, one code: Automatic differentiation for x-ray nanotomography beyond the depth of focus limit. *Sci Adv.* **6**, eaay3700 (2020).
