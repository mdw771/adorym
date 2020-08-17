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

To learn the settings of the ``reconstruct_ptychography`` function and the expected HDF5 format,
please visit `Usage <usage.html#parameter-settings-in-main-function>`__.