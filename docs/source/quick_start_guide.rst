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
