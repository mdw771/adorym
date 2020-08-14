Installation
------------

Get this repository to your hard drive using

::

    git clone https://github.com/mdw771/adorym

and then use PIP to build and install:

::

    pip install ./adorym

| If you will modify internal functions of Adorym, *e.g.*, add new
forward
| models or refinable parameters, it is suggested to use the ``-e`` flag
to
| enable editable mode so that you don't need to rebuild Adorym each
time
| you make changes to the source code:

::

    pip install -e ./adorym

| After installation, type ``python`` to open a Python console, and
check
| the installation status using ``import adorym``. If an ``ImportError``
occurs,
| you may need to manually install the missing dependencies. Most
| dependencies are available on PIP and can be acquired with

::

    pip install <package_name>

or through Conda if you use the Anaconda or Miniconda distribution of
Python:

::

    conda install <package_name>

| In order to run Adorym using PyTorch backend with GPU support, please
| make sure the right version of PyTorch that matches your CUDA version
| is installed. The latter can be checked through ``nvidia-smi``.
