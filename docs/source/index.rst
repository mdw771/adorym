.. adorym documentation master file, created by
   sphinx-quickstart on Fri Aug 14 11:50:10 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Adorym's documentation!
==================================

Adorym, (\ **A**\ utomatic \ **D**\ ifferentiation-based \ **O**\ bject \ **R**\ econstruction with D\ **y**\ na\ **m**\ ical Scattering) is a Python package for solving inverse problems in generic optical image reconstruction tasks (\ *e.g.*\ , coherent diffraction imaging, holography, ptychography, line-projection tomography, ptychotomography, and multislice ptychotomography). The package uses automatic differentiation to calculate the gradient used for iterative optimization algorithms from the given forward model with minimal human intervention, saving the labors needed for rederiving the gradient when one switches/tweaks the forward model. Adorym supports the use of user-defined forward model, making it easy to test/build algorithms for novel imaging techniques.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   installation
   quick_start_guide
   api_references
   customization
   publications

