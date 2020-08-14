Customization
-------------

Adding your own forward model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| You can create additional forward models beyond the existing ones. To
begin with, in ``adorym/forward_model.py``,
| create a class inheriting ``ForwardModel`` (*i.e.*,
``class MyNovelModel(ForwardModel)``). Each forward model class
| should contain 4 essential methods: ``predict``, ``get_data``,
``loss``, and ``get_loss_function``. ``predict`` maps input variables
| to predicted quantities (usually the real-numbered magnitude of the
detected wavefield). ``get_data`` reads from
| the HDF5 file the raw data corresponding to the minibatch currently
being processed. ``loss`` is the last-layer
| loss node that computes the (regularized)
| loss values from the predicted data and the experimental measurement
for the current minibatch. ``get_loss_function``
| concatenates the above methods and return the end-to-end loss
function. If your ``predict`` returns the real-numbered
| magnitude of the detected wavefield, you can use ``loss`` inherented
from the parent class, although you still need to
| make a copy of ``get_loss_function`` and explicitly change its
arguments to match those of ``predict`` (do not use
| implicit argument tuples or dictionaries like ``*args`` and
``**kwargs``, as that won't work with Autograd!). If your ``predict``
| returns something else, you may also need to override ``loss``. Also
make sure your new forward model class contains
| a ``self.argument_ls`` attribute, which should be a list of argument
strings that exactly matches the signature of ``predict``.

| To use your forward model, pass your forward model class to the
``forward_model`` argument of ``reconstruct_ptychography``.
| For example, in the script that you execute with Python, do the
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

| Whenever possible, users who want to create new forward models with
new refinable parameters are always
| recommended to make use of parameter variables existing in the
program, because they all have optimizers
| already linked to them. These include the following:

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

| Adding new refinable parameters (at the current stage) involves some
hard coding. To do that, take the following
| steps:

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
