# Notes

* I believe `delta_beta` is the same as `mag_phase`? Not quite. I think it is `phase_mag` instead

* Delta is the phase term, beta is the absorption term.

* `delta_beta`, `beta` is the complex/imaginary part of the complex number. Since it is multiplied by i, it becomes real.

* n = 1 - `delta` - `i*beta` 

In the object initialization and representation now, util.py also needs to be updated.
