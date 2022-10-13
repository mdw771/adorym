# Notes

- I believe `delta_beta` is the same as `mag_phase`? Not quite. I think it is
  `phase_mag` instead

- Delta is the phase term, beta is the absorption term.

- `delta_beta`, `beta` is the complex/imaginary part of the complex number.
  Since it is multiplied by i, it becomes real.

- n = 1 - `delta` - `i*beta`

In the object initialization and representation now, util.py also needs to be
updated.

## Optimizers

Optimizers need to be fixed for complex numbers, especially ADAM. ADAM uses the
gradients and arrays to calculate momentum and things and this is not correctly
implemented for complex numbers.

The second problem is that random optimizers are used for the other parameters
in the parameter list, I think they should all be the same, with whatever is
specified in the parameter file.

## 2022.10.12

ADAM is working great, have fixed two bugs in some crucial functions.

## 2022.10.13

CG works, I think that means complex numbers as a whole work now. The line-search was changed based on the following links [[1](https://www.manopt.org/tutorial.html#manifolds), [2](https://math.stackexchange.com/questions/2449067/difference-between-grassmann-and-stiefel-manifolds), [3](https://github.com/pymanopt/pymanopt/blob/master/pymanopt/optimizers/conjugate_gradient.py#L286), [4](https://github.com/pymanopt/pymanopt/blob/master/pymanopt/manifolds/grassmann.py#L183)], with the idea that the mathematicians know what they're doing.
