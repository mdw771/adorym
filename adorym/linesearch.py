from typing import Callable, NamedTuple
import adorym.wrappers as w
import numpy as np

__all__ = ['BackTrackingLineSearch', 'AdaptiveLineSearch']


class LSState(NamedTuple):
    newf: object
    newx: object
    alpha: object
    step_count: object


class BackTrackingLineSearch:
    """Adapted from the backtracking line search in the manopt package"""
    def __init__(self, contraction_factor: float = 0.5,
                 optimism: float = 3.,
                 suff_decr: float = 1e-4,
                 initial_stepsize: float = 10.0,
                 stepsize_threshold_low: float = 1e-10,
                 dtype: np.dtype = np.float32,
                 maxiter: int = None,
                 name='backtracking_linesearch',
                 normalize_alpha=True) -> None:
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.initial_stepsize = initial_stepsize
        self.stepsize_threshold_low = stepsize_threshold_low
        self.normalize_alpha = normalize_alpha

        self._dtype = dtype
        self._machine_eps = np.finfo(dtype).eps

        self._name = name

        machine_maxiter = np.ceil(np.log(self._machine_eps) / np.log(self.contraction_factor))

        if maxiter is None:
            maxiter = np.inf
        self.maxiter = np.minimum(maxiter, machine_maxiter).astype('int32')

        self._oldf0 = -np.inf
        self._alpha = 0.

        self._variables = [self._oldf0, self._alpha]

    def search(self, objective_and_update: Callable,
               x0, descent_dir, gradient, f0=None):

        if f0 is None:
            f0, _ = objective_and_update(x0, w.zeros_like(x0))

        # Calculating the directional derivative along the descent direction
        descent_norm = w.vec_norm(descent_dir)
        df0 = w.tensordot(gradient.conj().squeeze(), descent_dir.squeeze(), axes=2).real

        if self._oldf0 >= f0:
            # Pick initial step size based on where we were last time
            alpha = 2 * (f0 - self._oldf0) / df0

            # Look a little further
            alpha *= self.optimism
            if alpha * descent_norm < self._machine_eps:
                if self.normalize_alpha:
                    alpha = self.initial_stepsize / descent_norm
                else:
                    alpha = self.initial_stepsize
        else:
            if self.normalize_alpha:
                alpha = self.initial_stepsize / descent_norm
            else:
                alpha = self.initial_stepsize

        # Make the chosen sten and compute the cost there
        newf, newx = objective_and_update(x0, alpha * descent_dir)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        def _cond(state: LSState):
            cond1 = state.newf > f0 + self.suff_decr * state.alpha * df0
            cond2 = state.step_count <= self.maxiter
            cond3 = state.alpha > self.stepsize_threshold_low
            return cond1 and cond2 and cond3

        lsstate_new = LSState(newf=newf, newx=newx, alpha=alpha, step_count=step_count)
        while _cond(lsstate_new):
            alpha = self.contraction_factor * lsstate_new.alpha
            newf, newx = objective_and_update(x0, alpha * descent_dir)
            lsstate_new = LSState(newf=newf,
                          newx=newx,
                          alpha=alpha,
                          step_count=lsstate_new.step_count + 1)

        self._oldf0 = f0
        self._alpha = lsstate_new.alpha
        if lsstate_new.newf <= f0:
            lsstate_updated = lsstate_new
        else:
            lsstate_updated = LSState(newf=f0, newx=x0, alpha=0., step_count=lsstate_new.step_count)

        return lsstate_updated


class AdaptiveLineSearch:
    """Adapted from the backtracking line search in the manopt package"""
    def __init__(self, contraction_factor: float = 0.5,
                 optimism: float = 2.,
                 suff_decr: float = 1e-4,
                 initial_stepsize: float = 10.0,
                 stepsize_threshold_low: float = 1e-10,
                 dtype: np.dtype = np.float32,
                 maxiter: int = None,
                 name='backtracking_linesearch',
                 normalize_alpha=True) -> None:
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.initial_stepsize = initial_stepsize
        self.stepsize_threshold_low = stepsize_threshold_low
        self.normalize_alpha=normalize_alpha

        self._dtype = dtype
        self._machine_eps = np.finfo(dtype).eps

        self._name = name

        machine_maxiter = np.ceil(np.log(self._machine_eps) / np.log(self.contraction_factor))

        if maxiter is None:
            maxiter = np.inf
        self.maxiter = np.minimum(maxiter, machine_maxiter).astype('int32')

        self._alpha = 0
        self._alpha_suggested = 0

        self._variables = [self._alpha, self._alpha_suggested]


    def search(self, objective_and_update: Callable,
               x0, descent_dir, gradient, f0=None):

        if f0 is None:
            f0, _ = objective_and_update(x0, w.zeros_like(x0))

        # Calculating the directional derivative along the descent direction
        descent_norm = w.vec_norm(descent_dir)
        df0 = w.tensordot(gradient.conj().squeeze(), descent_dir.squeeze(), axes=2).real

        if self._alpha_suggested > 0:
            alpha = self._alpha_suggested
        else:
            if self.normalize_alpha:
                alpha = self.initial_stepsize / descent_norm
            else:
                alpha = self.initial_stepsize

        # Make the chosen sten and compute the cost there
        newf, newx = objective_and_update(x0, alpha * descent_dir)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        def _cond(state: LSState):
            cond1 = state.newf > f0 + self.suff_decr * state.alpha * df0
            cond2 = (state.step_count <= self.maxiter)
            cond3 = state.alpha > self.stepsize_threshold_low
            return cond1 and cond2 and cond3

        lsstate_new = LSState(newf=newf, newx=newx, alpha=alpha, step_count=step_count)
        while _cond(lsstate_new):
            alpha = self.contraction_factor * lsstate_new.alpha
            newf, newx = objective_and_update(x0, alpha * descent_dir)
            lsstate_new = LSState(newf=newf,
                            newx=newx,
                            alpha=alpha,
                            step_count=lsstate_new.step_count + 1)

        # New suggestion for step size
        if lsstate_new.step_count - 1 == 0:
            # case 1: if things go very well (step count is 1), push your luck
            suggested_alpha = self.optimism * lsstate_new.alpha
        elif lsstate_new.step_count - 1 == 1:
            # case 2: if things go reasonably well (step count is 2), try to keep pace
            suggested_alpha = lsstate_new.alpha
        else:
            # case 3: if we backtracked a lot, the new stepsize is probably quite small:
            # try to recover
            suggested_alpha = self.optimism * lsstate_new.alpha

        self._alpha_suggested = suggested_alpha
        self._alpha = lsstate_new.alpha

        if lsstate_new.newf <= f0:
            lsstate_updated = lsstate_new
        else:
            print('Line search is unable to find a smaller loss ({} > {})!'.format(lsstate_new.newf, f0))
            lsstate_updated = LSState(newf=f0, newx=x0, alpha=0., step_count=lsstate_new.step_count)

        return lsstate_updated
