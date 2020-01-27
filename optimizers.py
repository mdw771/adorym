import autograd.numpy as np
from util import print_flush



def apply_gradient_adam(x, g, i_batch, m=None, v=None, step_size=0.001, b1=0.9, b2=0.999, eps=1e-7, verbose=True, comm=None):

    g = np.array(g)
    if m is None or v is None:
        m = np.zeros_like(x)
        v = np.zeros_like(v)
    m = (1 - b1) * g + b1 * m  # First moment estimate.
    v = (1 - b2) * (g ** 2) + b2 * v  # Second moment estimate.
    mhat = m / (1 - b1 ** (i_batch + 1))  # Bias correction.
    vhat = v / (1 - b2 ** (i_batch + 1))
    d = step_size * mhat / (np.sqrt(vhat) + eps)
    x = x - d
    if verbose:
        try:
            print_flush('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))), 0, comm.Get_rank())
        except:
            print('  Step size modifier is {}.'.format(np.mean(mhat / (np.sqrt(vhat) + eps))))
    return x, m, v


def apply_gradient_gd(x, g, step_size=0.001, dynamic_rate=True, i_batch=0, first_downrate_iteration=92, comm=None):

    g = np.array(g)
    if dynamic_rate:
        threshold_iteration = first_downrate_iteration
        i = 1
        while threshold_iteration < i_batch:
            threshold_iteration += first_downrate_iteration * 2 ** i
            i += 1
            step_size /= 2.
            print_flush('  -- Step size halved.', 0, comm.Get_rank())
    x = x - step_size * g

    return x