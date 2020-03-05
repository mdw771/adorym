import warnings
import os
import numpy as np
try:
    import autograd.numpy as anp
except:
    warnings.warn('Autograd backend is not available.')
try:
    import torch as tc
    import torch.autograd as ta
except:
    warnings.warn('PyTorch backend is not available.')

from global_settings import backend

func_mapping_dict = {'zeros':       {'autograd': anp.zeros,      'pytorch': tc.zeros},
                     'ones':        {'autograd': anp.ones,       'pytorch': tc.ones},
                     'zeros_like':  {'autograd': anp.zeros_like, 'pytorch': tc.zeros_like},
                     'ones_like':   {'autograd': anp.ones_like,  'pytorch': tc.ones_like},
                     'stack':       {'autograd': anp.stack,      'pytorch': tc.stack},
                     'exp':         {'autograd': anp.exp,        'pytorch': tc.exp},
                     }

class ADVariable(object):

    def __init__(self, arr, dtype=None, device=None, requires_grad=True):
        """
        Create a variable wrapper.
        :param arr: Numpy array of the intial value.
        :param dtype: Data type.
        :param device: A device object from PyTorch, etc. Use None for CPU.
        """
        self.device = device
        if isinstance(arr, np.ndarray):
            self.shape = arr.shape
            self.dtype = dtype if dtype is not None else arr.dtype
            if backend == 'autograd':
                self.var = anp.array(arr)
            elif backend == 'pytorch':
                self.var = tc.tensor(arr, dtype=self.dtype, device=device, requires_grad=requires_grad)
        else:
            self.var = arr
            if backend in ['autograd', 'pytorch']:
                self.shape = arr.shape
                self.dtype = dtype if dtype is not None else arr.dtype

    def __str__(self):
        return self.var.__str__()

    def __repr__(self):
        return self.var.__repr__()

    def __add__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var + other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var + other, dtype=self.dtype, device=self.device)

    def __sub__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var - other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var - other, dtype=self.dtype, device=self.device)

    def __mul__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var * other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var * other, dtype=self.dtype, device=self.device)

    def __truediv__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var / other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var / other, dtype=self.dtype, device=self.device)

    def __floordiv__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var // other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var // other, dtype=self.dtype, device=self.device)

    def __pow__(self, other):
        if isinstance(other, ADVariable):
            return ADVariable(self.var ** other.var, dtype=self.dtype, device=self.device)
        else:
            return ADVariable(self.var ** other, dtype=self.dtype, device=self.device)


def get_device(index=None, backend='autograd'):
    """
    Get device object.
    :param index: index of GPU. Set to None if the tensor is kept on host.
    """
    if backend == 'autograd': return None
    elif backend == 'pytorch':
        if index is None: return None
        else:
            return tc.device('cuda:{}'.format(index))

def zeros(shape, dtype=None, device=None):
    arr = func_mapping_dict['zeros'][backend](shape)
    return ADVariable(arr, dtype=dtype, device=device)


def ones(shape, dtype=None, device=None):
    arr = func_mapping_dict['ones'][backend](shape)
    return ADVariable(arr, dtype=dtype, device=device)


def zeros_like(var, dtype=None, device=None):
    if isinstance(var, ADVariable):
        arr = func_mapping_dict['zeros_like'][backend](var.var)
    else:
        arr = func_mapping_dict['zeros_like'][backend](var)
    return ADVariable(arr, dtype=dtype, device=device)

def exp(var):
    arr = func_mapping_dict['exp'][backend]


if __name__ == '__main__':

    a = ADVariable(np.array([1, 2, 3]))
    b = ADVariable(np.array([3, 2, 2]))
    c = a + b
    print(c)
    d = zeros_like(a)
    print(d)