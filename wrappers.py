import warnings
import os
import numpy as np

from global_settings import backend

engine_dict = {}
try:
    import autograd.numpy as anp
    engine_dict['autograd'] = anp
except:
    warnings.warn('Autograd backend is not available.')
try:
    import torch as tc
    import torch.autograd as tag
    engine_dict['pytorch'] = tc
except:
    warnings.warn('PyTorch backend is not available.')


func_mapping_dict = {'zeros':       {'autograd': 'zeros',      'tensorflow': 'zeros',      'pytorch': 'zeros'},
                     'ones':        {'autograd': 'ones',       'tensorflow': 'ones',       'pytorch': 'ones'},
                     'zeros_like':  {'autograd': 'zeros_like', 'tensorflow': 'zeros_like', 'pytorch': 'zeros_like'},
                     'ones_like':   {'autograd': 'ones_like',  'tensorflow': 'ones_like',  'pytorch': 'ones_like'},
                     'stack':       {'autograd': 'stack',      'tensorflow': 'stack',      'pytorch': 'stack'},
                     'concatenate': {'autograd': 'concatenate','tensorflow': 'cat',        'pytorch': 'cat'},
                     'exp':         {'autograd': 'exp',        'tensorflow': 'exp',        'pytorch': 'exp'},
                     'round':       {'autograd': 'round',      'tensorflow': 'round',      'pytorch': 'round'},
                     'clip':        {'autograd': 'clip',       'tensorflow': 'clip',       'pytorch': 'clamp'},
                     'reshape':     {'autograd': 'reshape',    'tensorflow': 'reshape',    'pytorch': 'reshape'},
                     'floor':       {'autograd': 'floor',      'tensorflow': 'floor',      'pytorch': 'floor'},
                     'ceil':        {'autograd': 'ceil',       'tensorflow': 'ceil',       'pytorch': 'ceil'},
                     'sqrt':        {'autograd': 'sqrt',       'tensorflow': 'sqrt',       'pytorch': 'sqrt'},
                     'real':        {'autograd': 'real',       'tensorflow': 'real',       'pytorch': 'real'},
                     'imag':        {'autograd': 'imag',       'tensorflow': 'imag',       'pytorch': 'imag'},
                     'sin':         {'autograd': 'sin',        'tensorflow': 'sin',        'pytorch': 'sin'},
                     'cos':         {'autograd': 'cos',        'tensorflow': 'cos',        'pytorch': 'cos'},
                     'abs':         {'autograd': 'abs',        'tensorflow': 'abs',        'pytorch': 'abs'},
                     'sum':         {'autograd': 'sum',        'tensorflow': 'reduce_sum', 'pytorch': 'sum'},
                     'arctan2':     {'autograd': 'arctan2',    'tensorflow': 'atan2',      'pytorch': 'atan2'},
                     }

dtype_mapping_dict = {'float32':    {'autograd': 'float32',    'tensorflow': 'float32',    'pytorch': 'float'},
                      'float64':    {'autograd': 'float64',    'tensorflow': 'float64',    'pytorch': 'double'},
                      'float16':    {'autograd': 'float16',    'tensorflow': 'float16',    'pytorch': 'half'},
                      'int8':       {'autograd': 'int8',       'tensorflow': 'int8',       'pytorch': 'int8'},
                      'int16':      {'autograd': 'int16',      'tensorflow': 'int16',      'pytorch': 'short'},
                      'int32':      {'autograd': 'int32',      'tensorflow': 'int32',      'pytorch': 'int'},
                      'int64':      {'autograd': 'int64',      'tensorflow': 'int64',      'pytorch': 'long'},
                      'bool':       {'autograd': 'bool',       'tensorflow': 'bool',       'pytorch': 'bool'},
                      }

# _____________
# |Flow control|_____________________________________________________________

def create_variable(arr, dtype=None, device=None, requires_grad=True):
    """
    Create a variable wrapper.
    :param arr: Numpy array of the intial value.
    :param dtype: str; Data type.
    :param device: A device object from PyTorch, etc. Use None for CPU.
    """
    args = {}
    if backend == 'autograd':
        if dtype is not None:
            args['dtype'] = dtype_mapping_dict[dtype]['autograd']
        var = anp.array(arr, **args)
    elif backend == 'pytorch':
        if dtype is not None:
            args['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[self.dtype]['pytorch'])
        if device is not None:
            args['device'] = device
        args['requires_grad'] = requires_grad
        var = tc.tensor(arr, **args)
    return var


def to_numpy(var):
    if backend == 'autograd':
        return var._value
    elif backend == 'pytorch':
        if var.device.type == 'cpu':
            return var.data.numpy()
        else:
            return var.cpu().data.numpy()


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


def prepare_loss_node(loss, opt_args_ls=None):
    if backend == 'autograd':
        return anp.grad(loss, opt_args_ls)
    elif backend == 'pytorch':
        return loss


def get_gradients(loss_node, opt_args_ls=None, **kwargs):
    if backend == 'autograd':
        # For Autograd, loss_node is the grad function that takes the loss function arguments and
        # returns the gradients.
        return loss_node(**kwargs)
    elif backend == 'pytorch':
        # For PyTorch, loss_node is the loss function itself.
        l = loss_node(**kwargs)
        kwargs_ls = list(kwargs.values())
        dx_ls = []
        for i, node in enumerate(kwargs_ls):
            if i in opt_args_ls: dx_ls.append(node)
        return tag.grad(l, dx_ls, retain_graph=True)

# ________________
# |Maths functions|_____________________________________________________________

def zeros(shape, dtype=None, device=None):
    func = getattr(engine_dict[backend], func_mapping_dict['zeros'][backend])
    arr = func(shape)
    return arr


def ones(shape, dtype=None, device=None):
    func = getattr(engine_dict[backend], func_mapping_dict['ones'][backend])
    arr = func(shape)
    return arr


def zeros_like(var, dtype=None, device=None):
    """
    :param var: ADVariable or tensor.
    """
    func = getattr(engine_dict[backend], func_mapping_dict['zeros_like'][backend])
    arr = func(var)
    return arr


def exp(var):
    func = getattr(engine_dict[backend], func_mapping_dict['exp'][backend])
    arr = func(var)
    return arr


def sin(var):
    func = getattr(engine_dict[backend], func_mapping_dict['sin'][backend])
    arr = func(var)
    return arr


def cos(var):
    func = getattr(engine_dict[backend], func_mapping_dict['cos'][backend])
    arr = func(var)
    return arr


def exp_complex(var_real, var_imag):
    e = exp(var_real)
    return e * cos(var_imag), e * sin(var_imag)


def abs(var):
    func = getattr(engine_dict[backend], func_mapping_dict['abs'][backend])
    arr = func(var)
    return arr


def stack(var_list, axis=0):
    func = getattr(engine_dict[backend], func_mapping_dict['stack'][backend])
    arr = func(var_list, axis)
    return arr


def concatenate(var_list, axis=0):
    func = getattr(engine_dict[backend], func_mapping_dict['concatenate'][backend])
    arr = func(var_list, axis)
    return arr


def cast(var, dtype):
    dtype = str(dtype)
    if backend == 'autograd':
        return var.astype(dtype)
    elif backend == 'pytorch':
        return getattr(var, dtype_mapping_dict[dtype]['pytorch'])()


def round(var):
    func = getattr(engine_dict[backend], func_mapping_dict['round'][backend])
    arr = func(var)
    return arr


def round_and_cast(var, dtype='int32'):
    return cast(round(var), dtype=dtype)


def fft2(var_real, var_imag, axes=(-2, -1)):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        var = anp.fft.fft2(var, axes=axes)
        return var.real, var.imag
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], axis=-1)
        var = tc.fft(var, signal_ndim=2)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        return var_real.squeeze(), var_imag.squeeze()


def ifft2(var_real, var_imag, axes=(-2, -1)):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        var = anp.fft.ifft2(var, axes=axes)
        return var.real, var.imag
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], axis=-1)
        var = tc.ifft(var, signal_ndim=2)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        return var_real.squeeze(), var_imag.squeeze()


def convolve_with_transfer_function(arr_real, arr_imag, h_real, h_imag):
    f_real, f_imag = fft2(arr_real, arr_imag)
    fh_real = f_real * h_real - f_imag * h_imag
    fh_imag = f_real * h_imag + f_imag * h_real
    return ifft2(fh_real, fh_imag)


def convolve_with_impulse_response(arr_real, arr_imag, h_real, h_imag):
    f_real, f_imag = fft2(arr_real, arr_imag)
    h_real, h_imag = fft2(h_real, h_imag)
    fh_real = f_real * h_real - f_imag * h_imag
    fh_imag = f_real * h_imag + f_imag * h_real
    return ifft2(fh_real, fh_imag)


def fftshift(var, axes=(1, 2)):
    """
    :param var: [N, H, W, 2], where the last dimension represents real and imaginary parts.
    """
    if backend == 'autograd':
        return anp.fft.fftshift(var, axes=axes)
    elif backend == 'pytorch':
        s = var.shape
        for i in axes:
            p2 = (s[i] + 1) // 2
            v = tc.split(var, p2, dim=i)
            if len(v) == 3:
                v1, v2 = (v[0], tc.cat([v[1], v[2]], dim=i))
            else:
                v1, v2 = v
            var = tc.cat([v2, v1], dim=i)
        return var


def ifftshift(var, axes=(1, 2)):
    """
    :param var: [N, H, W, 2], where the last dimension represents real and imaginary parts.
    """
    if backend == 'autograd':
        return anp.fft.ifftshift(var, axes=axes)
    elif backend == 'pytorch':
        s = var.shape
        for i in axes:
            p2 = s[i] - (s[i] + 1) // 2
            v = tc.split(var, p2, dim=i)
            if len(v) == 3:
                v1, v2 = (v[0], tc.cat([v[1], v[2]], dim=i))
            else:
                v1, v2 = v
            var = tc.cat([v2, v1], dim=i)
        return var


def split_channel(var):
    if backend == 'autograd':
        var0, var1 = anp.split(var, var.shape[-1], axis=-1)
        return anp.squeeze(var0), anp.squeeze(var1)
    elif backend == 'pytorch':
        var0, var1 = tc.split(var, 1, dim=-1)
        return var0.squeeze(), var1.squeeze()
   
    
def clip(var, a1, a2):
    func = getattr(engine_dict[backend], func_mapping_dict['clip'][backend])
    arr = func(var, a1, a2)
    return arr


def reshape(var, newshape):
    func = getattr(engine_dict[backend], func_mapping_dict['reshape'][backend])
    arr = func(var, newshape)
    return arr


def floor(var):
    func = getattr(engine_dict[backend], func_mapping_dict['floor'][backend])
    arr = func(var)
    return arr


def floor_and_cast(var, dtype='int32'):
    return cast(floor(var), dtype=dtype)


def ceil(var):
    func = getattr(engine_dict[backend], func_mapping_dict['ceil'][backend])
    arr = func(var)
    return arr


def ceil_and_cast(var, dtype='int32'):
    return cast(ceil(var), dtype=dtype)


def sqrt(var):
    func = getattr(engine_dict[backend], func_mapping_dict['sqrt'][backend])
    arr = func(var)
    return arr


def mean(var, axis=None):
    if backend == 'autograd':
        return anp.mean(var, axis=axis)
    elif backend == 'pytorch':
        if axis is not None: warnings.warn('PyTorch function "mean" does not support argument "axis".')
        return tc.mean(var)


def max(var, return_number=True, axis=None):
    if backend == 'autograd':
        a = anp.max(var, axis=axis)
    elif backend == 'pytorch':
        if axis is None:
            a = tc.max(var)
            if return_number:
                a = float(to_numpy(a))
        else:
            a = tc.max(var, dim=axis)
    return a


def min(var, return_number=True, axis=None):
    if backend == 'autograd':
        a = anp.min(var, axis=axis)
    elif backend == 'pytorch':
        if axis is None:
            a = tc.min(var)
            if return_number:
                a = float(to_numpy(a))
        else:
            a = tc.min(var, dim=axis)
    return a


def real(var):
    func = getattr(engine_dict[backend], func_mapping_dict['real'][backend])
    arr = func(var)
    return arr


def imag(var):
    func = getattr(engine_dict[backend], func_mapping_dict['imag'][backend])
    arr = func(var)
    return arr


def tile(var, cp):
    if backend == 'autograd':
        return anp.tile(var, cp)
    elif backend == 'pytorch':
        return var.repeat(*cp)


def pad(var, pad_len, mode='constant'):
    """
    :param pad_len: A tuple of tuples. Consistent with the format of numpy.pad.
    """
    if backend == 'autograd':
        return np.pad(var, pad_len, mode=mode)
    elif backend == 'pytorch':
        pad_len = [x for y in pad_len[::-1] for x in y]
        return tc.nn.functional.pad(var, pad_len, mode=mode)


def sum(var):
    func = getattr(engine_dict[backend], func_mapping_dict['sum'][backend])
    arr = func(var)
    return arr


def roll(var, shifts, axes=0):
    if backend == 'autograd':
        return np.roll(var, shifts, axes=axes)
    elif backend == 'pytorch':
        return tc.roll(var, shifts, dims=axes)


def arctan2(var):
    func = getattr(engine_dict[backend], func_mapping_dict['arctan2'][backend])
    arr = func(var)
    return arr


if __name__ == '__main__':

    import torch as tc
    backend = 'pytorch'


    dev = get_device(0)
    a = create_variable(tc.tensor([1., 2., 3.]))
    b = create_variable(np.array([3., 2., 2.]))
    print(a)
    print(b)
    c = a + b
    print(c)
    d = tc.zeros_like(a)
    d = getattr(tc, 'zeros_like')(a)
    d = zeros_like(a)
    print(d)
    print(to_numpy(tc.exp(a)))