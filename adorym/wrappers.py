import warnings
import os
import gc
import numpy as np
import scipy
import scipy.signal

import adorym.global_settings as global_settings

engine_dict = {}
try:
    import autograd.numpy as anp
    import autograd as ag
    engine_dict['autograd'] = anp
    flag_autograd_avail = True
except:
    warnings.warn('Autograd backend is not available.')
    flag_autograd_avail = False
try:
    import torch as tc
    import torch.autograd as tag
    engine_dict['pytorch'] = tc
    flag_pytorch_avail = True
except:
    warnings.warn('PyTorch backend is not available.')
    flag_pytorch_avail = False


func_mapping_dict = {'zeros':       {'autograd': 'zeros',      'tensorflow': 'zeros',      'pytorch': 'zeros',      'numpy': 'zeros'},
                     'ones':        {'autograd': 'ones',       'tensorflow': 'ones',       'pytorch': 'ones',       'numpy': 'ones'},
                     'zeros_like':  {'autograd': 'zeros_like', 'tensorflow': 'zeros_like', 'pytorch': 'zeros_like', 'numpy': 'zeros_like'},
                     'ones_like':   {'autograd': 'ones_like',  'tensorflow': 'ones_like',  'pytorch': 'ones_like',  'numpy': 'ones_like'},
                     'stack':       {'autograd': 'stack',      'tensorflow': 'stack',      'pytorch': 'stack',      'numpy': 'stack'},
                     'concatenate': {'autograd': 'concatenate','tensorflow': 'cat',        'pytorch': 'cat',        'numpy': 'concatenate'},
                     'exp':         {'autograd': 'exp',        'tensorflow': 'exp',        'pytorch': 'exp'},
                     'log':         {'autograd': 'log',        'tensorflow': 'log',        'pytorch': 'log'},
                     'round':       {'autograd': 'round',      'tensorflow': 'round',      'pytorch': 'round'},
                     'clip':        {'autograd': 'clip',       'tensorflow': 'clip',       'pytorch': 'clamp'},
                     'reshape':     {'autograd': 'reshape',    'tensorflow': 'reshape',    'pytorch': 'reshape'},
                     'floor':       {'autograd': 'floor',      'tensorflow': 'floor',      'pytorch': 'floor'},
                     'ceil':        {'autograd': 'ceil',       'tensorflow': 'ceil',       'pytorch': 'ceil'},
                     'sqrt':        {'autograd': 'sqrt',       'tensorflow': 'sqrt',       'pytorch': 'sqrt'},
                     'real':        {'autograd': 'real',       'tensorflow': 'real',       'pytorch': 'real'},
                     'imag':        {'autograd': 'imag',       'tensorflow': 'imag',       'pytorch': 'imag'},
                     'sin':         {'autograd': 'sin',        'tensorflow': 'sin',        'pytorch': 'sin',        'numpy': 'sin'},
                     'cos':         {'autograd': 'cos',        'tensorflow': 'cos',        'pytorch': 'cos',        'numpy': 'cos'},
                     'abs':         {'autograd': 'abs',        'tensorflow': 'abs',        'pytorch': 'abs',        'numpy': 'abs'},
                     'sum':         {'autograd': 'sum',        'tensorflow': 'reduce_sum', 'pytorch': 'sum'},
                     'prod':        {'autograd': 'prod',       'tensorflow': 'prod',       'pytorch': 'prod'},
                     'arctan2':     {'autograd': 'arctan2',    'tensorflow': 'atan2',      'pytorch': 'atan2'},
                     'nonzero':     {'autograd': 'nonzero',    'tensorflow': 'nonzero',    'pytorch': 'nonzero'},
                     'sign':        {'autograd': 'sign',       'tensorflow': 'sign',       'pytorch': 'sign',       'numpy': 'sign'},
                     'argmax':      {'autograd': 'argmax',     'tensorflow': 'argmax',     'pytorch': 'argmax',     'numpy': 'argmax'},
                     'tensordot':   {'autograd': 'tensordot',  'tensorflow': 'tensordot',  'pytorch': 'tensordot',  'numpy': 'tensordot'},
                     }

dtype_mapping_dict = {'float32':    {'autograd': 'float32',    'tensorflow': 'float32',    'pytorch': 'float',  'numpy': 'float32'},
                      'float64':    {'autograd': 'float64',    'tensorflow': 'float64',    'pytorch': 'double', 'numpy': 'float64'},
                      'float16':    {'autograd': 'float16',    'tensorflow': 'float16',    'pytorch': 'half',   'numpy': 'float16'},
                      'int8':       {'autograd': 'int8',       'tensorflow': 'int8',       'pytorch': 'int8',   'numpy': 'int8'},
                      'int16':      {'autograd': 'int16',      'tensorflow': 'int16',      'pytorch': 'short',  'numpy': 'int16'},
                      'int32':      {'autograd': 'int32',      'tensorflow': 'int32',      'pytorch': 'int',    'numpy': 'int32'},
                      'int64':      {'autograd': 'int64',      'tensorflow': 'int64',      'pytorch': 'long',   'numpy': 'int64'},
                      'bool':       {'autograd': 'bool',       'tensorflow': 'bool',       'pytorch': 'bool',   'numpy': 'bool'},
                      }

if flag_pytorch_avail:
    try:
        pytorch_dtype_query_mapping_dict = {tc.float32: 'float32',
                                            tc.float64: 'float64',
                                            'float32': 'float32',
                                            'float64': 'float64',
                                            'single': 'float32',
                                            'double': 'float64'}
    except:
        pass


def set_bn(f):
    def func(*args, override_backend=None, **kwargs):
        if 'backend' in kwargs.keys():
            # If "backend" in the wrapper function is specified by user, it overrides the
            # "override_backend" argument in the decorator.
            pass
        else:
            # If "backend" in the wrapper function is not specified, check if "override_backend"
            # argument in the decorator.
            # If so, use its value for the wrappers "backend" argument.
            # If not, use global setting.
            kwargs['backend'] = override_backend if override_backend is not None else global_settings.backend
        return f(*args, **kwargs)
    return func

# _____________
# |Flow control|_____________________________________________________________

class EmptyWith(object):
    def __init__(self):
            pass
    
    def __enter__(self):
            pass
    
    def __exit__(self, exc_type, exc_value, tb):
            pass

@set_bn
def create_variable(arr, dtype='float32', device=None, requires_grad=True, backend='autograd'):
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
            args['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        if device is not None:
            args['device'] = device
        args['requires_grad'] = requires_grad
        var = tc.tensor(arr, **args)
    return var


@set_bn
def create_constant(arr, dtype='float32', device=None, backend='autograd'):
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
        var = np.array(arr, **args)
    elif backend == 'pytorch':
        if dtype is not None:
            args['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        if device is not None:
            args['device'] = device
        args['requires_grad'] = False
        var = tc.tensor(arr, **args)
    else:
        if dtype is not None:
            args['dtype'] = dtype_mapping_dict[dtype]['autograd']
        var = np.array(arr, **args)
    return var


@set_bn
def to_numpy(var, backend='autograd'):
    if isinstance(var, np.ndarray):
        return var
    elif isinstance(var, np.float64):
        return var
    else:
        if backend == 'autograd':
            return var._value
        elif backend == 'pytorch':
            if var.device.type == 'cpu':
                return var.data.numpy()
            else:
                return var.cpu().data.numpy()


@set_bn
def to_cpu(var, backend='autograd'):
    if isinstance(var, np.ndarray):
        return var
    elif isinstance(var, np.float64):
        return var
    else:
        if backend == 'autograd':
            return var
        elif backend == 'pytorch':
            if var.device.type == 'cpu':
                return var
            else:
                return var.cpu()


@set_bn
def to_gpu(var, device='cuda:0', backend='autograd'):
    if isinstance(var, np.ndarray):
        return var
    elif isinstance(var, np.float64):
        return var
    else:
        if backend == 'autograd':
            return var
        elif backend == 'pytorch':
            if var.device.type == 'cuda':
                return var
            else:
                return var.cuda(device=device)


@set_bn
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


@set_bn
def get_var_device(var, backend='autograd'):
    if backend == 'autograd':
        return None
    elif backend == 'pytorch':
        return var.device


@set_bn
def get_var_device_type(var, backend='autograd'):
    if backend == 'autograd':
        return 'cpu'
    elif backend == 'pytorch':
        return var.device.type


@set_bn
def set_device(device, backend='autograd'):
    """
    Set device object. Not useful is backend is Autograd.
    :param device: Device object. Set to None if the tensor is kept on host.
    """
    if backend == 'autograd':
        return None
    elif backend == 'pytorch':
        try:
            tc.cuda.set_device(device)
        except:
            pass


@set_bn
def prepare_loss_node(loss, opt_args_ls=None, backend='autograd'):
    if backend == 'autograd':
        return ag.grad(loss, opt_args_ls)
    elif backend == 'pytorch':
        return loss


@set_bn
def get_gradients(loss_node, opt_args_ls=None, backend='autograd', **kwargs):
    """
    Get gradient.

    :param loss_node: Callable. A function which, given arguments in kwargs, returns the loss.
    :param opt_args_ls: List of Int. Indices of optimizable variables in the loss function's argument list.
    :param backend: Backend.
    :param kwargs: Keyword arguments of the loss function.
    :return: A list of gradients.
    """
    if backend == 'autograd':
        # For Autograd, loss_node is the grad function that takes the loss function arguments and
        # returns the gradients.
        return loss_node(*list(kwargs.values()))
    elif backend == 'pytorch':
        # For PyTorch, loss_node is the loss function itself.
        l = loss_node(**kwargs)
        kwargs_ls = list(kwargs.values())
        dx_ls = []
        for i, node in enumerate(kwargs_ls):
            if i in opt_args_ls: dx_ls.append(node)
        grads = tag.grad(l, dx_ls, retain_graph=True, create_graph=False, allow_unused=False)
        # grads = []
        # l.backward(retain_graph=True)
        # for n in dx_ls:
        #     print(n.grad)
        #     grads.append(n.grad)
        l.detach()
        del l

        return grads


@set_bn
def vjp(func, x, backend='autograd'):
    """
    Returns a constructor that would generate a function that computes the VJP between its argument and the
    Jacobian of func.
    :param func: Function handle of loss function.
    :param x: List. A list of all arguments to func. The order of arguments must match.
    :return: The returned constructor receives the input of the differentiated function as input, and the function it returns
             receives the (adjoint) vector as input.
    """
    if backend == 'autograd':
        return ag.make_vjp(func, x)
    elif backend == 'pytorch':
        raise NotImplementedError('VJP for Pytorch backend is not implemented yet.')


@set_bn
def jvp(func, x, backend='autograd'):
    """
    Returns a constructor that would generate a function that computes the JVP between its argument and the
    Jacobian of func.
    :param func: Function handle of loss function.
    :param x: List. A list of all arguments to func. The order of arguments must match.
    :return: The returned constructor receives the input of the differentiated function as input, and the function it returns
             receives the (adjoint) vector as input.
    """
    if backend == 'autograd':
        return ag.differential_operators.make_jvp_reversemode(func, x)
    elif backend == 'pytorch':
        raise NotImplementedError('VJP for Pytorch backend is not implemented yet.')


@set_bn
def hvp(func, x, backend='autograd'):
    """
    Returns a constructor that would generate a function that computes the HVP between its argument and the
    Hessian of func.
    :param func: Function handle of loss function.
    :param x: List. A list of all arguments to func. The order of arguments must match.
    :return: The returned constructor receives the input of the differentiated function as input, and the function it returns
             receives the (adjoint) vector as input.
    """
    if backend == 'autograd':
        return ag.differential_operators.make_hvp(func, x)
    elif backend == 'pytorch':
        raise NotImplementedError('VJP for Pytorch backend is not implemented yet.')


@set_bn
def get_gpu_memory_usage_mb(backend='autograd'):
    if backend == 'autograd':
        return 0
    elif backend == 'pytorch':
        return tc.cuda.memory_allocated() / 1024 ** 2


@set_bn
def get_gpu_memory_cache_mb(backend='autograd'):
    if backend == 'autograd':
        return 0
    elif backend == 'pytorch':
        return tc.cuda.memory_cached() / 1024 ** 2


@set_bn
def get_peak_gpu_memory_usage_mb(backend='autograd'):
    if backend == 'autograd':
        return 0
    elif backend == 'pytorch':
        return tc.cuda.max_memory_allocated() / 1024 ** 2

@set_bn
def collect_gpu_garbage(backend='autograd'):
    if backend == 'autograd':
        pass
    elif backend == 'pytorch':
        tc.cuda.empty_cache()

@set_bn
def get_allocated_tensors(backend='autograd'):

    def _getr(slist, olist, seen):
            for e in slist:
                if id(e) in seen:
                    continue
                seen[id(e)] = None
                olist.append(e)
                tl = gc.get_referents(e)
                if tl:
                    _getr(tl, olist, seen)

    def get_all_objects():
        """Return a list of all live Python
        objects, not including the list itself."""
        gcl = gc.get_objects()
        olist = []
        seen = {}
        # Just in case:
        seen[id(gcl)] = None
        seen[id(olist)] = None
        seen[id(seen)] = None
        # _getr does the real work.
        _getr(gcl, olist, seen)
        return olist

    if backend == 'pytorch':
        objects = get_all_objects()
        for obj in objects:
            try:
                if tc.is_tensor(obj) or (hasattr(obj, 'data') and tc.is_tensor(obj.data)):
                    print(type(obj), obj.shape, obj.device)
            except:
                pass

@set_bn
def no_grad(backend='autograd'):
    if backend == 'pytorch':
        return tc.no_grad()
    else:
        return EmptyWith()

@set_bn
def detach(var, backend='autograd'):
    if backend == 'pytorch':
        var.requires_grad_(False)
        return var
    else:
        return var

@set_bn
def reattach(var, backend='autograd'):
    if backend == 'pytorch':
        var.requires_grad_()
        return var
    else:
        return var

# ________________
# |Maths functions|_____________________________________________________________

@set_bn
def get_dtype(arr, backend='autograd'):
    """
    Get dtype of array in standard string format ('float32', 'float64' etc.)

    :param arr: Tensor.
    :return: Dtype string.
    """
    if backend == 'pytorch':
        return pytorch_dtype_query_mapping_dict[arr.dtype]
    elif backend == 'autograd':
        return str(arr.dtype)

@set_bn
def zeros(shape, dtype=None, device=None, requires_grad=True, backend='autograd'):
    kwargs = {}
    if dtype is not None: kwargs['dtype'] = dtype
    func = getattr(engine_dict[backend], func_mapping_dict['zeros'][backend])
    if backend == 'pytorch':
        if dtype is not None: kwargs['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        arr = func(shape, device=device, requires_grad=requires_grad, **kwargs)
    else:
        arr = func(shape, **kwargs)
    return arr


@set_bn
def ones(shape, dtype=None, device=None, requires_grad=True, backend='autograd'):
    kwargs = {}
    if dtype is not None: kwargs['dtype'] = dtype
    func = getattr(engine_dict[backend], func_mapping_dict['ones'][backend])
    if backend == 'pytorch':
        if dtype is not None: kwargs['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        arr = func(shape, device=device, requires_grad=requires_grad, **kwargs)
    else:
        arr = func(shape, **kwargs)
    return arr


@set_bn
def zeros_like(var, dtype=None, device=None, requires_grad=True, backend='autograd'):
    """
    :param var: ADVariable or tensor.
    """
    kwargs = {}
    if dtype is not None: kwargs['dtype'] = dtype
    func = getattr(engine_dict[backend], func_mapping_dict['zeros_like'][backend])
    if backend == 'pytorch':
        if dtype is not None: kwargs['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        arr = func(var, device=device, requires_grad=requires_grad, **kwargs)
    else:
        arr = func(var, **kwargs)
    return arr


@set_bn
def ones_like(var, dtype=None, device=None, requires_grad=True, backend='autograd'):
    """
    :param var: ADVariable or tensor.
    """
    kwargs = {}
    if dtype is not None: kwargs['dtype'] = dtype
    func = getattr(engine_dict[backend], func_mapping_dict['ones_like'][backend])
    if backend == 'pytorch':
        if dtype is not None: kwargs['dtype'] = getattr(engine_dict['pytorch'], dtype_mapping_dict[dtype]['pytorch'])
        arr = func(var, device=device, requires_grad=requires_grad, **kwargs)
    else:
        arr = func(var, **kwargs)
    return arr


@set_bn
def exp(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['exp'][backend])
    arr = func(var)
    return arr


@set_bn
def log(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['log'][backend])
    arr = func(var)
    return arr


@set_bn
def sign(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['sign'][backend])
    arr = func(var)
    return arr


@set_bn
def sin(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['sin'][backend])
    arr = func(var)
    return arr


@set_bn
def cos(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['cos'][backend])
    arr = func(var)
    return arr


@set_bn
def exp_complex(var_real, var_imag, backend='autograd'):
    if backend == 'pytorch':
        if not isinstance(var_real, tc.Tensor):
            var_real = tc.tensor(var_real)
        if not isinstance(var_imag, tc.Tensor):
            var_real = tc.tensor(var_imag)
    e = exp(var_real)
    return e * cos(var_imag), e * sin(var_imag)


@set_bn
def arange(*args, **kwargs):
    backend = kwargs['backend']
    del kwargs['backend']
    if backend == 'pytorch':
        return tc.arange(*args, **kwargs)
    elif backend == 'autograd':
        return anp.arange(*args, **kwargs)


@set_bn
def abs(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['abs'][backend])
    arr = func(var)
    return arr


@set_bn
def stack(var_list, axis=0, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['stack'][backend])
    arr = func(var_list, axis)
    return arr


@set_bn
def concatenate(var_list, axis=0, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['concatenate'][backend])
    arr = func(var_list, axis)
    return arr


@set_bn
def cast(var, dtype, backend='autograd'):
    dtype = str(dtype)
    if backend == 'autograd':
        return var.astype(dtype)
    elif backend == 'pytorch':
        return getattr(var, dtype_mapping_dict[dtype]['pytorch'])()
    elif backend == 'numpy':
        return var.astype(dtype)


@set_bn
def round(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['round'][backend])
    arr = func(var)
    return arr


@set_bn
def fix(a, backend='autograd'):
    if backend == 'pytorch':
        return tc.trunc(a)
    elif backend == 'autograd':
        return anp.fix(a)


@set_bn
def round_and_cast(var, dtype='int32', backend='autograd'):
    return cast(round(var), dtype=dtype, override_backend=backend)


@set_bn
def fft(var_real, var_imag, axis=-1, backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.fft(var, axis=axis, norm=norm)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.fft(var, signal_ndim=1, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        return var_real[tuple(slicer)], var_imag[tuple(slicer)]


@set_bn
def ifft(var_real, var_imag, axis=-1, backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.ifft(var, axis=axis, norm=norm)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.ifft(var, signal_ndim=1, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        return var_real[tuple(slicer)], var_imag[tuple(slicer)]


@set_bn
def fft2(var_real, var_imag, axes=(-2, -1), backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.fft2(var, axes=axes, norm=norm)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.fft(var, signal_ndim=2, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        return var_real[tuple(slicer)], var_imag[tuple(slicer)]


@set_bn
def ifft2(var_real, var_imag, axes=(-2, -1), backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.ifft2(var, axes=axes, norm=norm)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.ifft(var, signal_ndim=2, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        return var_real[tuple(slicer)], var_imag[tuple(slicer)]


@set_bn
def fft2_and_shift(var_real, var_imag, axes=(-2, -1), backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.fftshift(anp.fft.fft2(var, axes=axes, norm=norm), axes=axes)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.fft(var, signal_ndim=2, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        var_real = var_real[tuple(slicer)]
        var_imag = var_imag[tuple(slicer)]
        var_real = fftshift(var_real, axes=axes)
        var_imag = fftshift(var_imag, axes=axes)
        return var_real, var_imag


@set_bn
def ifft2_and_shift(var_real, var_imag, axes=(-2, -1), backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.fftshift(anp.fft.ifft2(var, axes=axes, norm=norm), axes=axes)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.ifft(var, signal_ndim=2, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        var_real = var_real[tuple(slicer)]
        var_imag = var_imag[tuple(slicer)]
        var_real = fftshift(var_real, axes=axes)
        var_imag = fftshift(var_imag, axes=axes)
        return var_real, var_imag


@set_bn
def ishift_and_ifft2(var_real, var_imag, axes=(-2, -1), backend='autograd', normalize=False):
    if backend == 'autograd':
        var = var_real + 1j * var_imag
        norm = None if not normalize else 'ortho'
        var = anp.fft.ifft2(anp.fft.ifftshift(var, axes=axes), axes=axes, norm=norm)
        return anp.real(var), anp.imag(var)
    elif backend == 'pytorch':
        var_real = ifftshift(var_real, axes=axes)
        var_imag = ifftshift(var_imag, axes=axes)
        var = tc.stack([var_real, var_imag], dim=-1)
        var = tc.ifft(var, signal_ndim=2, normalized=normalize)
        var_real, var_imag = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (len(var_real.shape) - 1) + [0]
        var_real = var_real[tuple(slicer)]
        var_imag = var_imag[tuple(slicer)]
        return var_real, var_imag


@set_bn
def convolve_with_transfer_function(arr_real, arr_imag, h_real, h_imag, axes=(-2, -1), backend='autograd'):
    f_real, f_imag = fft2(arr_real, arr_imag, axes=axes, override_backend=backend)
    fh_real = f_real * h_real - f_imag * h_imag
    fh_imag = f_real * h_imag + f_imag * h_real
    return ifft2(fh_real, fh_imag, override_backend=backend)


@set_bn
def convolve_with_impulse_response(arr_real, arr_imag, h_real, h_imag, axes=(-2, -1), backend='autograd', normalize=True):
    f_real, f_imag = fft2(arr_real, arr_imag, axes=axes, override_backend=backend, normalize=normalize)
    h_real, h_imag = fft2(h_real, h_imag, override_backend=backend, normalize=normalize)
    fh_real = f_real * h_real - f_imag * h_imag
    fh_imag = f_real * h_imag + f_imag * h_real
    return ifft2(fh_real, fh_imag, override_backend=backend, normalize=normalize)


@set_bn
def complex_mul(a_real, a_imag, b_real, b_imag, backend='autograd'):
    return (a_real * b_real - a_imag * b_imag, a_real * b_imag + a_imag * b_real)


@set_bn
def fftshift(var, axes=(1, 2), backend='autograd'):
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


@set_bn
def ifftshift(var, axes=(1, 2), backend='autograd'):
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


@set_bn
def split_channel(var, backend='autograd'):
    if backend == 'autograd':
        var0, var1 = anp.split(var, var.shape[-1], axis=-1)
        slicer = [slice(None)] * (var.ndim - 1) + [0]
        return var0[tuple(slicer)], var1[tuple(slicer)]
    elif backend == 'pytorch':
        var0, var1 = tc.split(var, 1, dim=-1)
        slicer = [slice(None)] * (var.ndim - 1) + [0]
        return var0[tuple(slicer)], var1[tuple(slicer)]
   
@set_bn
def clip(var, a1, a2, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['clip'][backend])
    if backend == 'pytorch':
        if not isinstance(var, tc.Tensor):
            var = tc.tensor(var)
    arr = func(var, a1, a2)
    return arr


@set_bn
def reshape(var, newshape, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['reshape'][backend])
    arr = func(var, newshape)
    return arr


@set_bn
def floor(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['floor'][backend])
    arr = func(var)
    return arr


@set_bn
def floor_and_cast(var, dtype='int32', backend='autograd'):
    return cast(floor(var, override_backend=backend), dtype=dtype, override_backend=backend)


@set_bn
def ceil(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['ceil'][backend])
    arr = func(var)
    return arr


@set_bn
def ceil_and_cast(var, dtype='int32', backend='autograd'):
    return cast(ceil(var, override_backend=backend), dtype=dtype, override_backend=backend)


@set_bn
def sqrt(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['sqrt'][backend])
    arr = func(var)
    return arr


@set_bn
def mean(var, axis=None, backend='autograd'):
    args = {}
    if backend == 'autograd':
        if axis is not None:
            args['axis'] = axis
        return anp.mean(var, **args)
    elif backend == 'pytorch':
        if axis is not None:
            args['dim'] = axis
        return tc.mean(var, **args)


@set_bn
def std(var, backend='autograd'):
    if backend == 'autograd':
        return anp.std(var)
    elif backend == 'pytorch':
        return tc.std(var)


@set_bn
def max(var, return_number=True, axis=None, backend='autograd'):
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


@set_bn
def min(var, return_number=True, axis=None, backend='autograd'):
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


@set_bn
def real(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['real'][backend])
    arr = func(var)
    return arr


@set_bn
def imag(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['imag'][backend])
    arr = func(var)
    return arr


@set_bn
def tile(var, cp, backend='autograd'):
    if backend == 'autograd':
        return anp.tile(var, cp)
    elif backend == 'pytorch':
        return var.repeat(*cp)


@set_bn
def repeat(var, cp, axis=None, backend='autograd'):
    if backend == 'autograd':
        return anp.repeat(var, cp, axis=axis)
    elif backend == 'pytorch':
        return tc.repeat_interleave(var, cp, dim=axis)


@set_bn
def flip(var, axis=[0], backend='autograd'):
    if backend == 'autograd':
        return anp.flip(var, axis=axis)
    elif backend == 'pytorch':
        try:
            _ = len(axis)
            return tc.flip(var, dims=axis)
        except:
            return tc.flip(var, dims=[axis])


@set_bn
def pad(var, pad_len, mode='constant', constant_values=0, backend='autograd'):
    """
    Pad array.
    [ATTENTION: The behavior of this function is different between Autograd and Pytorch backend.]

    :param pad_len: A tuple of tuples. Consistent with the format of numpy.pad.
    :param mode: Choose from 'constant', 'reflect'.
    """
    args = {}
    mode_dict = {'constant': {'autograd': 'constant', 'pytorch': 'constant'},
                 'edge':    {'autograd': 'edge',    'pytorch': 'replicate'},
                 'reflect': {'autograd': 'reflect', 'pytorch': 'reflect'},
                 'wrap':    {'autograd': 'wrap',    'pytorch': 'circular'}}
    if mode == 'constant':
        args['constant_values'] = 0
    if backend == 'autograd':
        return anp.pad(var, pad_len, mode=mode_dict[mode][backend], **args)
    elif backend == 'pytorch':
        pad_len = [x for y in pad_len[::-1] for x in y]
        return tc.nn.functional.pad(var, pad_len, mode=mode_dict[mode][backend], value=constant_values)
    elif backend == 'numpy':
        return np.pad(var, pad_len, mode=mode, **args)


@set_bn
def sum(var, axis=None, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['sum'][backend])
    if backend == 'autograd':
        arr = func(var, axis=axis)
    elif backend == 'pytorch':
        if axis is None:
            arr = tc.sum(var)
        else:
            arr = tc.sum(var, dim=axis)
    return arr


@set_bn
def prod(var, axis=None, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['prod'][backend])
    if backend == 'autograd':
        args = {}
        if axis is not None:
            args['axis'] = axis
        arr = func(var, **args)
    elif backend == 'pytorch':
        args = {}
        if axis is not None:
            args['dim'] = axis
        arr = tc.prod(var, **args)
    return arr


@set_bn
def roll(var, shifts, axes=0, backend='autograd'):
    if backend == 'autograd':
        return anp.roll(var, shifts, axis=axes)
    elif backend == 'pytorch':
        return tc.roll(var, shifts, dims=axes)


@set_bn
def arctan2(var1, var2, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['arctan2'][backend])
    arr = func(var1, var2)
    return arr


@set_bn
def nonzero(var, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['nonzero'][backend])
    arr = func(var)
    return arr


@set_bn
def norm(var_real, var_imag, backend='autograd'):
    if backend == 'autograd':
        return abs(var_real + 1j * var_imag)
    elif backend == 'pytorch':
        return tc.norm(tc.stack([var_real, var_imag], dim=0), dim=0)


@set_bn
def vec_norm(arr, backend='autograd'):
    if backend == 'autograd':
        return anp.sqrt(anp.sum(abs(arr ** 2)))
    elif backend == 'pytorch':
        return tc.sqrt(tc.sum(arr ** 2))


@set_bn
def swap_axes(arr, axes=(0, 1), backend='autograd'):
    if backend == 'autograd':
        temp = [*axes]
        if axes[0] < axes[1]:
            temp = [axes[1], axes[0]]
        axes = []
        for i in range(len(arr.shape)):
            if i == temp[0]:
                axes.append(temp[1])
            elif i == temp[1]:
                axes.append(temp[0])
            else:
                axes.append(i)
        return anp.transpose(arr, axes)
    elif backend == 'pytorch':
        return tc.transpose(arr, axes[0], axes[1])


@set_bn
def permute_axes(arr, axes_order, backend='autograd'):
    if backend == 'autograd':
        return anp.transpose(arr, axes_order)
    elif backend == 'pytorch':
        return arr.permute(axes_order)


@set_bn
def grid_sample(arr, grid, interpolation='bilinear', axis=0, device=None, backend='autograd'):
    """
    :param arr: a stack of 2D images in [N, H, W, C].
    :param grid: [N, 2].
    """
    assert flag_pytorch_avail, 'Wrapper function grid_sample requires Pytorch.'
    flag_convert_arr = False
    if not isinstance(arr, tc.Tensor):
        flag_convert_arr = True
        arr = tc.tensor(arr, requires_grad=False, device=device)
    if not isinstance(grid, tc.Tensor):
        grid = tc.tensor(grid, requires_grad=False, device=device)
    # x coordinates comes first in torch.grid_sample.
    grid = tc.flip(grid, (1,))

    axis_arrangement = [0, 1, 2, 3]
    # Move channel to the 2nd dimension.
    axis_arrangement[1], axis_arrangement[3] = axis_arrangement[3], axis_arrangement[1]
    # Move invariant axis to front.
    if axis != 0:
        q = axis_arrangement.index(axis)
        axis_arrangement[0], axis_arrangement[q] = axis_arrangement[q], axis_arrangement[0]
    if axis_arrangement[2] > axis_arrangement[3]:
        axis_arrangement[2], axis_arrangement[3] = axis_arrangement[3], axis_arrangement[2]
    arr = permute_axes(arr, axis_arrangement, override_backend='pytorch')

    # Convert grid to [-1, 1] scale.
    # arr_center = (tc.tensor(arr.shape[2:4], requires_grad=False, device=device) - 1) / 2
    # grid = (grid - arr_center) / (arr_center + 0.5)
    arr_shape = create_constant(arr.shape[2:], dtype=pytorch_dtype_query_mapping_dict[arr.dtype],
                                device=get_var_device(arr))
    grid = -1 + 2. * grid / arr_shape + 1. / arr_shape

    grid = reshape(grid, [1, *arr.shape[2:4], 2], override_backend='pytorch')
    grid = tile(grid, [arr.shape[0], 1, 1, 1], override_backend='pytorch')
    grid = cast(grid, pytorch_dtype_query_mapping_dict[arr.dtype], override_backend='pytorch')
    arr = tc.nn.functional.grid_sample(arr, grid, padding_mode='border', mode=interpolation, align_corners=False)
    arr = permute_axes(arr, [axis_arrangement.index(0), axis_arrangement.index(1),
                             axis_arrangement.index(2), axis_arrangement.index(3)], override_backend='pytorch')
    if flag_convert_arr:
        arr = arr.data.numpy()
    return arr


@set_bn
def matmul(a, b, backend='autograd'):
    if backend == 'autograd':
        return anp.matmul(a, b)
    elif backend == 'pytorch':
        return tc.matmul(a, b)


@set_bn
def affine_transform(arr, transform, backend='autograd'):
    """
    :param arr: a stack of 2D images in [N, H, W].
    :param transform: A [2, 3] matrix for affine transform.
    """
    if backend == 'autograd':
        raise NotImplementedError('Rescaling in Autograd is not yet implemented. Use Pytorch backend instead.')
    elif backend == 'pytorch':
        n = arr.shape[0]
        arr_size = arr.shape[1:]
        m = reshape(transform, [-1, 2, 3], override_backend=backend)
        m = cast(tile(m, [n, 1, 1], override_backend=backend), pytorch_dtype_query_mapping_dict[arr.dtype], override_backend=backend)
        g = tc.nn.functional.affine_grid(m, [n, 1, *arr_size])
        arr_new = tc.reshape(arr, [n, 1, *arr.shape[1:]])
        arr_new = tc.nn.functional.grid_sample(arr_new, g, padding_mode='border')
        return arr_new[:, 0, :, :]


@set_bn
def rotate(arr, theta, axis=0, backend='autograd', device=None):
    """
    A rotate function that allows taking gradient with regards to theta.

    :param arr: a 3D object in [len_y, len_x, len_z, n_channels].
    """
    if backend == 'autograd':
        warnings.warn('Rotate (with grad) in Autograd is not yet implemented. Use Pytorch backend instead.')
        axes = []
        for i in range(3):
            if i != axis:
                axes.append(i)
        return scipy.ndimage.rotate(arr, -anp.rad2deg(theta), reshape=False, axes=axes, mode='nearest', order=1)
    elif backend == 'pytorch':
        try:
            theta = theta.view(1)
        except:
            theta = tc.tensor(theta, requires_grad=False, device=device)
            theta = theta.view(1)
        axis_arrangement = [0, 1, 2, 3]
        # Move channel to the 2nd dimension.
        axis_arrangement[1], axis_arrangement[3] = axis_arrangement[3], axis_arrangement[1]
        # Move invariant axis to front.
        if axis != 0:
            q = axis_arrangement.index(axis)
            axis_arrangement[0], axis_arrangement[q] = axis_arrangement[q], axis_arrangement[0]
        if axis_arrangement[2] < axis_arrangement[3]:
            theta = -theta
        arr = permute_axes(arr, axis_arrangement, override_backend='pytorch')
        naught = cast(tc.tensor([0.], device=device), pytorch_dtype_query_mapping_dict[theta.dtype], override_backend='pytorch')
        m0 = tc.cat([tc.cos(theta), -tc.sin(theta), naught])
        m1 = tc.cat([tc.sin(theta), tc.cos(theta), naught])
        m = tc.stack([m0, m1]).view(1, 2, 3)
        m = cast(tile(m, [arr.shape[0], 1, 1], override_backend='pytorch'), pytorch_dtype_query_mapping_dict[arr.dtype], override_backend='pytorch')
        g = tc.nn.functional.affine_grid(m, arr.shape, align_corners=False)

        arr = tc.nn.functional.grid_sample(arr, g, padding_mode='border', align_corners=False)
        arr = permute_axes(arr, [axis_arrangement.index(0), axis_arrangement.index(1),
                                 axis_arrangement.index(2), axis_arrangement.index(3)], override_backend='pytorch')
        return arr


@set_bn
def pcc(obj, backend='autograd'):
    """
    Calculate the Pearson correlation coefficient of images in an array along the last dimension.
    :param obj: Tensor. 
    :return: Pearson correlation coefficient.
    """
    slicer_z = [slice(None)] * (len(obj.shape) - 1)
    for i_slice in range(obj.shape[-1]):
        if i_slice == 0:
            nom = obj[slicer_z + [i_slice]] - mean(obj[slicer_z + [i_slice]])
            denom = std(obj[slicer_z + [i_slice]])
        else:
            nom = nom * (obj[slicer_z + [i_slice]] - mean(obj[slicer_z + [i_slice]]))
            denom = denom * std(obj[slicer_z + [i_slice]])
    nom = sum(nom)
    return abs(nom / denom)


@set_bn
def tomography_filter(arr, axis=2, filter_type='hamming', backend='autograd'):
    """
    Apply a 1D ramp filter needed for tomography reconstruction.

    :param arr: Data array.
    :param axis: Axis of slice projection.
    :return:
    """
    func = getattr(scipy.signal.windows, filter_type)
    filter = func(arr.shape[axis])
    if axis != len(arr.shape) - 1:
        arr = swap_axes(arr, [axis, len(arr.shape) - 1])
    if backend == 'pytorch':
        args = {'device': arr.device}
    else:
        args = {}
    arr_r, arr_i = fft(arr, zeros_like(arr, requires_grad=False, **args))
    arr_r = arr_r * filter
    arr_i = arr_i * filter
    arr, _ = ifft(arr_r, arr_i)
    if axis != len(arr.shape) - 1:
        arr = swap_axes(arr, [axis, len(arr.shape) - 1])
    return arr


@set_bn
def argmax(arr, backend='autograd'):
    func = getattr(engine_dict[backend], func_mapping_dict['argmax'][backend])
    arr = func(arr)
    return arr


@set_bn
def tensordot(a, b, axes=None, backend='autograd'):
    """
    :param axes: Comply to Numpy format.
    """
    dims = axes
    if backend == 'pytorch':
        if isinstance(axes, (list, tuple)):
            if isinstance(axes[0], int):
                dims = []
                for i in axes:
                    dims.append((axes[i],))
        return tc.tensordot(a, b, dims=dims)
    elif backend == 'autograd':
        return anp.tensordot(a, b, axes=dims)


@set_bn
def isnan(arr, backend='autograd'):
    if backend == 'pytorch':
        return tc.isnan(arr)
    elif backend == 'autograd':
        return anp.isnan(arr)
