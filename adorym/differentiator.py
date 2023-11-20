import torch

try:
    try:
        import intel_extension_for_pytorch as ipex
    except:
        import ipex
except:
    pass
# backward compatibility
try:
    import torch_ipex
except:
    pass

import numpy as np

import adorym
import adorym.wrappers as w

class Differentiator(object):

    def __init__(self):
        self.loss_object = None
        self.opt_args_ls = []
        self.loss_args = {}

    def create_loss_node(self, loss, opt_args_ls=None):
        """
        :param loss: The loss object (loss node in PyTorch, or loss function handle in Autograd).
        :param opt_args_ls: List of Int. A list of indices of arguments in forward_model.get_loss_function/predict to 
                            which the gradient should be calculated.
        """
        self.loss_object = w.prepare_loss_node(loss, opt_args_ls)
        self.opt_args_ls = opt_args_ls

    def get_gradients(self, **kwargs):
        """
        :param kwargs: Unwrapped dictionary or key word arguments that contain all optimizable variables.
        """
        gradients = w.get_gradients(self.loss_object, opt_args_ls=self.opt_args_ls, **kwargs)
        return gradients

    def get_l_h_hessian_and_h_x_jacobian_mvps(self, forward_model, ind_opt_arg, **kwargs):
        """
        Create functions to compute the matrix-vector product of loss-predict-Hessian and predict-object-Jacobian
        with any arbitrary vector in its argument.
        The predict function of forward_model must return detected **magnitudes**.
        :param forward_model: adorym.ForwardModel object.
        :param ind_opt_arg: List of Int. A list of indices of arguments in forward_model.get_loss_function to
                            which the gradient should be calculated.
        :param kwargs: Unwrapped dictionary or key word arguments that contain ALL arguments of forward_model.get_loss_function/predict.
        """
        assert isinstance(forward_model, adorym.ForwardModel)
        self.func_vjp, _ = w.vjp(forward_model.predict, [ind_opt_arg])(*list(kwargs.values()))
        self.func_jvp = w.jvp(forward_model.predict, [ind_opt_arg])(*list(kwargs.values()))
        #if forward_model.loss_function_type == 'lsq':
        #    self.func_hvp = lambda x: x
        #    self.jloss = 2 * forward_model.predict(**kwargs)
        #else:
        #    self.func_hvp, self.jloss = w.hvp(forward_model.get_mismatch_loss, [0])(forward_model.this_pred_batch, forward_model.this_prj_batch)

        # Calculate HVP of loss using predicted and measured data.
        obj = kwargs['obj']
        this_pred_batch = forward_model.predict(**kwargs)
        this_prj_batch = forward_model.get_data(kwargs['this_i_theta'], kwargs['this_ind_batch'], 
                                               theta_downsample=forward_model.common_vars['theta_downsample'],
                                               ds_level=forward_model.common_vars['ds_level'])
        self.func_hvp, self.jloss = w.hvp(forward_model.loss, [0])(this_pred_batch, this_prj_batch, obj)

        # GVP is Gauss-Newton-vector product.
        def f_gvp(g):
            g = self.func_jvp([g])
            g = self.func_hvp(g)
            g = self.func_vjp(g)[0]
            return g
        self.func_gvp = f_gvp
        self.full_grad = self.func_vjp(self.jloss)[0]
        # gradients = w.get_gradients(self.loss_object, opt_args_ls=[0], **kwargs) # self.full_grad should match gradients
     
