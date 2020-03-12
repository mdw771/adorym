import numpy as np

import adorym.wrappers as w

class Differentiator(object):

    def __init__(self):
        self.loss_object = None
        self.opt_args_ls = []

    def create_loss_node(self, loss, opt_args_ls=None):
        self.loss_object = w.prepare_loss_node(loss, opt_args_ls)
        self.opt_args_ls = opt_args_ls

    def get_gradients(self, **kwargs):
        gradients = w.get_gradients(self.loss_object, opt_args_ls=self.opt_args_ls, **kwargs)
        return gradients
