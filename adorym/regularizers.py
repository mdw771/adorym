import adorym.wrappers as w
from adorym.util import *


class Regularizer(object):
    """
    Parent regularizer class.

    :param unknown_type: String. Can be ``'delta_beta'`` or ``'real_imag'``.
    :param device: Device object or ``None``.
    """
    def __init__(self, unknown_type='delta_beta'):
        self.unknown_type = unknown_type

    def get_value(self, obj, device=None, **kwargs):
        pass


class L1Regularizer(Regularizer):
    """
    L1-norm regularizer.

    :param alpha_d: Weight of l1-norm of delta or real part.
    :param alpha_b: Weight of l1-norm of beta or imaginary part.
    """

    def __init__(self, alpha_d, alpha_b, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.alpha_d = alpha_d
        self.alpha_b = alpha_b

    def get_value(self, obj, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            if self.alpha_d not in [None, 0]:
                reg = reg + self.alpha_d * w.mean(w.abs(obj[slicer + [0]]))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(w.abs(obj[slicer + [1]]))
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            if self.alpha_d not in [None, 0]:
                om = w.sqrt(r ** 2 + i ** 2)
                reg = reg + self.alpha_d * w.mean(w.abs(om - w.mean(om)))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(w.abs(w.arctan2(i, r)))
        return reg


class ReweightedL1Regularizer(Regularizer):
    """
    Reweighted l1-norm regularizer.

    :param alpha_d: Weight of l1-norm of delta or real part.
    :param alpha_b: Weight of l1-norm of beta or imaginary part.
    """
    def __init__(self, alpha_d, alpha_b, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.alpha_d = alpha_d
        self.alpha_b = alpha_b
        self.weight_l1 = None

    def update_l1_weight(self, weight_l1):
        self.weight_l1 = weight_l1

    def get_value(self, obj, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            if self.alpha_d not in [None, 0]:
                reg = reg + self.alpha_d * w.mean(self.weight_l1[slicer + [0]] * w.abs(obj[slicer + [0]]))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(self.weight_l1[slicer + [1]] * w.abs(obj[slicer + [1]]))
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            wr = self.weight_l1[slicer + [0]]
            wi = self.weight_l1[slicer + [1]]
            wm = wr ** 2 + wi ** 2
            if self.alpha_d not in [None, 0]:
                om = w.sqrt(r ** 2 + i ** 2)
                reg = reg + self.alpha_d * w.mean(wm * w.abs(om - w.mean(om)))
            if self.alpha_b not in [None, 0]:
                reg = reg + self.alpha_b * w.mean(wm * w.abs(w.arctan2(i, r)))
        return reg


class TVRegularizer(Regularizer):
    """
    Total variation regularizer.

    :param gamma: Weight of TV term.
    """
    def __init__(self, gamma, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.gamma = gamma

    def get_value(self, obj, distribution_mode=None, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            o1 = obj[slicer + [0]]
            o2 = obj[slicer + [1]]
            axis_offset = 0 if distribution_mode is None else 1
            reg = reg + self.gamma * total_variation_3d(o1, axis_offset=axis_offset)
            reg = reg + self.gamma * total_variation_3d(o2, axis_offset=axis_offset)
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            axis_offset = 0 if distribution_mode is None else 1
            reg = reg + self.gamma * total_variation_3d(r ** 2 + i ** 2, axis_offset=axis_offset)
            reg = reg + self.gamma * total_variation_3d(w.arctan2(i, r), axis_offset=axis_offset)
        return reg


class CorrRegularizer(Regularizer):
    """
    Pearson correlation regularizer along z axis of object.

    :param gamma: Weight of correlation term.
    """
    def __init__(self, gamma, unknown_type='delta_beta'):
        super().__init__(unknown_type)
        self.gamma = gamma

    def get_value(self, obj, distribution_mode=None, device=None, **kwargs):
        slicer = [slice(None)] * (len(obj.shape) - 1)
        slicer_z = [slice(None)] * (len(obj.shape) - 2)
        reg = w.create_variable(0., device=device)
        if self.unknown_type == 'delta_beta':
            o1 = obj[slicer + [0]]
            o2 = obj[slicer + [1]]
            axis_offset = 2 if distribution_mode is None else 3
        elif self.unknown_type == 'real_imag':
            r = obj[slicer + [0]]
            i = obj[slicer + [1]]
            axis_offset = 0 if distribution_mode is None else 1
            o1 = w.sqrt(r ** 2 + i ** 2)
            o2 = w.arctan2(i, r)
        else:
            raise ValueError('Invalid value for unknown_type.')
        for i_slice in range(o1.shape[axis_offset]):
            if i_slice == 0:
                nom_1 = o1[slicer_z + [i_slice]] - w.mean(o1[slicer_z + [i_slice]])
                nom_2 = o2[slicer_z + [i_slice]] - w.mean(o2[slicer_z + [i_slice]])
                denom_1 = w.std(o1[slicer_z + [i_slice]])
                denom_2 = w.std(o2[slicer_z + [i_slice]])
            else:
                nom_1 = nom_1 * (o1[slicer_z + [i_slice]] - w.mean(o1[slicer_z + [i_slice]]))
                nom_2 = nom_2 * (o2[slicer_z + [i_slice]] - w.mean(o2[slicer_z + [i_slice]]))
                denom_1 = denom_1 * w.std(o1[slicer_z + [i_slice]])
                denom_2 = denom_2 * w.std(o2[slicer_z + [i_slice]])
        nom_1 = w.sum(nom_1)
        nom_2 = w.sum(nom_2)
        reg = reg + self.gamma * w.abs(nom_1 / denom_1)
        reg = reg + self.gamma * w.abs(nom_2 / denom_2)
        return reg