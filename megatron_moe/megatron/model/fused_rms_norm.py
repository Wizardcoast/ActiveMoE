# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""This code is copied fron NVIDIA apex:
      https://github.com/NVIDIA/apex
   with some changes. """

import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import init
import importlib
import inspect

from apex.normalization.fused_layer_norm import FusedRMSNormAffineFunction, FusedRMSNormFunction


global fused_layer_norm_cuda
fused_layer_norm_cuda = None


class MixedFusedRMSNorm(torch.nn.Module):
    r"""Applies RMS Normalization over a mini-batch of inputs
    Currently only runs on cuda() tensors.
    .. math::
        y = \frac{x}{\mathrm{RMS}[x]} * \gamma
    The root-mean-square is calculated separately over the last
    certain number dimensions which have to be of the shape specified by
    :attr:`normalized_shape`.
    :math:`\gamma` is a learnable affine transform parameter of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    `epsilon` is added to the mean-square, then the root of the sum is taken.
    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, RMS Normalization applies per-element scale
        with :attr:`elementwise_affine`.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size
            .. math::
                [* \times \text{normalized}\_\text{shape}[0] \times \text{normalized}\_\text{shape}[1]
                    \times \ldots \times \text{normalized}\_\text{shape}[-1]]
            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)
    Examples::
        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = FusedRMSNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = FusedRMSNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = FusedRMSNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = FusedRMSNorm(10)
        >>> # Activating the module
        >>> output = m(input)
    .. _`Root Mean Square Layer Normalization`: https://arxiv.org/pdf/1910.07467.pdf
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 elementwise_affine=True,
                 sequence_parallel=False,
                 mem_efficient_ln=True,
                 **kwargs):
        super(MixedFusedRMSNorm, self).__init__()

        global fused_layer_norm_cuda
        fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")

        self.mem_efficient_ln = mem_efficient_ln

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()
        self.sequence_parallel = sequence_parallel

        # set sequence parallelism flag on weight parameters
        setattr(self.weight, "sequence_parallel", self.sequence_parallel)

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)

    def forward(self, input):
        if self.elementwise_affine:
            if 'memory_efficient' in inspect.getfullargspec(FusedRMSNormAffineFunction.forward).args:
                return FusedRMSNormAffineFunction.apply(input, self.weight,
                                                        self.normalized_shape, self.eps, self.mem_efficient_ln)
            else:
                return FusedRMSNormAffineFunction.apply(input, self.weight, self.normalized_shape, self.eps)
        else:
            if 'memory_efficient' in inspect.getfullargspec(FusedRMSNormFunction.forward).args:
                return FusedRMSNormFunction.apply(input, self.normalized_shape, self.eps, self.mem_efficient_ln)
            else:
                return FusedRMSNormFunction.apply(input, self.normalized_shape, self.eps)
