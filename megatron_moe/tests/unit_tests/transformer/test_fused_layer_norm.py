"""
Compare the results between FusedLayerNorm and Torch version.
"""
import os
import sys
import pdb

sys.path.insert(0, "/path/to/megatron")
import megatron

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApexLayerNorm(megatron.model.LayerNorm):
    # FusedLayerNormAffineFunction -> fused_layer_norm_cuda.forward_affine
    #   -> layer_norm_affine -> cuda_layer_norm
    def __init__(self,
                 normalized_shape,
                 eps=1e-5,
                 no_persist_layer_norm=False):
        super(ApexLayerNorm, self).__init__(normalized_shape,
                                            eps,
                                            no_persist_layer_norm)


class TorchLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5):
        super(TorchLayerNorm, self).__init__(normalized_shape,
                                             eps=eps)

    def _mean(self, x):
        return x.mean(-1, keepdim=True)

    def _var(self, x):
        return torch.rsqrt(torch.var(x, dim=-1, unbiased=False, keepdim=True) + self.eps)

    def forward(self, input):
        if False:
            out = F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)
            return out

        _m = self._mean(input.float())
        _v = self._var(input.float())
        output = (input.float() - _m) * _v
        y = self.weight * output.type_as(input) + self.bias
        return y


if __name__ == "__main__":
    normalized_shape = 5120
    eps = 1e-5

    torch.manual_seed(666)
    weight = torch.rand(normalized_shape).bfloat16()
    bias = torch.rand(normalized_shape).bfloat16()
    weight *= 10000
    bias *= 1000

    x = torch.rand(32, normalized_shape).bfloat16()
    x = x.cuda()

    ln_apex = ApexLayerNorm(normalized_shape, eps=eps).cuda()
    ln_apex = ln_apex.bfloat16()
    ln_torch = TorchLayerNorm(normalized_shape, eps=eps).cuda()
    ln_torch = ln_torch.bfloat16()

    ln_apex.weight.data.copy_(weight)
    ln_apex.bias.data.copy_(bias)
    ln_torch.weight.data.copy_(weight)
    ln_torch.bias.data.copy_(bias)

    y_apex = ln_apex(x)
    y_torch = ln_torch(x)
    diff = (y_apex != y_torch).sum(dim=-1)
    print(f"diff: {diff}")
    pdb.set_trace()
