# encoding:utf-8

import os
import torch
import torch.nn as nn
from torchvision import models

from megatron.core.profiler import register_all_hooks, set_dump_path


def test_module_hooks():
    model = models.resnet50().cuda()
    set_dump_path("./test_dump_tensor")

    register_all_hooks(model)
    inputs = torch.randn(1, 3, 244, 244).cuda()
    labels = torch.randn(1).long().cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    output = model(inputs)
    loss = criterion(output, labels)
    loss.backward()
    try:
        assert len(os.listdir("./test_dump_tensor")) == 0
    finally:
        for filename in os.listdir('./test_dump_tensor'):
            os.remove('./test_dump_tensor/' + filename)
        os.rmdir("./test_dump_tensor")


if __name__ == '__main__':
    test_module_hooks()
