# coding:utf-8

import os
import torch


def set_dump_path(fpath=None):
    assert fpath is not None
    os.environ["DUMP_PATH"] = fpath


def get_dump_path():
    dump_path = os.environ.get("DUMP_PATH", None)
    assert dump_path is not None and len(dump_path) > 0, "Please set dump path for hook tools."
    os.makedirs(dump_path, exist_ok=True)
    return dump_path


def dump_tensor(x, prefix=""):
    if isinstance(x, (tuple, list)) and x:
        for i, item in enumerate(x):
            dump_tensor(item, prefix="{}.{}".format(prefix, i))
    elif isinstance(x, torch.Tensor):
        if not hasattr(dump_tensor, 'dump_init_enable'):
            dump_tensor.call_number = 0
            dump_tensor.dump_init_enable = False
        else:
            dump_tensor.call_number = dump_tensor.call_number + 1
        prefix = f"{str(dump_tensor.call_number).zfill(5)}_{prefix}"
        torch.save(x.cpu(), os.path.join(get_dump_path(), prefix))


def warp_dump_hook(name):
    def dump_hook(module, in_feat, out_feat):
        name_template = f"{name}_{module.__class__.__name__}"+ "_{}"
        dump_tensor(in_feat, name_template.format("input"))
        dump_tensor(out_feat, name_template.format("output"))

    return dump_hook


def register_fwd_hook(model):
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        module.register_forward_hook(warp_dump_hook("fwd"))


def register_bwd_hook(model):
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        module.register_backward_hook(warp_dump_hook("bwd"))


def register_all_hooks(model):
    for _, module in model.named_modules():
        if not hasattr(module, "named_modules") or len(list(module.named_modules())) > 1:
            continue

        module.register_forward_hook(warp_dump_hook("fwd"))
        module.register_backward_hook(warp_dump_hook("bwd"))
