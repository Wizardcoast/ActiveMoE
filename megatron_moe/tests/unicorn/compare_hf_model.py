import os
import glob
import argparse
from collections import OrderedDict
import torch
import pdb
import re
from safetensors import safe_open


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-model", required=True, type=str,
                        help="Path to source hf model.")
    parser.add_argument("--dst-model", required=True, type=str,
                        help="Path to destination hf model.")

    return parser.parse_args()


def load_hf_model(load_path):
    model_files = glob.glob(os.path.join(load_path, "*.bin"))
    if len(model_files) == 0:
        model_files = glob.glob(os.path.join(load_path, "*.safetensors"))
    assert len(model_files) > 0
    model_files = sorted(model_files)

    state_dicts = OrderedDict()
    state_dicts["path"] = load_path
    for f in model_files:
        if f.endswith(".bin"):
            _state = torch.load(f, "cpu")
        elif f.endswith(".safetensors"):
            _state = OrderedDict()
            with safe_open(f, framework="pt", device="cpu") as fd:
                for k in fd.keys():
                    _state[k] = fd.get_tensor(k)
        else:
            raise NotImplementedError("=> Error!")
        state_dicts.update(_state)

    return state_dicts


def compare_state_dicts(src_state_dict, dst_state_dict):
    print(f"=> Comparing SRC[{src_state_dict['path']}] to DST[{dst_state_dict['path']}] ...")

    src_keys = set(src_state_dict.keys())
    dst_keys = set(dst_state_dict.keys())
    same_keys = src_keys.intersection(dst_keys)
    src_only_keys = src_keys.difference(same_keys)
    dst_only_keys = dst_keys.difference(same_keys)
    if len(src_only_keys) > 0:
        print(f"=> Only these keys in src_model: [{src_state_dict['path']}]: "
              f"== {src_only_keys}")
    if len(dst_only_keys) > 0:
        print(f"=> Only these keys in dst_model: [{dst_state_dict['path']}]: "
              f"== {dst_only_keys}")

    # delete path in same_keys
    same_keys.remove('path')
    for key in sorted(same_keys):
        src_tensor = src_state_dict[key]
        dst_tensor = dst_state_dict[key]

        if src_tensor.dtype == dst_tensor.dtype and \
                src_tensor.shape == dst_tensor.shape:
            diff_cnt = (src_tensor != dst_tensor).sum()
        else:
            diff_cnt = -1
            diff_str = " & ".join([f"dtype {src_tensor.dtype} vs. {dst_tensor.dtype}",
                                   f"shape {src_tensor.shape} vs. {dst_tensor.shape}"])

        if diff_cnt < 0:
            print(f"=> {key}: {diff_str} ...")
        elif diff_cnt > 0:
            print(f"=> {key}: total {diff_cnt} different elements ...")


def main():
    args = parse_args()
    src_state_dict = load_hf_model(args.src_model)
    dst_state_dict = load_hf_model(args.dst_model)
    _ = compare_state_dicts(src_state_dict, dst_state_dict)


if __name__ == "__main__":
    main()
