import os
import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gqa-load",
        type=str,
        required=True,
        help="Path to the GQA Megatron checkpoint.",
    )
    parser.add_argument(
        "--mha-save",
        type=str,
        required=True,
        help="Path to the MHA Megatron checkpoint.",
    )

    args = parser.parse_args()

    os.makedirs(args.mha_save, exist_ok=True)
    assert not os.path.samefile(os.path.abspath(args.gqa_load),
                                os.path.abspath(args.mha_save)), "GQA & MHA should have different path."

    return args


def convert_gqa_to_mha(args):
    # Load Megatron-LM checkpoint arguments from the state dict
    sub_dirs = os.listdir(args.gqa_load)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    rank0_checkpoint_path = None
    for subdir in possible_sub_dirs:
        if subdir in sub_dirs:
            rank0_checkpoint_path = os.path.join(args.gqa_load, subdir, "model_optim_rng.pt")
            break
    assert rank0_checkpoint_path is not None
    print(f"Loading Megatron-LM checkpoint from: {rank0_checkpoint_path}")

    state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
    megatron_args = state_dict.get("args", None)
    if megatron_args is None:
        raise ValueError(
            "Megatron-LM checkpoint does not contain arguments. This utility only supports Megatron-LM checkpoints"
            " containing all the megatron arguments. This is because it loads all config related to model"
            " architecture, the tensor and pipeline model parallel size from the checkpoint instead of user having to"
            " manually specify all the details. Please save Megatron-LM checkpoint along with all the megatron"
            " arguments to use this utility."
        )

    for subdir in sub_dirs:
        gqa_checkpoint_path = os.path.join(args.gqa_load, subdir, "model_optim_rng.pt")
        mha_checkpoint_path = os.path.join(args.mha_save, subdir, "model_optim_rng.pt")
        os.makedirs(os.path.join(args.mha_save, subdir), exist_ok=True)

        state_dict = torch.load(gqa_checkpoint_path, map_location='cpu')
        select_keys = [k for k in state_dict['model']['language_model']['encoder']
                       if 'self_attention.query_key_value' in k]

        hidden_size = megatron_args.hidden_size
        kv_channels = megatron_args.kv_channels
        for k in select_keys:
            param = state_dict['model']['language_model']['encoder'][k]
            last_dim = hidden_size
            if 'bias' in k:
                param = param.view(-1, 1)
                last_dim = 1

            num_query_groups_pp = megatron_args.num_query_groups // megatron_args.tensor_model_parallel_size
            param = param.view(-1, kv_channels, last_dim)
            param = param.view(num_query_groups_pp, -1, kv_channels, last_dim)
            query, key, value = torch.split(param, (param.size(1)-2, 1, 1), dim=1)
            group_size = query.size(1)
            query = query.reshape(-1, kv_channels, last_dim)
            key = key.repeat(1, group_size, 1, 1).view(-1, kv_channels, last_dim)
            value = value.repeat(1, group_size, 1, 1).view(-1, kv_channels, last_dim)
            new_param = torch.cat((query, key, value), dim=1).view(-1, last_dim)
            if last_dim == 1:
                new_param = new_param.squeeze(-1)

            state_dict['model']['language_model']['encoder'][k] = new_param

        torch.save(state_dict, mha_checkpoint_path)


if __name__ == "__main__":
    args = get_args()
    convert_gqa_to_mha(args)
