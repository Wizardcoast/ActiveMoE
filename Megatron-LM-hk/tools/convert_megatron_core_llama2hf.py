import argparse
import os
from collections import OrderedDict
from typing import Any, Mapping

import accelerate
import torch
from accelerate.utils import set_module_tensor_to_device
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from torch.nn import Module
import shutil

# import debugpy
# debugpy.listen(('0.0.0.0', 5678))
# # debugpy.breakpoint()
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()

transformer_layer_name_list = {
    "input_layernorm": [
        "input_layernorm.weight",
        "self_attention.layernorm_qkv.layer_norm_weight",
        "self_attention.linear_qkv.layer_norm_weight"
    ],
    "query_key_value": [
        "self_attention.query_key_value.weight",
        "self_attention.layernorm_qkv.weight",
        "self_attention.linear_qkv.weight"
    ],
    "query": ["self_attention.query.weight"],
    "key_value": ["self_attention.key_value.weight"],
    "o_proj": ["self_attention.dense.weight", "self_attention.proj.weight","self_attention.linear_proj.weight"],
    "mlp_gate_up": ["mlp.dense_h_to_4h.weight", "layernorm_mlp.fc1_weight","mlp.linear_fc1.weight"],
    "mlp_down": ["mlp.dense_4h_to_h.weight", "layernorm_mlp.fc2_weight","mlp.linear_fc2.weight"],
    "post_attention_layernorm": [
        "post_attention_layernorm.weight",
        "layernorm_mlp.layer_norm_weight",
        "mlp.linear_fc1.layer_norm_weight"
    ],
}


def recursive_print(name, val, spaces=0):
    # Format the message.
    if name is None:
        msg = None
    else:
        fmt = "." * max(0, spaces - 2) + "# {:" + str(50 - spaces) + "s}"
        msg = fmt.format(name)

    # Print and recurse (if needed).
    if isinstance(val, dict):
        if msg is not None:
            print(msg)
        for k in val.keys():
            recursive_print(k, val[k], spaces + 2)
    elif isinstance(val, torch.Tensor):
        print(msg, ":", val.size())
    else:
        print(msg, ":", val)


def get(dicts, key):
    return [dict.pop(key) for dict in dicts]


def check_get(dicts, prefix, key_list):
    return [
        dict.pop(prefix + key)
        for dict in dicts
        for key in key_list
        if prefix + key in dict
    ]


def check_assign(model_weight, this_layer_index, this_model_weight, layer_index, key_list):
    for key in key_list:
        full_key = f"decoder.layers.{layer_index}." + key
        if full_key in this_model_weight:
            model_weight[f"decoder.layers.{this_layer_index}." + key] = this_model_weight[full_key]
            break
    return model_weight


def merge_col(tensors):
    return torch.cat(
        [
            tensor["weight"] if type(tensor) is OrderedDict else tensor
            for tensor in tensors
        ],
        dim=0,
    )


def merge_row(tensors):
    return torch.cat(
        [
            tensor["weight"] if type(tensor) is OrderedDict else tensor
            for tensor in tensors
        ],
        dim=1,
    )


def load_state_dict_meta(self, state_dict: Mapping[str, Any], strict: bool = True):
    for key, val in state_dict.items():
        set_module_tensor_to_device(self, key, val.device, val.clone(), val.dtype)


def convert_megatron_checkpoint(
    hf_model, state_dicts, model_config: LlamaConfig, train_args 
):
    # The model.
    models = get(state_dicts, "model")

    # # The language model.
    # lms = get(models, "language_model")

    # # The embeddings.
    # embeddings = get(lms, "embedding")

    # The word embeddings.
    word_embeddings = get(models, "embedding.word_embeddings.weight")

    # Truncate the embedding table to vocab_size rows.
    merged_padded_word_embeddings = merge_col(word_embeddings)
    merged_word_embeddings = merged_padded_word_embeddings[: model_config.vocab_size, :]
    hf_model.model.embed_tokens.load_state_dict(
        {"weight": merged_word_embeddings}, strict=True
    )

    # The transformer.
    transformers = models
    train_args.use_mcore_models = True

    for i in range(model_config.num_hidden_layers):
        print(f"Converting layer-{i}", flush=True)
        prefix = f"decoder.layers.{i}."
        layer: LlamaDecoderLayer = hf_model.model.layers[i]

        layer.input_layernorm.load_state_dict(
            {
                "weight": check_get(
                    transformers, prefix, transformer_layer_name_list["input_layernorm"]
                )[0]
            },
            strict=True,
        )

        hidden_size = model_config.hidden_size
        inter_size = model_config.intermediate_size
        num_heads = model_config.num_attention_heads
        kv_heads = model_config.num_key_value_heads
        kv_hidden_size = hidden_size // num_heads * kv_heads
        if num_heads == kv_heads:
            qkv = merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["query_key_value"]
                )
            )
            qkv = qkv.view(num_heads, 3, hidden_size // num_heads, hidden_size)
            q, k, v = torch.chunk(qkv, 3, dim=1)
            q, k, v = (
                q.reshape(hidden_size, hidden_size),
                k.reshape(hidden_size, hidden_size),
                v.reshape(hidden_size, hidden_size),
            )
        elif train_args.use_mcore_models:
            # megatron
            qkv = merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["query_key_value"]
                )
            )

            # qkv_weight_interleaved:
            # ng: kv_heads
            # np: num_heads
            # hn: head_dim = 128
            # qkv_weight: [ng, (np/ng + 2), hn, h]
            num_queries_per_key_value = num_heads // kv_heads
            qkv = qkv.view(
                kv_heads,
                num_queries_per_key_value + 2,
                hidden_size // num_heads,
                hidden_size,
            )
            q, k, v = torch.split(qkv, [num_queries_per_key_value, 1, 1], dim=1)
            q, k, v = (
                q.reshape(hidden_size, hidden_size),
                k.reshape(kv_hidden_size, hidden_size),
                v.reshape(kv_hidden_size, hidden_size),
            )
        else:
            # megatron without transformer engine
            q = merge_col(
                check_get(transformers, prefix, transformer_layer_name_list["query"])
            )
            kv = merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["key_value"]
                )
            )
            kv = kv.view(kv_heads, 2, hidden_size // num_heads, hidden_size)
            k, v = torch.chunk(kv, 2, dim=1)
            q, k, v = (
                q.reshape(hidden_size, hidden_size),
                k.reshape(kv_hidden_size, hidden_size),
                v.reshape(kv_hidden_size, hidden_size),
            )

        layer.self_attn.q_proj.load_state_dict({"weight": q}, strict=True)
        layer.self_attn.k_proj.load_state_dict({"weight": k}, strict=True)
        layer.self_attn.v_proj.load_state_dict({"weight": v}, strict=True)

        layer.self_attn.o_proj.load_state_dict(
            {
                "weight": merge_row(
                    check_get(
                        transformers, prefix, transformer_layer_name_list["o_proj"]
                    )
                )
            },
            strict=True,
        )

        gate, up = (
            merge_col(
                check_get(
                    transformers, prefix, transformer_layer_name_list["mlp_gate_up"]
                )
            )
            .view(len(state_dicts), 2, -1, hidden_size)
            .chunk(2, dim=1)
        )
        gate, up = gate.reshape(inter_size, hidden_size), up.reshape(
            inter_size, hidden_size
        )
        layer.mlp.gate_proj.load_state_dict({"weight": gate}, strict=True)
        layer.mlp.up_proj.load_state_dict({"weight": up}, strict=True)
        layer.mlp.down_proj.load_state_dict(
            {
                "weight": merge_row(
                    check_get(
                        transformers, prefix, transformer_layer_name_list["mlp_down"]
                    )
                )
            },
            strict=True,
        )

        layer.post_attention_layernorm.load_state_dict(
            {
                "weight": check_get(
                    transformers,
                    prefix,
                    transformer_layer_name_list["post_attention_layernorm"],
                )[0]
            },
            strict=True,
        )

    # The final layernorm.
    hf_model.model.norm.load_state_dict(
        {"weight": transformers[0]["decoder.final_layernorm.weight"]}, strict=True
    )

    # For LM head, transformers' wants the matrix to weight embeddings.
    output_layers = get(models, "output_layer.weight")

    merged_padded_output_layers = merge_col(output_layers)
    merged_output_layers = merged_padded_output_layers[: model_config.vocab_size, :]
    hf_model.lm_head.load_state_dict({"weight": merged_output_layers}, strict=True)

def get_train_args(state_dict):
    args = state_dict.get("args", None)
    assert args is not None
    return args


def check_padded_vocab_size(train_args, orig_vocab_size):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = (
        train_args.make_vocab_size_divisible_by * train_args.tensor_model_parallel_size
    )
    while (after % multiple) != 0:
        after += 1
    assert (
        train_args.padded_vocab_size == after
    ), "Mismatched vocab size and padded vocab size."


def get_model_config(train_args, vocab_size):
    config = LlamaConfig()
    check_padded_vocab_size(train_args, vocab_size)
    config.vocab_size = vocab_size
    config.max_position_embeddings = train_args.max_position_embeddings
    config.hidden_size = train_args.hidden_size
    config.num_hidden_layers = train_args.num_layers
    config.num_attention_heads = train_args.num_attention_heads
    config.num_key_value_heads = train_args.num_key_value_heads if 'num_key_value_heads' in train_args else train_args.num_query_groups
    config.intermediate_size = train_args.ffn_hidden_size
    config.rope_theta = train_args.rope_base if 'rotary_base' in train_args else 5000000
    config.rms_norm_eps = train_args.layernorm_epsilon if 'layernorm_epsilon' in train_args else train_args.norm_epsilon
    return config


def get_lm_from_model(state_dict, virtual_pp_index, virtual_pp_enabled):
    if virtual_pp_enabled:
        return state_dict[f"model{virtual_pp_index}"]

    return state_dict["model"]


def load_state_dicts(input_dir):
    dirs = [f for f in os.scandir(input_dir) if f.is_dir()]
    args = get_train_args(
        torch.load(os.path.join(dirs[0].path, "model_optim_rng.pt"), map_location="cpu")
    )
    if args.transformer_pipeline_model_parallel_size == 1:
        state_dicts = [
            torch.load(os.path.join(dir.path, "model_optim_rng.pt"), map_location="cpu")
            for dir in dirs
        ]
        return state_dicts, args

    state_dicts = []
    tp_size = args.tensor_model_parallel_size
    pp_size = args.transformer_pipeline_model_parallel_size
    virtual_pp_size = args.virtual_pipeline_model_parallel_size
    virtual_pp_enabled = args.num_layers_per_virtual_pipeline_stage is not None
    if not virtual_pp_enabled:
        virtual_pp_size = 1

    # pipeline parallel split network into multiple stages
    assert (
        args.num_layers % pp_size == 0
    ), "num_layers must be divisible by transformer_pipeline_model_parallel_size"
    num_layers_per_stage = args.num_layers // pp_size

    # virtual pipeline parallel split stage into multiple chunks
    assert (
        num_layers_per_stage % virtual_pp_size == 0
    ), "num_layers_per_stage must be divisible by virtual_pipeline_model_parallel_size"
    num_layers_per_chunk = num_layers_per_stage // virtual_pp_size
    num_layers_per_virtual = args.num_layers // virtual_pp_size

    for tp_index in range(tp_size):
        state_dict = torch.load(
            f"{input_dir}/mp_rank_{tp_index:02d}_000/model_optim_rng.pt",
            map_location="cpu",
        )
        model_weight = get_lm_from_model(state_dict, 0, virtual_pp_enabled)
        #encoder = lm["encoder"]

        print(f">begin load tp-{tp_index}...", flush=True)
        for virtual_pp_index in range(virtual_pp_size):
            print(f"\t>begin load virtual-pp-{virtual_pp_index}...", flush=True)
            for pp_index in range(pp_size):
                print(f"\t\t>begin load pp-{pp_index}...", flush=True)
                if pp_index == 0:
                    if virtual_pp_index == 0:
                        continue
                    this_state_dict = state_dict
                else:
                    this_state_dict = torch.load(
                        f"{input_dir}/mp_rank_{tp_index:02d}_{pp_index:03d}/model_optim_rng.pt",
                        map_location="cpu",
                    )
                this_model_weight = get_lm_from_model(
                    this_state_dict, virtual_pp_index, virtual_pp_enabled
                )
                #this_encoder = this_lm["encoder"]

                if pp_index == pp_size - 1 and virtual_pp_index == virtual_pp_size - 1:
                    model_weight["output_layer.weight"] = this_model_weight["output_layer.weight"]
                    model_weight["decoder.final_layernorm.weight"] = this_model_weight[
                        "decoder.final_layernorm.weight"
                    ]

                for layer_index in range(num_layers_per_chunk):
                    this_layer_index = (
                        layer_index
                        + virtual_pp_index * num_layers_per_virtual
                        + pp_index * num_layers_per_chunk
                    )
                    print(
                        f"\t\t\ttp_index={tp_index}, virtual_pp_index = {virtual_pp_index}, pp_index={pp_index}, ",
                        f"layer_index->global_layer_index={layer_index}->{this_layer_index}",
                        flush=True,
                    )
                    # if (
                    #     args.num_attention_heads == args.num_key_value_heads
                    #     or args.use_mcore_model
                    # ):
                    if (
                           args.use_mcore_models or args.num_attention_heads == args.num_key_value_heads
                    ):
                        model_weight = check_assign(
                            model_weight,
                            this_layer_index,
                            this_model_weight,
                            layer_index,
                            key_list=transformer_layer_name_list["query_key_value"],
                        )
                    else:
                        for key in ["query", "key_value"]:
                            model_weight = check_assign(
                                model_weight,
                                this_layer_index,
                                this_model_weight,
                                layer_index,
                                key_list=transformer_layer_name_list[key],
                            )
                    for key in transformer_layer_name_list.keys():
                        if key not in ("query_key_value", "query", "key_value"):
                            model_weight = check_assign(
                                model_weight,
                                this_layer_index,
                                this_model_weight,
                                layer_index,
                                key_list=transformer_layer_name_list[key],
                            )
        if virtual_pp_enabled:
            state_dict["model"] = state_dict["model0"]
            for virtual_pp_index in range(virtual_pp_size):
                del state_dict[f"model{virtual_pp_index}"]

        state_dicts.append(state_dict)

    return state_dicts, args


def main():
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the megatron checkpoint dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to the huggingface checkpoint dir",
    )
    parser.add_argument(
        "--vit-raw-path",
        type=str,
        help="Path to the original vit",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=64000,
        help="unpadded tokenizer vocab size",
    )
    args = parser.parse_args()

    print("Load megatron checkpoint", flush=True)
    state_dicts, train_args = load_state_dicts(args.input_dir)

    if hasattr(train_args, "vocab_size") and args.vocab_size != train_args.vocab_size:
        print(
            f"Warning: override args.vocab_size({args.vocab_size}) ",
            f"with vocab_size({train_args.vocab_size}) from model state_dicts",
            flush=True,
        )
        args.vocab_size = train_args.vocab_size
    print(f"vocab_size: {args.vocab_size}", flush=True)

    model_config = get_model_config(train_args, args.vocab_size)

    print(f"Model config: {model_config}", flush=True)

    print("Create hf model", flush=True)
    # with accelerate.init_empty_weights():
    #     hf_model = LlamaForCausalLM(model_config)

    hf_model = LlamaForCausalLM(model_config)
    hf_model = hf_model.to(torch.bfloat16)

    Module.load_state_dict = load_state_dict_meta
    # print("convert megatron to hf", flush=True)
    # vit_save_path = os.path.join(args.output_dir, 'saved_vit')
    # mlp_save_path = os.path.join(args.output_dir, 'saved_mlp')
    # convert_megatron_checkpoint(hf_model, state_dicts, model_config, train_args, args.vit_raw_path, vit_save_path, mlp_save_path)
    convert_megatron_checkpoint(hf_model, state_dicts, model_config, train_args )

    print("save hf model", flush=True)
    hf_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

