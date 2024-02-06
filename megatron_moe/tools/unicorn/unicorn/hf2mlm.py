import glob
import os
import argparse
from collections import OrderedDict
import re
import json
import types
import random
import numpy as np
import torch

from .utils import recursive_print, get_element_from_dict_by_path
# TODO: understanding fix_query_key_value_ordering
from .utils import megatron_to_transformers_fix_query_key_value_ordering, \
                  transformers_to_megatron_fix_query_key_value_ordering

from .prep_args import add_checkpointing_args, \
                      add_megatron_checkpoint_args, \
                      add_transformers_checkpoint_args

from .cfg_parser import get_megatron_args_from_config

from transformers import GPT2Config

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


tensor_parallel_params = [
    # megatron-lm layers to merge across tp ranks
    # NOTE: RP's bias not need sharding
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight"
]


def convert_layernorm_to_norm_legacy():
    pass


def merge_transformers_sharded_states(path, num_checkpoints):
    # TODO: support different case
    state_dict = OrderedDict()
    for i in range(1, num_checkpoints + 1):
        checkpoint_path = os.path.join(path, f"pytorch_model-{i:05d}-of-{num_checkpoints:05d}.bin")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(path, f"pytorch_model_{i:05d}-of-{num_checkpoints:05d}.bin")
        print(f"=> Loading {checkpoint_path} ...")
        current_chunk = torch.load(checkpoint_path, map_location="cpu")
        state_dict.update(current_chunk)
    return state_dict


def merge_safetensors_sharded_states(path):
    from safetensors import safe_open
    state_dict = OrderedDict()
    pt_files = glob.glob(os.path.join(path, "*.safetensors"))
    assert len(pt_files) > 0

    for pt_file in pt_files:
        print(f"=> Loading {pt_file} ...")
        with safe_open(pt_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)

    return state_dict


def load_huggingface_model(args):
    subdirs = [p for p in os.listdir(args.load_path) if p.startswith("pytorch_model")]
    if len(subdirs) == 0:
        hf_state_dict = merge_safetensors_sharded_states(args.load_path)
    elif len(subdirs) == 1:
        checkpoint_name = "pytorch_model.bin"
        hf_state_dict = torch.load(os.path.join(args.load_path, checkpoint_name), map_location="cpu")
    else:
        num_checkpoints = len(subdirs) - 1
        hf_state_dict = merge_transformers_sharded_states(args.load_path, num_checkpoints)

    return hf_state_dict


def make_megatron_state_dict(args, config, meta_json, hf_state_dict):
    megatron_state_dict = OrderedDict()
    key_map = meta_json['key_map']

    # processing encoder
    encoder_map = key_map['encoder']
    megatron_prefix = "transformer.layers"  # TODO: use config file?
    hf_prefix = key_map['hf_prefix']
    for layer_id in range(config.num_hidden_layers):
        for megatron_key, hf_item in encoder_map.items():
            if megatron_key.startswith("final_norm"):
                continue
            if isinstance(hf_item, str):
                src_key = '.'.join([megatron_prefix, str(layer_id), megatron_key])
                dst_key = '.'.join([hf_prefix, str(layer_id), hf_item])
                megatron_state_dict[src_key] = hf_state_dict[dst_key]
            elif isinstance(hf_item, list):
                if megatron_key.startswith("self_attention.query_key_value"):
                    assert len(hf_item) == 3
                    _hf_chunks = []
                    for hf_sub_key in hf_item:
                        _dst_keys = '.'.join([hf_prefix, str(layer_id), hf_sub_key])
                        _dst_tensor = hf_state_dict[_dst_keys]
                        _hf_chunks.append(_dst_tensor)
                    _hf_chunks = torch.cat(_hf_chunks, dim=0)
                    # TODO: check need fix ordering
                    # hf_new_chunks = transformers_to_megatron_fix_query_key_value_ordering(
                    #     _hf_chunks, )
                    src_key = '.'.join([megatron_prefix, str(layer_id), megatron_key])
                    megatron_state_dict[src_key] = _hf_chunks
                elif megatron_key.startswith("mlp.dense_h_to_4h"):
                    assert len(hf_item) == 2
                    _hf_chunks = []
                    for hf_sub_key in hf_item:
                        _dst_key = '.'.join([hf_prefix, str(layer_id), hf_sub_key])
                        _dst_tensor = hf_state_dict[_dst_key]
                        _hf_chunks.append(torch.chunk(_dst_tensor,
                                                      args.target_tensor_model_parallel_size,
                                                      dim=0))
                    hf_new_chunks = []
                    for i in range(args.target_tensor_model_parallel_size):
                        hf_new_chunks.append(_hf_chunks[0][i])  # w2
                        hf_new_chunks.append(_hf_chunks[1][i])  # w1
                    src_key = '.'.join([megatron_prefix, str(layer_id), megatron_key])
                    megatron_state_dict[src_key] = torch.cat(hf_new_chunks, dim=0)
                else:
                    raise NotImplementedError(f"=> {megatron_key} is not supported yet!")
            else:
                raise NotImplementedError(f"=> {megatron_key} is not implemented yet!")

    # processing final_norm
    for megatron_key, hf_item in encoder_map.items():
        if megatron_key.startswith("final_norm"):
            if len(hf_prefix.split('.')) == 2:
                hf_fst_prefix, hf_sec_prefix = hf_prefix.split('.')
                final_norm_prefix = hf_fst_prefix
            elif len(hf_prefix.split('.')) == 1:
                final_norm_prefix = ""
            else:
                raise NotImplementedError(f"=> hf_prefix: {hf_prefix} is not supported yet!")
            src_key = f"transformer.{megatron_key}"
            dst_key = hf_item if final_norm_prefix == "" \
                              else '.'.join([final_norm_prefix, hf_item])

            megatron_state_dict[src_key] = hf_state_dict[dst_key]

    # processing word embedding
    wte_map = key_map['embedding']
    hf_item = wte_map["word_embeddings.weight"]
    if len(hf_prefix.split(".")) == 2:
        hf_fst_prefix, hf_sec_prefix = hf_prefix.split('.')
        wte_prefix = hf_fst_prefix
    elif len(hf_prefix.split('.')) == 1:
        wte_prefix = ""
    else:
        raise NotImplementedError(f"=> hf_prefix: {hf_prefix} is not supported yet!")
    dst_key = hf_item if wte_prefix == "" \
        else '.'.join([wte_prefix, hf_item])
    megatron_state_dict["transformer.word_embeddings.weight"] = \
        hf_state_dict[dst_key]

    # processing output layer
    if "output_layer" in key_map:
        output_map = key_map['output_layer']
        for megatron_key, hf_item in output_map.items():
            megatron_state_dict[f"transformer.{megatron_key}"] = \
                hf_state_dict[hf_item]

    return megatron_state_dict


def make_margs_and_configs(args):
    os.makedirs(args.save_path, exist_ok=True)

    # Saving config and tokenizer files
    os.system("cp -rf " + args.load_path + "/*.json " + args.save_path)
    os.system("cp -rf " + args.load_path + "/*.tiktoken " + args.save_path)

    # Saving the tracker file
    tracker_filepath = os.path.join(args.save_path, "latest_checkpointed_iteration.txt")
    with open(tracker_filepath, "w") as fd:
        fd.write("release")

    # Create `release` directory in args.load_path
    release_dir = os.path.join(args.save_path, "release")
    os.makedirs(release_dir, exist_ok=True)

    # Create megatron args
    megatron_args, config = get_megatron_args_from_config(
        args.model_name,
        args,
        os.path.join(args.load_path, "config.json")
    )

    margs = types.SimpleNamespace()
    for k, v in megatron_args.items():
        setattr(margs, k, v)

    # Get params dtype
    if args.target_params_dtype in ("fp16", "float16", "half"):
        dtype = torch.float16
    elif args.target_params_dtype in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    setattr(margs, "params_dtype", dtype)

    return config, margs


def _init_embedding_weights(module, std=0.02):
    module.weight.data.normal_(mean=0.0, std=std)


def shard_state_dict(args, config, meta_json, margs, state_dict):
    print("=> Converting ...")
    dtype = margs.params_dtype
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append(OrderedDict())

    # Embedding layer
    print("=> converting embedding layer ...")
    word_embedding = state_dict["transformer.word_embeddings.weight"].to(dtype)
    orig_vocab_size = config.vocab_size
    # padding for orig_vocab_size
    padded_vocab_size = orig_vocab_size
    setattr(margs, "padded_vocab_size", padded_vocab_size)

    if args.extra_num_vocabs > 0:
        setattr(margs, "padded_vocab_size", margs.padded_vocab_size + args.extra_num_vocabs)

    # Cut out extra padding we don't need
    if args.extra_num_vocabs == 0:
        full_word_embed = word_embedding
    else:
        new_embeddings = torch.nn.Embedding(args.extra_num_vocabs, word_embedding.shape[1])
        # initialize all new embeddings (in particular added tokens)
        _init_embedding_weights(new_embeddings)
        full_word_embed = torch.cat([word_embedding, new_embeddings.weight])

    # Split into new tensor model parallel sizes
    out_word_embed = torch.chunk(full_word_embed, args.target_tensor_model_parallel_size, dim=0)
    for i in range(args.target_tensor_model_parallel_size):
        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "model.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embed[i]

    # if untied word embedding
    if state_dict.get("transformer.lm_head.weight", None) is not None:
        lm_head = state_dict["transformer.lm_head.weight"].to(dtype)
        if args.extra_num_vocabs == 0:
            full_lm_head = lm_head
        else:
            full_lm_head = torch.cat([lm_head, new_embeddings.weight])

        out_lm_head = torch.chunk(full_lm_head, args.target_tensor_model_parallel_size, dim=0)
        for i in range(args.target_tensor_model_parallel_size):
            lm_head_dict = get_element_from_dict_by_path(
                output_state_dict[i], "model.language_model.output_layer"
            )
            lm_head_dict['weight'] = out_lm_head[i]

    # Let's play with transformer blocks
    print("=> Converting transformer blocks ...")
    if config.num_hidden_layers % args.target_tensor_model_parallel_size != 0:
        raise ValueError(
            f"Number of layers ({config.num_hidden_layers}) must be divisible by number of "
            f"tensor parallelism ({args.target_tensor_model_parallel_size})"
        )
    num_layers = config.num_hidden_layers // args.target_pipeline_model_parallel_size

    layer_re = re.compile("transformer.layers\.(\d+)\.([a-z0-9_.]+)\.([a-z]+)")
    # The number of heads.
    heads = config.num_attention_heads
    # The hidden_size per head.
    hidden_size_per_head = config.hidden_size // config.num_attention_heads
    for pp_rank in range(args.target_pipeline_model_parallel_size):
        layer_offset = pp_rank * num_layers
        if pp_rank > 0:
            # NOTE: Because the output_state_dict is built already on making
            #       word_embedding and lm_head
            output_state_dict = []
            for i in range(args.target_tensor_model_parallel_size):
                output_state_dict.append(OrderedDict())

        for layer in range(num_layers):
            pp_layer_id = layer + layer_offset
            layers_to_copy = [
                layer_name
                for layer_name in state_dict.keys()
                if layer_name.startswith(f"transformer.layers.{pp_layer_id}.")
            ]

            for layer_name in layers_to_copy:
                m = layer_re.match(layer_name)
                # Stop if that's not a layer
                if m is None:
                    print(f"=> none detected in {layer_name} ...")
                    break

                # The index of the layer.
                _ = int(m.group(1))
                # The name of the operation.
                op_name = m.group(2)
                # Is it a weight or a bias?
                weight_or_bias = m.group(3)

                params = state_dict[layer_name].to(dtype)
                # handle layernorm
                if op_name.startswith("input_norm") or op_name.startswith("post_attention_norm"):
                    out_name = "input_norm" if op_name.endswith("input_norm") else "post_attention_norm"
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                elif op_name.startswith("self_attention.query_key_value"):
                    if args.template_name.lower().startswith("gptx"):
                        params = params
                    else:
                        # transformers stores D X (3*D) but Megatron-LM expects (3*D) X D.
                        params = transformers_to_megatron_fix_query_key_value_ordering(
                            params,
                            3.0,
                            3,
                            heads,
                            hidden_size_per_head,
                        )
                    layer_name = f"layers.{layer}.self_attention.query_key_value.{weight_or_bias}"

                # handle attention and mlp weights
                elif weight_or_bias == "weight":
                    out_name = op_name # transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        print(f"== skip {layer_name} ...")
                        continue
                    # params = params.transpose(0, 1)
                    layer_name = f"layers.{layer}.{out_name}.{weight_or_bias}"

                # handle attention and mlp bias
                elif weight_or_bias == 'bias':
                    out_name = op_name # transformers_to_megatron.get(op_name, None)
                    if out_name is None:
                        continue
                    layer_name = f'layers.{layer}.{out_name}.{weight_or_bias}'

                # skip
                else:
                    print(f"=> skipping {layer_name} ...")
                    continue

                if op_name + "." + weight_or_bias in tensor_parallel_params:
                    dim = 1 if op_name in ["self_attention.dense", "mlp.dense_4h_to_h"] else 0
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=dim)

                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = (
                        params[i].clone() if (
                            op_name + "." + weight_or_bias in tensor_parallel_params) else params.clone()
                    )

        if pp_rank == args.target_pipeline_model_parallel_size - 1:
            # handle final_norm
            final_norm_keys = [k for k in meta_json['key_map']['encoder'] if k.startswith('final_norm')]
            for final_norm_key in final_norm_keys:
                params = state_dict[f"transformer.{final_norm_key}"].to(dtype)
                layer_name = f"{final_norm_key}"
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.encoder")
                    params_dict[layer_name] = params.clone()

            # Add the LM head: compatible with pipeline parallelism.
            for i in range(args.target_tensor_model_parallel_size):
                params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.word_embeddings_for_head")
                params_dict["weight"] = out_word_embed[i].clone()

            # add the LM head
            if state_dict.get("transformer.lm_head.weight", None) is not None:
                for i in range(args.target_tensor_model_parallel_size):
                    params_dict = get_element_from_dict_by_path(output_state_dict[i], "model.language_model.output_layer")
                    params_dict["weight"] = out_lm_head[i].clone()

        # saving the state dict as per the tp_rank and pp_rank
        for tp_rank in range(args.target_tensor_model_parallel_size):
            output_state_dict[tp_rank]["checkpoint_version"] = 3.0
            output_state_dict[tp_rank]["args"] = margs
            if args.iteration >= 0:
                output_state_dict[tp_rank]["iteration"] = args.iteration  # Compatible issue with tools/checkpoint
            checkpoint_dir = (
                f"mp_rank_{tp_rank:02d}"
                if args.target_pipeline_model_parallel_size == 1
                else f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )

            checkpoint_name = "model_optim_rng.pt"
            checkpoint_dir = os.path.join(os.path.join(args.save_path, "release"), checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
            if args.print_checkpoint_structure:
                print(
                    f"Checkpoint structure of model state dict shard belonging to TP rank {tp_rank} and PP rank"
                    f" {pp_rank}:"
                )
                recursive_print(None, output_state_dict[tp_rank])
            torch.save(output_state_dict[tp_rank], checkpoint_path)


def convert_checkpoint_from_transformers_to_megatron(args):
    # 1. read huggingface model into megatron_state_dict
    # 2. make configuration
    # 3. Do TP sharding

    hf_state_dict = load_huggingface_model(args)
    meta_file = os.path.join(os.path.dirname(__file__),
                             "templates",
                             f"{args.template_name}.json")
    with open(meta_file, "r") as fd:
        meta_json = json.load(fd)

    config, margs = make_margs_and_configs(args)
    megatron_state_dict = make_megatron_state_dict(args, config, meta_json, hf_state_dict)
    shard_state_dict(args, config, meta_json, margs, megatron_state_dict)


def main():
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_transformers_to_megatron(args)


if __name__ == "__main__":
    main()
