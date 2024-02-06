import argparse
import json
import os
import re
from collections import OrderedDict
import torch

from transformers.modeling_utils import WEIGHTS_INDEX_NAME, WEIGHTS_NAME, shard_checkpoint

from .utils import get_element_from_dict_by_path, recursive_print
from .utils import megatron_to_transformers_fix_query_key_value_ordering
from .prep_args import add_checkpointing_args, add_transformers_checkpoint_args
from .cfg_parser import get_config_from_megatron_args, _save_config

TP_KEYS = (  # Only need for ColumnParallel
    "self_attention.query_key_value.weight",
    "self_attention.query_key_value.bias",
    "self_attention.dense.weight",
    "mlp.dense_h_to_4h.weight",
    "mlp.dense_h_to_4h.bias",
    "mlp.dense_4h_to_h.weight"
)

RP_KEYS = (
    "self_attention.dense.weight",
    "mlp.dense_4h_to_h.weight",
)

def get_megatron_sharded_states(args, tp_size, pp_size, pp_rank):
    """
        Get sharded checkpoints from NVIDIA Megatron-LM checkpoint based on the provided tensor
        parallel size, pipeline parallel size and pipeline parallel rank.
        Args:
            args (argparse.Namespace): the arguments to the script
            tp_size (int): the tensor parallel size
            pp_size (int): the pipeline parallel size
            pp_rank (int): the pipeline parallel rank
        """
    tp_state_dict = list()
    for i in range(tp_size):
        sub_dir_name = f'mp_rank_{i:02d}' if pp_size == 1 else f"mp_rank_{i:02d}_{pp_rank:03d}"
        checkpoint_name = "model_optim_rng.pt"  # os.listdir(os.path.join(args.load_path, sub_dir_name))[0]
        checkpoint_path = os.path.join(args.load_path, sub_dir_name, checkpoint_name)
        print(f"=> loading {checkpoint_path} to get sharded states ...")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        tp_state_dict.append(state_dict)
    return tp_state_dict


def convert_checkpoint_from_megatron_to_transformers(args):

    # Prepare directory
    os.makedirs(args.save_path, exist_ok=True)

    # Loading meta_json
    meta_file = os.path.join(os.path.dirname(__file__),
                             "templates",
                             f"{args.template_name}.json")
    with open(meta_file, "r") as fd:
        meta_json = json.load(fd)

    encoder_map = meta_json['key_map']['encoder']
    key_remap = meta_json['key_remap'] if 'key_remap' in meta_json else None
    ignore_keys = meta_json['ignore_keys'] if 'ignore_keys' in meta_json else None

    # Load Megatron-LM checkpoint arguments from the state_dict
    subdirs = os.listdir(args.load_path)
    possible_sub_dirs = ["mp_rank_00", "mp_rank_00_000"]
    rank0_checkpoint_path = None
    for subdir in possible_sub_dirs:
        if subdir in subdirs:
            rank0_checkpoint_name = "model_optim_rng.pt"
            rank0_checkpoint_path = os.path.join(args.load_path, subdir, rank0_checkpoint_name)
            break
    assert rank0_checkpoint_path is not None
    print(f"=> Loading Megatron-LM checkpoint from: {rank0_checkpoint_path}")

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

    # Saving config and tokenizer files
    # TODO
    # config = "/".join(args.load_path.split("/")[:-1])
    # os.system("cp -rf " + config_path + "/*.json " + args.save_path)
    # os.system("cp -rf " + config_path + "/tokenizer.model " + args.save_path

    vocab_size = (
        megatron_args.padded_vocab_size
        if getattr(megatron_args, "orig_vocab_size", None) is None
        else megatron_args.orig_vocab_size
    )
    print(f"=> vocab_size: {vocab_size}")

    # TODO: Add Template
    config = get_config_from_megatron_args(args.template_name, megatron_args)
    _save_config(config, args.save_path)

    output_state_dict = OrderedDict()

    checkpoint_version = state_dict.get("checkpoint_version", 0.0)
    heads = megatron_args.num_attention_heads
    hidden_size_per_head = megatron_args.hidden_size // heads
    tp_size = megatron_args.tensor_model_parallel_size
    pp_size = megatron_args.pipeline_model_parallel_size

    # params dtype
    if args.target_params_dtype in ("fp16", "float16", "half"):
        dtype = torch.float16
    elif args.target_params_dtype in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Converting
    print("=> converting ...")

    # Embeddings
    tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, 0)  # Stage-0 has embeddings

    # Convert and store the position embeddings
    # skip

    # Convert and store the word embeddings
    print("=> converting word embeddings ...")
    word_embeddings = []
    extra_pos_embs = []
    for tp_rank in range(tp_size):
        embeddings = get_element_from_dict_by_path(
            tp_state_dicts[tp_rank], "model.language_model.embedding"
        )
        if "word_embeddings.weight" in embeddings:
            word_embeddings.append(embeddings['word_embeddings.weight'])
        else:
            word_embeddings.append(embeddings['word_embeddings']['weight'])

        # if args.has_position_embeddings:
        #     pass

    word_embeddings = torch.cat(word_embeddings, dim=0)
    word_embeddings = word_embeddings.to(dtype)
    #
    layer_prefix = meta_json["key_map"]["hf_prefix"]
    layer_splits = layer_prefix.split(".")
    if len(layer_splits) == 2:
        layer_fst_prefix, layer_sec_prefix = layer_splits
    elif len(layer_splits) == 1:
        layer_fst_prefix = None
        layer_sec_prefix = layer_prefix
    else:
        raise NotImplementedError(f"=> layer_prefix: {layer_prefix} is not supported!")
    dst_prefix = "" if layer_fst_prefix is None else layer_fst_prefix + "."
    dst_suffix = meta_json["key_map"]["embedding"]["word_embeddings.weight"]
    dst_key = dst_prefix + dst_suffix
    output_state_dict[dst_key] = word_embeddings

    # Extract output_layer
    if 'output_layer' in meta_json['key_map']:
        tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_size - 1)  # Last PP stage
        output_layer = []
        for tp_rank in range(tp_size):
            _output_layer = get_element_from_dict_by_path(
                tp_state_dicts[tp_rank], "model.language_model.output_layer"
            )
            output_layer.append(_output_layer['weight'])
        output_layer = torch.cat(output_layer, dim=0).to(dtype)

    # Transformer layers
    print("=> Converting transformer layers ...")
    # The number of heads.
    # head = config.n_head
    # The hidden_size per head.
    # hidden_size_per_head = config.hidden_size // config.n_head
    num_layers = megatron_args.num_layers // pp_size
    new_layer_idx = -1
    new_layer_type = path = ""
    # TODO: use template

    layer_idx_set = set()
    for pp_rank in range(pp_size):
        print(f"=> converting pipeline parallel rank {pp_rank} ...")
        tp_state_dicts = get_megatron_sharded_states(args, tp_size, pp_size, pp_rank)

        # The transformer block.
        path = (
            "model.language_model.transformer"
            if "transformer" in get_element_from_dict_by_path(tp_state_dicts[0], "model.language_model").keys()
            else "model.language_model.encoder"
        )

        # Extract the layers.
        # The regex to extract layer names.
        layer_re = re.compile("layers\.(\d+)\.([a-z0-9_.]+)\.([a-z0-9_.]+)")
        for key, val in get_element_from_dict_by_path(tp_state_dicts[0], path).items():
            _skip = False
            if ignore_keys:
                _skip = False
                for _ignore_k in ignore_keys:
                    if key.endswith(_ignore_k):
                        _skip = True
                        break
            if _skip:
                print(f"\t=> skipping processing {key} ...")
                continue

            print(f"\t=> processing {key} ...")

            # remapping
            if key_remap:
                for _remap_k, _remap_v in key_remap.items():
                    if key.endswith(_remap_k):
                        o_key = key
                        key = o_key.replace(_remap_k, _remap_v)
                        print(f"\t=> remap {o_key} to {key}")
                        break

            m = layer_re.match(key)
            if m is None:
                # it will skip `final_norm.weight` ...
                print(f"=> layer regex match failed [{key}] , skip ...")
                continue

            # The index of the layer
            layer_idx = int(m.group(1))
            # The name of the operation
            op_name = m.group(2)
            # Is it a weight or a bias?
            weight_or_bias = m.group(3)
            # Update to new layer_idx
            new_layer_idx = layer_idx + pp_rank * num_layers
            layer_idx_set.add(new_layer_idx)

            # get merged weights on tps
            _has_merge_shard = False
            if op_name + "." + weight_or_bias not in TP_KEYS:
                params = val.to(dtype)
            else:
                dim = 1 if op_name + "." + weight_or_bias in RP_KEYS else 0
                _has_merge_shard = True
                # print(f"\t== DIM: {dim}")
                params = torch.cat(
                    [val] + [
                        get_element_from_dict_by_path(tp_state_dicts[tp_rank], f'{path}')[key]
                        for tp_rank in range(1, tp_size)
                    ], dim=dim
                ).to(dtype)

            layer_name = op_name + "." + weight_or_bias
            if layer_name not in encoder_map:
                print(f"=> skipping processing {layer_name} ...")
                continue
            else:
                mapping_vals = encoder_map[layer_name]
                if isinstance(mapping_vals, list):
                    if op_name == "mlp.dense_h_to_4h":
                        assert len(mapping_vals) == 2
                        _params_chunks = torch.chunk(params, tp_size, dim=0)
                        _sharded_chunks = [[] for _ in range(len(mapping_vals))]
                        for i in range(tp_size):
                            _params = _params_chunks[i]
                            _chunks = torch.chunk(_params, len(mapping_vals), dim=0)
                            for j, _chunk in enumerate(_chunks):
                                _sharded_chunks[j].append(_chunk)

                        for i, _sharded_chunk in enumerate(_sharded_chunks):
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals[i]])
                            output_state_dict[_new_hf_key] = torch.cat(_sharded_chunk, dim=0)

                    elif op_name == "mlp_share.dense_h_to_4h":
                        assert len(mapping_vals) == 2
                        _params_chunks = torch.chunk(params, tp_size, dim=0)
                        _sharded_chunks = [[] for _ in range(len(mapping_vals))]
                        for i in range(tp_size):
                            _params = _params_chunks[i]
                            _chunks = torch.chunk(_params, len(mapping_vals), dim=0)
                            for j, _chunk in enumerate(_chunks):
                                _sharded_chunks[j].append(_chunk)

                        for i, _sharded_chunk in enumerate(_sharded_chunks):
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals[i]])
                            output_state_dict[_new_hf_key] = torch.cat(_sharded_chunk, dim=0)

                    elif op_name == "self_attention.query_key_value":
                        assert len(mapping_vals) == 3
                        out_val = megatron_to_transformers_fix_query_key_value_ordering(
                            params,
                            checkpoint_version,
                            3,
                            heads,
                            hidden_size_per_head
                        )
                        for out_key, out_sub_val in zip(mapping_vals,
                                                        torch.split(out_val, out_val.shape[1], dim=0)):
                            # q_proj, k_proj, v_proj
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), out_key])
                            output_state_dict[_new_hf_key] = out_sub_val

                    elif op_name+'.'+weight_or_bias == "mlp.moe.experts.mlp.w1":
                        assert len(mapping_vals) == 1
                        _params_chunks = torch.chunk(params, tp_size, dim=0)
                        _sharded_chunks = [[] for _ in range(megatron_args.moe_num_experts)]
                        for i in range(tp_size):
                            _params = _params_chunks[i]
                            _chunks = torch.chunk(_params, megatron_args.moe_num_experts, dim=0)
                            for j, _chunk in enumerate(_chunks):
                                _sharded_chunks[j].append(_chunk)

                        for i, _sharded_chunk in enumerate(_sharded_chunks):
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals[0].replace('*',str(i))])
                            output_state_dict[_new_hf_key] = torch.cat(_sharded_chunk, dim=0)

                    elif op_name+'.'+weight_or_bias == "mlp.moe.experts.mlp.v1":
                        assert len(mapping_vals) == 1
                        _params_chunks = torch.chunk(params, tp_size, dim=0)
                        _sharded_chunks = [[] for _ in range(megatron_args.moe_num_experts)]
                        for i in range(tp_size):
                            _params = _params_chunks[i]
                            _chunks = torch.chunk(_params, megatron_args.moe_num_experts, dim=0)
                            for j, _chunk in enumerate(_chunks):
                                _sharded_chunks[j].append(_chunk)

                        for i, _sharded_chunk in enumerate(_sharded_chunks):
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals[0].replace('*',str(i))])
                            output_state_dict[_new_hf_key] = torch.cat(_sharded_chunk, dim=0)

                    elif op_name+'.'+weight_or_bias == "mlp.moe.experts.mlp.w2":
                        assert len(mapping_vals) == 1
                        _params_chunks = torch.chunk(params, tp_size, dim=0)
                        _sharded_chunks = [[] for _ in range(megatron_args.moe_num_experts)]
                        for i in range(tp_size):
                            _params = _params_chunks[i]
                            _chunks = torch.chunk(_params, megatron_args.moe_num_experts, dim=0)
                            for j, _chunk in enumerate(_chunks):
                                _sharded_chunks[j].append(_chunk)

                        for i, _sharded_chunk in enumerate(_sharded_chunks):
                            _new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals[0].replace('*',str(i))])
                            output_state_dict[_new_hf_key] = torch.cat(_sharded_chunk, dim=0).t()

                    else:
                        raise NotImplementedError(f"Error in processing {layer_name}:[{op_name}/{weight_or_bias}]!!")
                    pass
                elif _has_merge_shard:
                    if op_name == "self_attention.query_key_value":  # Qwen
                        if args.template_name.lower().startswith("gptx"):
                            out_val = params
                        else:
                            out_val = megatron_to_transformers_fix_query_key_value_ordering(
                                params,
                                checkpoint_version,
                                3,
                                heads,
                                hidden_size_per_head
                            )
                        new_hf_key = ".".join([layer_prefix, str(new_layer_idx), encoder_map[layer_name]])
                        output_state_dict[new_hf_key] = out_val
                    else:
                        new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals])
                        output_state_dict[new_hf_key] = params
                elif not _has_merge_shard:
                    new_hf_key = ".".join([layer_prefix, str(new_layer_idx), mapping_vals])
                    output_state_dict[new_hf_key] = params
                else:
                    raise NotImplementedError(f"=> Error in processing {layer_name}:[{op_name}/{weight_or_bias}]!!")

    if megatron_args.num_layers != len(layer_idx_set):
        raise ValueError(f"Expected {megatron_args.num_layers} layers but found {len(layer_idx_set)}!")

    # making pos_emb for hf model
    if "pos_emb_key" in meta_json['key_map']:
        pos_emb_key = meta_json['key_map']['pos_emb_key']
        pos_emb_dim = megatron_args.hidden_size // megatron_args.num_attention_heads \
                      if megatron_args.kv_channels is None else megatron_args.kv_channels
        if "rotary_base" in megatron_args:
            pos_emb_base = megatron_args.rotary_base
        else:
            pos_emb_base = 10000
        def _make_inv_freq(dim, base):
            # NOTE: LLaMA-2-hf model's rotary embedding `inv_freq` is cast to float16 already,
            # such as (1. / (base ** (torch.arange(0, dim, 2).float() / dim)).half(),
            # so it will lead to different comparison results by tests/unicorn/compare_hf_model.py,
            # but it is okay according to our implementation of RoPE.
            # NOTE: Take care if you want to use meta/LLaMA for finetuning.
            return 1. / (base ** (torch.arange(0, dim, 2).float() / dim))

        print(f"=> making pos_emb (only support RoPE now): "
              f"dim={pos_emb_dim}, base={pos_emb_base}")
        pos_inv_freq = _make_inv_freq(pos_emb_dim, pos_emb_base)
        for _idx in layer_idx_set:
            new_hf_key = f"{layer_prefix}.{_idx}.{pos_emb_key}"
            output_state_dict[new_hf_key] = pos_inv_freq  # on CPU device

    # TODO: Qwen
    # convert final norm
    print("=> converting final norm ...")
    params = get_element_from_dict_by_path(tp_state_dicts[0], str(path))
    final_norm_keys = [k for k in encoder_map.keys() if k.startswith("final_norm") or k.startswith("final_layernorm")]
    layer_splits = layer_prefix.split(".")
    if len(layer_splits) == 2:
        layer_fst_prefix, layer_sec_prefix = layer_splits
    elif len(layer_splits) == 1:
        layer_fst_prefix = None
        layer_sec_prefix = layer_prefix
    else:
        raise NotImplementedError(f"=> layer_prefix: {layer_prefix} is not supported!")
    for final_norm_key in final_norm_keys:
        dst_final_norm_key = encoder_map[final_norm_key]
        dst_key = dst_final_norm_key if layer_fst_prefix is None else f"{layer_fst_prefix}.{dst_final_norm_key}"
        output_state_dict[dst_key] = params[final_norm_key].to(dtype)

    # For LM head, transformers's wants the matrix to weight embeddings.
    if "output_layer" in meta_json['key_map']:
        print("=> converting lm_head ...")
        dst_key = meta_json['key_map']['output_layer']['lm_head.weight']
        output_state_dict[dst_key] = output_layer

    # Converting done.
    print("=> convertion from Megatron-LM to Transformers is done!")

    # Print the structure of converted state_dict.
    if args.print_checkpoint_structure:
        recursive_print(None, output_state_dict)

    # TODO: Store the config file.
    # print("=> saving config ...")
    # config.save_pretrained(args.save_path)

    # Store the state_dict to file.
    max_shard_size = int(args.max_shard_size) if args.max_shard_size.isdigit() else args.max_shard_size
    shards, index = shard_checkpoint(output_state_dict, max_shard_size=max_shard_size)

    # Save the model
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(args.save_path, shard_file))

    if index is None:
        print(f"=> Model weights saved in {os.path.join(args.save_path, WEIGHTS_NAME)}")
    else:
        save_index_file = os.path.join(args.save_path, WEIGHTS_INDEX_NAME)
        # Save the index file
        with open(save_index_file, "w", encoding="utf-8") as fd:
            content = json.dumps(index, indent=4, sort_keys=True)
            fd.write(content + "\n")
        print(f"The model is bigger than the maximum size per checkpoint ({args.max_shard_size}) and is going to be "
              f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
              f"index located at {save_index_file}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-params-dtype",
        type=str,
        default='fp32',
        help='The dtype of the converted checkpoint.'
    )
    parser = add_checkpointing_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    args = parser.parse_args()
    convert_checkpoint_from_megatron_to_transformers(args)


if __name__ == "__main__":
    main()
