"""
* Convert megatron-checkpoint to the special TP=1 & PP=1 megatron-checkpoint (t1p1-mlm-ckpt)
* Compare weights between t1p1-mlm-ckpt and huggingface model
* Compare forward computation results between t1p1-mlm-ckpt and huggingface model
"""
import sys
import os
import glob
import pdb
import argparse
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))

import subprocess
import torch
from transformers import AutoModelForCausalLM
from tests.unit_tests.test_utilities import Utils
from megatron.arguments import parse_args


def transformers_to_megatron_fix_query_key_value_ordering(
    param, checkpoint_version, num_splits, num_heads, hidden_size
):
    """
    Permutes layout of param tensor to the one compatible with respective NVIDIA Megatron-LM chekpoint versions. Input
    is [num_splits * num_heads * hidden_size, :] and output is [num_heads * hidden_size * num_splits, :] for version
    1.0 and [num_heads * num_splits * hidden_size, :] for version 2.0 and later. If param is the weight tensor of the
    self-attention block, the param needs to be already transposed before calling this function.
    Args:
        param (torch.Tensor): the tensor to permute
        checkpoint_version (float): the version of the checkpoint.
        num_splits (int): the number of projections, usually 3 for (Query, Key, Value)
        num_heads (int): the number of attention heads
        hidden_size (int): the hidden size per head
    """

    # Input is [num_splits * num_heads * hidden_size, :]
    input_shape = param.size()
    if checkpoint_version == 1.0:
        # version 1.0 stores [num_heads * hidden_size * num_splits, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 2)
        param = param.transpose(1, 2).contiguous()
    elif checkpoint_version >= 2.0:
        # other versions store [num_heads * num_splits * hidden_size, :]
        current_shape = (num_splits, num_heads, hidden_size) + input_shape[1:]
        param = param.view(*current_shape)
        param = param.transpose(0, 1).contiguous()
    param = param.view(*input_shape)
    return param


def parse_input_args():
    parser = argparse.ArgumentParser(
        description='compare weight and computation results between MLM and HF.')
    parser.add_argument("--model-type", required=True, type=str,
                        choices=('GPT', 'LLaMA'),
                        help="Choose GPT or LLaMA.")
    parser.add_argument("--megatron-checkpoint", required=True, type=str,
                        help="Path to megatron checkpoint.")
    parser.add_argument("--huggingface-model", required=True, type=str,
                        help="Path to huggingface model.")
    parser.add_argument("--pre-convert-megatron", action="store_true",
                        help="Convert megatron checkpoint to TP=1 & PP=1 before compare.")
    parser.add_argument("--pre-megatron-checkpoint", type=str, default=None,
                        help="Path to megatron checkpoint if need pre-converted.")
    parser.add_argument("--compare-weight", action="store_true",
                        help="Compare weight.")
    parser.add_argument("--compare-forward", action="store_true",
                        help="Compare forward.")
    return parser.parse_args()


def pre_convert_func(model_type, load_dir, save_dir):
    megatron_dir = os.path.join(os.path.dirname(__file__),
                                os.path.pardir, os.path.pardir)
    print(f"=> megatron_dir: {megatron_dir}")
    print(f"=> convert model from {load_dir} to {save_dir}.")

    cmds = [
        'python',
        'tools/checkpoint/util.py',
        '--model-type', model_type,
        '--load-dir', load_dir,
        '--save-dir', save_dir,
        '--target-tensor-parallel-size', '1',
        '--target-pipeline-parallel-size', '1',
    ]
    subprocess.run(cmds, cwd=megatron_dir)


class TestConvertCheckpoint(object):
    def __init__(self,
                 hf_model,
                 mlm_ckpt,
                 model_type,
                 param_dtype=torch.bfloat16,
                 pre_convert=False,
                 pre_convert_mlm_ckpt="",
                 ):
        self.mlm_ckpt = mlm_ckpt
        self.model_type = model_type.lower()
        if pre_convert:
            print("=> Do pre-converting before compare ...")
            pre_convert_func(model_type,
                             os.path.abspath(pre_convert_mlm_ckpt),
                             os.path.abspath(self.mlm_ckpt))

        self.hf_model = AutoModelForCausalLM.from_pretrained(hf_model,
                                                             torch_dtype=param_dtype,
                                                             trust_remote_code=True,
                                                             ).to("cuda")

    def _compare_llama_weight(self):
        hf_state_dict = self.hf_model.state_dict()
        # As TP=1 and PP=1
        pt_files = glob.glob(os.path.join(self.mlm_ckpt, '*', 'mp_rank_00', 'model_optim_rng.pt'))
        assert len(pt_files) == 1
        megatron_state_dict = torch.load(pt_files[0], map_location="cpu")

        margs = megatron_state_dict['args']
        _heads = margs.num_attention_heads
        _hidden_size_per_head = margs.hidden_size // _heads

        # Compare transformer-block weights.
        for layer_name, layer_parameters in megatron_state_dict['model']['language_model']['encoder'].items():
            # mapping layer_name to llama model
            hf_prefix = "model."
            if layer_name.endswith("input_norm.weight"):
                hf_layer_name = hf_prefix + layer_name.replace("input_norm.weight", "input_layernorm.weight")
                hf_parameters = hf_state_dict[hf_layer_name]
            elif layer_name.endswith("post_attention_norm.weight"):
                hf_layer_name = hf_prefix + layer_name.replace("post_attention_norm.weight",
                                                               "post_attention_layernorm.weight")
                hf_parameters = hf_state_dict[hf_layer_name]
            elif layer_name.endswith("self_attention.query_key_value.weight"):
                qkv_proj = []
                for proj in ('q_proj', 'k_proj', 'v_proj'):
                    _hf_layer_name = hf_prefix + layer_name.replace("self_attention.query_key_value.weight",
                                                                    f"self_attn.{proj}.weight")
                    _hf_parameters = hf_state_dict[_hf_layer_name]
                    qkv_proj.append(_hf_parameters)
                hf_layer_name = "(q_proj, k_proj, v_proj)"
                hf_parameters = torch.cat(qkv_proj, dim=0)
                hf_parameters = transformers_to_megatron_fix_query_key_value_ordering(hf_parameters,
                                                                                      3.0,
                                                                                      3,
                                                                                      _heads,
                                                                                      _hidden_size_per_head, )
            elif layer_name.endswith("self_attention.dense.weight"):
                hf_layer_name = hf_prefix + layer_name.replace("self_attention.dense.weight", "self_attn.o_proj.weight")
                hf_parameters = hf_state_dict[hf_layer_name]
            elif layer_name.endswith("mlp.dense_h_to_4h.weight"):
                gate_layer_name = hf_prefix + layer_name.replace("mlp.dense_h_to_4h.weight", "mlp.gate_proj.weight")
                gate_proj = hf_state_dict[gate_layer_name]
                up_layer_name = hf_prefix + layer_name.replace("mlp.dense_h_to_4h.weight", "mlp.up_proj.weight")
                up_proj = hf_state_dict[up_layer_name]
                # because TP=1
                hf_layer_name = "(gate_proj, up_proj)"
                hf_parameters = torch.cat([gate_proj, up_proj], dim=0)
            elif layer_name.endswith("mlp.dense_4h_to_h.weight"):
                hf_layer_name = hf_prefix + layer_name.replace("mlp.dense_4h_to_h.weight", "mlp.down_proj.weight")
                hf_parameters = hf_state_dict[hf_layer_name]
            elif layer_name.endswith("final_norm.weight"):
                hf_layer_name = hf_prefix + layer_name.replace("final_norm.weight", "norm.weight")
                hf_parameters = hf_state_dict[hf_layer_name]
            else:
                print(f"Unknown layer_name: {layer_name}")
                continue

            assert hf_parameters.size() == layer_parameters.size()
            diff_cnt = torch.sum(layer_parameters - hf_parameters.cpu())
            print(f"compare {layer_name} vs. {hf_layer_name}: {diff_cnt}")

        # compare wte and head.
        # untie word embeddings
        embedding_diff = torch.sum(
            megatron_state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] -
            hf_state_dict['model.embed_tokens.weight'].cpu())
        print(f"embedding_diff: {embedding_diff}")
        head_diff = torch.sum(megatron_state_dict['model']['language_model']['output_layer']['weight'] -
                              hf_state_dict['lm_head.weight'].cpu())
        print(f"head_diff: {head_diff}")

    def _compare_gpt_weight(self):
        hf_state_dict = self.hf_model.state_dict()
        # As TP=1 and PP=1
        pt_files = glob.glob(os.path.join(self.mlm_ckpt, '*', 'mp_rank_00', 'model_optim_rng.pt'))
        assert len(pt_files) == 1
        megatron_state_dict = torch.load(pt_files[0], map_location="cpu")
        key_mapping = {
            'final_layernorm': 'transformer.ln_f',
            'layers': 'transformer.h'
        }
        for layer_name, layer_parameters in megatron_state_dict['model']['language_model']['encoder'].items():
            for key, value in key_mapping.items():
                if layer_name.startswith(key):
                    hf_layer_name = layer_name.replace(key, value)
                    # TODO: align to new modeling_gptx.py
                    hf_layer_name = hf_layer_name.replace("_norm", "_layernorm")
                    hf_parameters = hf_state_dict[hf_layer_name]
                    assert hf_parameters.size() == layer_parameters.size()
                    diff_cnt = torch.sum(layer_parameters - hf_parameters.cpu())
                    print(f"compare {layer_name} vs. {hf_layer_name}: {diff_cnt}")

        embedding_diff = torch.sum(megatron_state_dict['model']['language_model']['embedding']['word_embeddings']['weight'] -
                                   hf_state_dict['transformer.word_embeddings.weight'].cpu())
        print(f"embedding_diff: {embedding_diff}")

    def compare_weight(self):
        if self.model_type == 'gpt':
            self._compare_gpt_weight()
        elif self.model_type == 'llama':
            self._compare_llama_weight()
        else:
            raise NotImplementedError(f"model_type {self.model_type} is not supported yet!")

    def compare_forward(self):
        sys.argv = ['script.py',
                    '--load', self.mlm_ckpt,
                    '--micro-batch-size', '1',
                    '--no-initialization',
                    '--no-async-tensor-model-parallel-allreduce',
                    '--train-iters', '10',
                    '--no-load-lr-scheduler',
                    '--no-load-rng',
                    '--position-embedding-type', 'rope',
                    # '--rotary-position-embeddings-type', 'normal',
                    '--bf16',
                    '--attention-softmax-in-fp32',
                    # '--no-bias-gelu-fusion',    # Diff in GPTx
                    # '--no-bias-dropout-fusion', # Doesn't matter
                    '--seq-length', '3',
                    '--attention-dropout', '0.0',
                    '--hidden-dropout', '0.0',
                    '--tokenizer-type', 'NullTokenizer',
                    '--vocab-size', '96511'
                    ]
        if self.model_type == 'llama':
            sys.argv += ['--normalization', 'RMSNorm']

        margs = parse_args()
        this_model, _, _ = Utils.gpt_load_checkpoint(
            margs, load_scheduler=False)

        # set boilerplate code for forward. Thank U Megatron!
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
        from megatron.core.parallel_state import _set_global_memory_buffer
        _set_global_memory_buffer()
        model_parallel_cuda_manual_seed(42)

        device = 'cuda'

        input_ids = torch.tensor([[13, 22, 35]]).to(device)
        position_ids = torch.tensor([[0, 1, 2]]).to(device)
        # attention_mask = torch.tensor([[True, True, True]]).to(device)
        seq_length = 3
        att_mask_batch = 1
        attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=device)).view(
            att_mask_batch, 1, seq_length, seq_length)
        attention_mask = (attention_mask < 0.5)
        this_model.to(device)
        this_model.eval()

        megatron_logits = this_model.forward(
            input_ids, position_ids, attention_mask)
        print(megatron_logits[0][0])

        self.hf_model.eval()
        hf_logits = self.hf_model(input_ids=torch.tensor(
            [[13, 22, 35]]).to(device), attention_mask=torch.tensor([[1, 1, 1]]).to(device)).logits

        print(hf_logits[0][0])
        diff_cnt = torch.sum(megatron_logits[0][0] != hf_logits[0][0])
        print(f"DIFF: {diff_cnt}")

        pdb.set_trace()
        # assert equal. u need to put them on the same deivce
        # to test more need to try different sequence length


if __name__ == "__main__":
    args = parse_input_args()
    assert args.compare_weight or args.compare_forward or args.pre_convert_megatron

    x = TestConvertCheckpoint(
        hf_model=args.huggingface_model,
        mlm_ckpt=args.megatron_checkpoint,
        model_type=args.model_type,
        param_dtype=torch.bfloat16,
        pre_convert=args.pre_convert_megatron,
        pre_convert_mlm_ckpt=args.pre_megatron_checkpoint,
    )

    if args.compare_weight:
        print("=> compare weight ...")
        x.compare_weight()
    if args.compare_forward:
        print("=> compare forward ...")
        x.compare_forward()
