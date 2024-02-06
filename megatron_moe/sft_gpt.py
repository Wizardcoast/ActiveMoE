# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import os
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.core.enums import ModelType
from megatron.data.gpt_sft_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from pretrain_gpt import model_provider, loss_func


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['input_tokens', 'targets', 'attn_mask', 'loss_mask']
    datatype = torch.int64

    # Broadcast data.
    if not args.tp_shared_dataloader:
        raise ValueError(
            'disable dp shared dataloader is not supported . There is not much gain for sft task.')

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens = data_b['input_tokens'].long()
    attn_mask = data_b['attn_mask'].long()
    labels = data_b['targets'].contiguous()  # next tokens
    data_loss_mask = data_b['loss_mask'].contiguous()

    """
    get_ltor_masks_and_position_ids reposition attention and position_id, loss_mask is only affected
    when using prefix_lm.
    """
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    # print(tokens.size(),data_loss_mask.size())
    data_loss_mask = data_loss_mask.float()
    # if loss_weight=0.01 , loss_mask=100 , then the final loss weight is 1.0
    data_loss_mask = data_loss_mask * args.sft_loss_weight_factor

    # sft datasets has already computed loss mask considering pad & Answer-only. Merge these two.
    loss_mask = data_loss_mask * loss_mask

    """
    attn_mask contains padding mask(same as huggingface). use it to mask pad tokens in the generated casual mask 
    Note: this atention_mask is not used when flash-attention is on. 
    Fortunately, we only has one sft sample per micro batch and loss_mask is 0 for pad token,
    the results are still correct despite 'wrong' attn mask.
    If in the futher we want to pack different sft example and reset attn mask,
    it requires modification of attention module. See InternLM sft-code as an example.
    Currently comment out. will revisit when supporting sft with packing multiple samples.

    expanded_mask = ~(attn_mask[:, None, None, :].to(torch.bool))
    expanded_mask = expanded_mask.expand(attention_mask.size(0), 1, attention_mask.size(2), attention_mask.size(3))
    attention_mask = attention_mask | expanded_mask
    """

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')

    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(data_prefix=args.data_path,
                                                                  seed=args.seed,
                                                                  seq_length=args.seq_length,
                                                                  dataloader_type=args.dataloader_type,
                                                                  train_val_test_num_samples=train_val_test_num_samples,
                                                                  splits=args.split,
                                                                  train_data_prefix=args.train_data_path,
                                                                  valid_data_prefix=args.valid_data_path,
                                                                  test_data_prefix=args.test_data_path,
                                                                  )
    print_rank_0("> finished creating GPT SFT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":
    # 所有的训练都要统一参数这个设计实在有蛋疼
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

