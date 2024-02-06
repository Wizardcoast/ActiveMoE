# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT"""

import os
import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import tensor_parallel, parallel_state
from megatron.core.enums import ModelType
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import GPTModel, megablocks_utils
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.arguments import core_transformer_config_from_args
from megatron.model.megablocks_utils import moe

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()

    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(get_args())
    model = GPTModel(
        config,
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    if args.tp_shared_dataloader:
        # Broadcast data.
        data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    else:
        data_b = {}
        assert data is not None
        for key in keys:
            assert data[key].dtype == datatype, (
                '{} has data type {} which '
                'is different than {}'.format(key, data[key].dtype, datatype)
            )
            data_b[key] = data[key].cuda(non_blocking=True)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    """
    目前megatron-lm对flash-attn的整合只支持causal mask，see ParallelAttention
    right pad的模式，我们可以通过改动loss mask去适配pad
    无法支持其他的pad情况,比如
        1)data1,pad1,data2,pad2
        2)reset attn mask

    除此之外,pad会在eod之后隔断下一个doc的first token
    tokens_:   token_1_0,token_1_1,eod,pad,pad,token_2_0
    loss_mask:    1          1        0    0    0
    无论是src还是tgt为pad，loss mask都为0
    所以p(token| eod)无法被训练。 即doc的first token不会被训练到。
    """
    if args.tokenizer_type != "NullTokenizer":
        pad_mask = torch.logical_and(tokens_[:,:-1]!=tokenizer.pad_token_id , tokens_[:,1:]!=tokenizer.pad_token_id)
        loss_mask = loss_mask * pad_mask

    return tokens, labels, loss_mask, attention_mask, position_ids

def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    # for some task loss_mask maybe float. use number of non-zero loss_mask 
    loss = torch.sum(losses.view(-1) * loss_mask) / torch.sum(loss_mask>0)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    args = get_args()
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    return loss, {'lm loss': loss}

#add moe related loss function
def moe_loss_func(loss_mask, output_tensor=None):
    # NOTE: For pipeline parallelism this function will be run on the
    # non-final stages to calculate load balancing loss contribution
    # for the MoE layers within the stage. For these cases, output_tensor
    # will be None.
    loss, loss_dict = (None, {})
    if parallel_state.is_pipeline_last_stage():
        assert output_tensor is not None
        loss, loss_dict = loss_func(loss_mask, output_tensor)
        assert loss.numel() == 1

    # NOTE: If recompute is enabled we will collect duplicate load
    # balancing loss contributions. Prune these before calculating
    # the load balancing loss.
    args = get_args()
    if args.recompute_granularity is not None:
        # Ignore load balancing loss contributions compute during
        # the forward pass if recompute is turned on.
        load_balancing_loss_data = moe.get_load_balancing_loss()
        if args.num_layers * 2 == len(load_balancing_loss_data):
            load_balancing_loss_data = (
                load_balancing_loss_data[args.num_layers:])
            moe.clear_load_balancing_loss()
            moe.save_load_balancing_loss(load_balancing_loss_data)

    # Compute the load balancing loss for all MoE layers.
    megablocks_args = megablocks_utils.arguments.from_megatron(args)
    lbl = moe.batched_load_balancing_loss(megablocks_args)
    moe.clear_load_balancing_loss()

    # Average the load balancing loss across data parallel
    # replicas and save for logging.
    averaged_lbl = average_losses_across_data_parallel_group([lbl])
    loss_dict['load balancing loss'] = averaged_lbl[0]

    # Compute the total loss, if necessary.
    total_loss = loss + lbl if loss is not None else lbl
    return total_loss, loss_dict


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
    # choose to use normal loss func or moe loss func
    loss_fn = (
        moe_loss_func if args.moe_num_experts is not None else loss_func)
    return output_tensor, partial(loss_fn, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path)
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
