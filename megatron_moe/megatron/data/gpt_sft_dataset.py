# coding=utf-8

import os
import sys
import torch
import numpy as np
import collections
import json
import logging
from megatron import print_rank_0, get_args
from megatron.data.dataset_utils import get_train_valid_test_split_
import math


class GPTSFTDataset(torch.utils.data.Dataset):

    def __init__(self, datafile_path):
        # Params to store.
        self.datafile_path = datafile_path
        self.samples = np.load(
            datafile_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        return {
            'input_tokens': sample['input_tokens'].astype(np.int64),
            'targets': sample['targets'].astype(np.int64),
            'loss_mask': sample['loss_mask'].astype(np.int64),
            'truncated': sample['truncated'] if 'truncated' in sample else 0,
            'attn_mask': sample['attn_mask'].astype(np.int64),
        }


def build_train_valid_test_datasets(data_prefix,
                                    seed,
                                    seq_length,
                                    dataloader_type,
                                    train_val_test_num_samples,
                                    splits=None,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    ):
    """Build train, valid, and test datasets."""

    # Single dataset.
    if data_prefix:
        raise NotImplementedError(
            'for SFT, single dataset mode is not supported. Specify train&valid&test yourself. ')

    else:
        print_rank_0(
            "Separate data paths provided for train, valid & test. Split string will be ignored.")

        if ((train_data_prefix is not None and len(train_data_prefix) != 1) or
                (valid_data_prefix is not None and len(valid_data_prefix) != 1) or
                (test_data_prefix is not None and len(test_data_prefix) != 1)):
            raise NotImplementedError(
                'blending dataset for sft is not supported.')

        train_dataset, valid_dataset, test_dataset = None, None, None
        # Single dataset.
        if train_data_prefix is not None:
            train_dataset = _build_dataset(
                train_data_prefix[0], seq_length, train_val_test_num_samples[0], seed, dataloader_type, is_train=True)

        if valid_data_prefix is not None:
            valid_dataset = _build_dataset(
                valid_data_prefix[0], seq_length, train_val_test_num_samples[1], seed, dataloader_type)

        if test_data_prefix is not None:
            test_dataset = _build_dataset(
                test_data_prefix[0], seq_length, train_val_test_num_samples[2], seed, dataloader_type)

    return (train_dataset, valid_dataset, test_dataset)


def _build_dataset(data_prefix, seq_length, num_samples, seed, dataloader_type='single', is_train=False):
    """
    if dataloader_type=='single', expand according to num_samples
    if dataloader_type=='cyclic', shuffle dataset. Note that in cyclic sampler, it only use epoch as random seed.
    so we need shuffle here with respect to args.seed.
    """
    repeat_to_num_samples = dataloader_type == 'single'
    data_path = data_prefix
    dataset = GPTSFTDataset(data_path)

    if is_train and dataloader_type == 'cyclic':
        generator_ = torch.Generator().manual_seed(seed)
        dataset, _, _ = torch.utils.data.random_split(
            dataset, lengths=[1, 0, 0], generator=generator_)

    num_epochs = math.ceil(num_samples*1.0 / seq_length /
                           len(dataset)) if repeat_to_num_samples else 1
    full_datasets = [dataset for e in range(num_epochs)]
    full_dataset = torch.utils.data.ConcatDataset(full_datasets)

    print(f"[{torch.distributed.get_rank() if torch.distributed.is_initialized() else -1 }] \
        dataset: {full_dataset[0]['input_tokens'][:20]}")
    print(
        f" Data Prefix: {data_prefix} Num samples: {num_samples} len(full_dataset):{len(full_dataset)} len(dataset):{len(dataset)}")
    return full_dataset


def _build_train_valid_test_datasets(data_prefix, seed, splits_string=None, shuffle=True):
    """Build train, valid, and test datasets.
    example:
    import sys
    sys.argv = ['script.py',
                '--split',"90,10,0",
                '--micro-batch-size' ,'1',
                '--num-layers', ]
    from megatron.arguments import parse_args,validate_args
    args = parse_args()
    from megatron.global_vars import set_args, set_global_variables
    validate_args(args, {})
    set_global_variables(args)

    a, b, c = _build_train_valid_test_datasets('sst.npy', 1, "90,10,0")
    print(len(a))
    print(len(b))
    """
    import os
    data_path = data_prefix

    full_dataset = GPTSFTDataset(data_path)

    generator_ = torch.Generator().manual_seed(seed)

    # build split according to split string
    if splits_string is None:
        splits_string = '100,0,0'
    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    splits_sum = sum(splits)
    splits = [split / splits_sum for split in splits]

    if not shuffle:
        full_size = len(full_dataset)

        idxs = [0]
        cur_idx = 0
        for split in splits:
            cur_idx += math.floor(full_size*split)
            idxs.append(cur_idx)

        train_dataset = torch.utils.data.Subset(full_dataset, range(idxs[0], idxs[1]))
        valid_dataset = torch.utils.data.Subset(full_dataset, range(idxs[1], idxs[2]))
        test_dataset = torch.utils.data.Subset(full_dataset, range(idxs[2], idxs[3]))

    else:
        generator_ = torch.Generator().manual_seed(seed)
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, lengths=splits, generator=generator_)

    # for debugging purpose:
    # train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    #    full_dataset, lengths=[len(full_dataset)-200, 200, 0], generator=generator_)

    print(f"[{torch.distributed.get_rank() if torch.distributed.is_initialized() else -1 }] train_dataset: {train_dataset[0]['input_tokens'][:20]}")
    print(
        f"len(train_dataset) = {len(train_dataset)} / {len(valid_dataset)} / {len(test_dataset)}")

    assert len(train_dataset) > 0
    if len(valid_dataset) == 0:
        valid_dataset = None
    if len(test_dataset) == 0:
        test_dataset = None

    return (train_dataset, valid_dataset, test_dataset)
