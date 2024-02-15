# Copyright (c) 2023, HKGAI CORPORATION. All rights reserved.

"""Megatron extra arguments."""

import argparse


def add_hkgai_args(parser: argparse.ArgumentParser):
    parser = _add_data_args(parser)

    return parser


def _add_data_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title='data argument group')

    group.add_argument('--tokenizer-alpha', type=float, default=0,
                       help='Sentencepiece encode alpha.')
    group.add_argument('--add-bos', action='store_true',
                       help='Add bos token in each sample')
    group.add_argument('--enable-shuffle', action='store_true',
                          help='Enable shuffle of the data')

    return parser
