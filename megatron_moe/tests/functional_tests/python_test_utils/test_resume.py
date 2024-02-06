import copy

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../')))

from megatron.arguments import parse_args
from tests.unit_tests.test_utilities import Utils

COMMON_TEST_PARAMS = ['script.py',
                      '--micro-batch-size', '1',
                      '--no-initialization',
                      '--no-async-tensor-model-parallel-allreduce',
                      '--train-iters', '350000',
                      '--no-load-rng',
                      '--position-embedding-type', 'rope',
                      '--bf16',
                      '--attention-softmax-in-fp32',
                      '--tokenizer-type', 'NullTokenizer',
                      '--vocab-size', '40831'
                      ]


def test_resume():
    test_ckpt = os.path.join(os.path.dirname(__file__), 'test_ckpt')
    params = copy.deepcopy(COMMON_TEST_PARAMS)
    params.append('--load')
    params.append(test_ckpt)
    sys.argv = params

    margs = parse_args()

    # checkpoint has iteration of 5. seq length 20, batch size 1
    # so iteration should be 5, lr should be warmup 100/10240 * 1e-5
    # consume samples should be 5

    model, iteration, scheduler = Utils.gpt_load_checkpoint(
        margs, load_scheduler=True)
    print(f"iteration: {iteration}")

    assert iteration == 5
    assert margs.consumed_train_samples == 5
    assert scheduler.get_lr() == 9.765625e-08
    assert scheduler.get_wd() == 0.1

    Utils._clean_global_vars()


def test_finetuning():
    test_ckpt = os.path.join(os.path.dirname(__file__), 'test_ckpt')
    params = copy.deepcopy(COMMON_TEST_PARAMS)
    params.append('--load')
    params.append(test_ckpt)

    # finetuning params
    params.append('--finetune')
    params.append('--no-load-optim')
    sys.argv = params

    margs = parse_args()

    model, iteration, scheduler = Utils.gpt_load_checkpoint(
        margs, load_scheduler=True)

    assert iteration == 0
    assert margs.consumed_train_samples == 0
    assert scheduler.get_lr() == 0.0
    assert scheduler.get_wd() == 0.1

    Utils._clean_global_vars()


def test_further_pretrain():
    test_ckpt = os.path.join(os.path.dirname(__file__), 'test_ckpt')
    params = copy.deepcopy(COMMON_TEST_PARAMS)
    params.append('--load')
    params.append(test_ckpt)

    # further pretrain params
    params.append('--reset-sample-stat')
    sys.argv = params

    margs = parse_args()

    model, iteration, scheduler = Utils.gpt_load_checkpoint(
        margs, load_scheduler=True)

    assert iteration == 0, f"iteration {iteration}"  # NOTE: take care
    assert margs.consumed_train_samples == 0
    assert scheduler.get_lr() == 9.765625e-08
    assert scheduler.get_wd() == 0.1

    Utils._clean_global_vars()


print("=> test_resume\n")
test_resume()
print("=> test_finetuning\n")
test_finetuning()
print("=> test_further_pretrain\n")
test_further_pretrain()
