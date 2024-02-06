# coding:utf-8


import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

import torch
import numpy as np

from transformers import AutoTokenizer

from megatron.data.indexed_dataset import MMapIndexedDataset
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.gpt_dataset import GPTDataset
from megatron.tokenizer.tokenizer import _AutoTokenizer


TEST_DATA_PATH=os.environ.get('TEST_DATA_PATH','tests/unit_tests/data/')
TOKENIZER_PATH=os.path.join(TEST_DATA_PATH,'tokenizer_v2')

TEST_CORPUS_DATA_PATH=os.path.join(TEST_DATA_PATH,'test_corpus_text_document')
TEST_PAD_DATA_PATH=os.path.join(TEST_DATA_PATH,'test_pad_text_document')


def test_indexed_dataset():

    #path, impl, skip_warmup=False

    idx_dataset = MMapIndexedDataset(TEST_CORPUS_DATA_PATH, False)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)

    assert np.array_equal(idx_dataset[0],np.array([ 1377, 408 ,11624 ,  422  ,7215 ,  380 ,  408 ,  211   ,  1]))
    print(idx_dataset[0])
    # first document of test_corpus_text_document. {"text": " \n = Robert Boulter = "}
    assert 'Robert Boulter' in tokenizer.decode(idx_dataset[0])
    print(tokenizer.decode(idx_dataset[0]))


def test_gpt_dataset():
    from argparse import Namespace
    from megatron.global_vars import set_args

    args = Namespace()
    args.tp_shared_dataloader = False
    set_args(args)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    indexed_dataset = MMapIndexedDataset(TEST_CORPUS_DATA_PATH, False)

    total_num_of_documents = indexed_dataset.sizes.shape[0]


    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    torch.distributed.init_process_group(
        backend='nccl',
        world_size=1, rank=0)

    from megatron.core import mpu

    if mpu.is_unitialized():
        mpu.initialize_model_parallel()

    gpt_dataset = GPTDataset(name='test',data_prefix='test_corpus_text_document',documents=documents,
    indexed_dataset=indexed_dataset,splits_string="split",num_samples=10,seq_length=128,seed=0,shuffle=True)

    print('test1:shuffled')
    print(gpt_dataset[0])
    print(tokenizer.decode(gpt_dataset[0]['text']))

    gpt_dataset = GPTDataset(name='test',data_prefix='test_corpus_text_document',documents=documents,
    indexed_dataset=indexed_dataset,splits_string="split",num_samples=10,seq_length=128,seed=1,shuffle=False)

    print('test2:no shuffle')
    print(gpt_dataset[0])
    print(tokenizer.decode(gpt_dataset[0]['text']))
    assert '\n = Robert Boulter =' in tokenizer.decode(gpt_dataset[0]['text'])

    print(tokenizer.decode([ 1377,   408, 11624,   422,  7215,   380,   408,   211,     1, 11624,
           422,  7215,   380,   325,   276,  6578,  3832,  1488, 10681,   288,
         27010, 18850,  1178,  1085,   825,   249, 14817,  1606,    16,    35,
         41304,  3781,   350,   255, 10681,  4217,   407, 12068,   278,   211,
            21,    19,    19,    19,  1178,  1066,   434,  5411,   468,   249,
         41304,  3781,   278,   255,  1717,  5662,  1415,  5271,   468, 21982,
         14250,  8383,  1488,   680,   434,  3739,   278,   211,    21,    19,
            19,    20,   442,   255, 13949,  6037, 24809,  1178,  1085,   825,
           249, 14817,  3781,   278,   255, 10681,  4217, 19241,  4382,  2219,
           269,   278,   211,    21,    19,    19,    21,  1178,   584,   211,
            21,    19,    19,    23,   422,  7215,   380, 27523,   249,  3781,
           381,   428, 34780,   428,   278,   255, 14412,   428, 74280,   802,
            86, 24023,   428,   283,   255, 10681,  4217,   407, 10957]))

    print('test3 : pad')
    """
    如果每个doc被pad成128，且seq_len=128，那么合理的预期是GPTDataset不会再有随机性
    每个sample_idx的offset都为0，也不用进一步拼接
    每个batch数据为：
    tokens_=   0-   127 0_
    label =    1-127 0_
    """
    pad_indexed_dataset = MMapIndexedDataset(TEST_PAD_DATA_PATH, False)

    print(pad_indexed_dataset[0])
    total_num_of_documents = pad_indexed_dataset.sizes.shape[0]

    documents = np.arange(start=0, stop=total_num_of_documents,
                        step=1, dtype=np.int32)

    pad_gpt_dataset = GPTDataset(name='test',data_prefix='test_pad_text_document',documents=documents,
    indexed_dataset=pad_indexed_dataset,splits_string="split",num_samples=2,seq_length=20,seed=1,shuffle=False)

    #{"text":"1234"}
    #{"text":"123457890123457890123457890123"}
    print(pad_gpt_dataset[0])
    print(pad_gpt_dataset[1])
    # num_samples=2 will double the dataset, which means it has 4
    print(pad_gpt_dataset[2])

    assert np.array_equal(pad_gpt_dataset[0]['text'],np.array([20, 21, 22, 23,  1,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
        3,  3,  3, 20]))
    assert np.array_equal(pad_gpt_dataset[1]['text'],np.array([20, 21, 22, 23, 24, 26, 27, 28, 19, 20, 21, 22, 23, 24, 26, 27, 28,
       19, 20, 21, 20]))

    print('test4 : blendable dataset')
    # 2个dataset, 100个sample, 1:99的混合比例
    bds = BlendableDataset([pad_gpt_dataset,gpt_dataset],weights=[0.01,0.99],size=100)
    ds_cnt=[0,0]

    for data in bds:
        ds_cnt[data['dataset_idx']]+=1

    assert ds_cnt[0]==1
    assert ds_cnt[1]==99


def test_tokenizer():
    mlm_tokenizer = _AutoTokenizer(tokenizer_name_or_path=TOKENIZER_PATH, vocab_extra_ids=-1)
    token_ids = mlm_tokenizer.tokenize_as_chunk(text='hello your guys',seq_length=20)
    print(token_ids)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../tools')))

    from preprocess_data import Encoder
    from dataclasses import dataclass
    # mock args
    @dataclass
    class TEST_ARGS(object):
        tokenizer_type = 'PretrainedFromHF'
        tokenizer_name_or_path=TOKENIZER_PATH
        append_eod = True
        fixed_length = None
        json_keys = ['text']
        split_sentences = False
        rank=0
        vocab_extra_ids=0
        make_vocab_size_divisible_by=128
        tensor_model_parallel_size=1

    args =TEST_ARGS()

    encoder= Encoder(args)
    encoder.initializer()

    #({'text': [1923, 43804, 27643, 9823, 1341, 1]}, 19)
    #({'text': [1923, 43804, 27643, 9823, 1341, 1, 3, 3, 3, 3, 3, 3]}, 19)
    print(encoder.encode(r'{"text":"让火焰净化一切！"}')[0]['text'])
    assert encoder.encode(r'{"text":"让火焰净化一切！"}')[0]['text'] == [1923, 43804, 27643, 9823, 1341, 1]

    args.fixed_length = 12
    assert encoder.encode(r'{"text":"让火焰净化一切！"}')[0]['text'] == [1923, 43804, 27643, 9823, 1341, 1, 3, 3, 3, 3, 3, 3]

    print(encoder.encode(r'{"text":"让火焰净化一切！"}'))


def test_get_ltor_masks_and_position_ids():
    from megatron.utils import  get_ltor_masks_and_position_ids

    mlm_tokenizer = _AutoTokenizer(tokenizer_name_or_path=TOKENIZER_PATH, vocab_extra_ids=-1)
    tokens_= torch.tensor([[1923, 43804, 27643, 9823, 1341, 1, 3, 3, 3, 3, 3, 3, 1340]])
    tokens = tokens_[:,:-1]
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(tokens,
                    mlm_tokenizer.eod,
                    False,
                    False,
                    False)

    # set all the pad token mask to 0

    pad_mask = torch.logical_and(tokens_[:,:-1]!=mlm_tokenizer.pad_token_id , tokens_[:,1:]!=mlm_tokenizer.pad_token_id)
    loss_mask = loss_mask * pad_mask
    print(loss_mask)
    assert np.array_equal(loss_mask.numpy(),np.array([[1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0. ,0]]))
