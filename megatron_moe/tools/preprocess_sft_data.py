# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import argparse
import json
import multiprocessing
import time

from transformers import AutoTokenizer
import numpy as np 
import logging

"""
准备SFT数据:

SFT数据有几个特点：
1）SFT的数据更新频繁，需要更加灵活，所以走jsonl+numpy的方式
2）SFT时间比较短，不涉及掉卡之后resume
3）训练风格更想大模型之前的一份数据训练多次的情况
4）不用blend，sft数据多数都需要人手动的处理，所以数据集的混合都在形成jsonl时处理好
5）包含每个token的loss mask

example data:
{
    'index or id': 15
    text:['我不计算loss'，'我计算全量的loss'，'我loss weight 0.01'],
    loss_weight:[0,100,1]
}

整体流程如下：
1.read from jsonlines 
2.tokenize 
3.append eod
4.pad with pad token or truncate to max_length

EXAMPLE:
python tools/preprocess_data_sft.py --input /cpfs01/user/medgpt03/pretrain_data/flanv2.jsonl \
    --tokenizer-name-or-path tokenizer_v2/ \
    --output_path flanv2.npy \
    --max-seq-length 2048 \
    --workers 32

WARNING:root:SFT data is bigger than 2048, turncate! line index: 5077380
Processed 2670000 documents (18995.007286821503 docs/s).
total_proc = 2641934, skip_proc = 28265
No. of points in each bin :  [1359891  627574  239835  101644   64356  128484   87436   26918    3588
     732     350     250     164     107      98     507]
Size of the bins          :  [0.000e+00 2.560e+02 5.120e+02 7.680e+02 1.024e+03 1.280e+03 1.536e+03
 1.792e+03 2.048e+03 2.304e+03 2.560e+03 2.816e+03 3.072e+03 3.328e+03
 3.584e+03 3.840e+03 1.000e+10] 

"""

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_name_or_path)
        Encoder.max_seq_length = self.args.max_seq_length
        Encoder.append_eod = not self.args.no_append_eod
        Encoder.pad_id = Encoder.tokenizer.pad_token_id
        Encoder.eos_id = Encoder.tokenizer.eos_token_id


        assert Encoder.tokenizer.pad_token_id!=None, "pad_id is missing, please edit tokenizer"
        assert Encoder.tokenizer.eos_token_id!=None, "eos_id is missing, please edit tokenizer"



    def encode(self, line ):
        data = json.loads(line)
        max_seq_length = Encoder.max_seq_length

        index_key = 'index' if 'index'in data else 'id'
        first_chunk = data['text'][0]
        data_index = data[index_key] if index_key in data else first_chunk[0: min(20,len(first_chunk))]
        assert len(data['text'])== len(data['loss_weight']), f"text and loss_weight should have the same length. check {data_index}"

        tokens = []
        loss_mask = []
        attn_mask=[]
        truncated = 0

        #try:
        for i in range(len(data['text'])):
            text = data['text'][i]
            loss_weight = data['loss_weight'][i]
            text_ids = self.tokenizer.encode(text)
            tokens.extend(text_ids)
            loss_mask.extend([int(loss_weight)]*len(text_ids))
        #except:
        #    print(f'line index: {data_index} is malformed')
        #    return None

        orig_token_length = len(tokens)
        # turncation or pad
        if len(tokens)>Encoder.max_seq_length:
            # in most cases, should not turncation.
            logging.warning(f'SFT data is bigger than {max_seq_length}, turncate! line index: {data_index}')
            tokens = tokens[:max_seq_length+1]
            loss_mask = loss_mask[:max_seq_length+1]
            # if turncation, don't add eos as it is inappropriate
            truncated = 1
            attn_mask = [1] * (max_seq_length+1)

        else:
            # padding
            # append eos, as assume last sentence is Agent response, loss mask append 1
            if Encoder.append_eod:
                tokens.append(Encoder.eos_id)
                loss_mask.append(100)

            len_tokens = len(tokens)
            # pad
            tokens.extend([Encoder.pad_id] * (max_seq_length+1 - len_tokens))
            # attn_mask
            attn_mask = [1] * len_tokens
            attn_mask.extend([0] * (max_seq_length+1 - len_tokens))
            # loss mask 0
            loss_mask.extend([0] * (max_seq_length+1 - len_tokens))

        assert len(tokens)==max_seq_length+1 and len(loss_mask)==max_seq_length+1  , "should contains max_seq_length+1 tokens "

        """
        假设12 为Q， 234为A
        token:     1  2  3  4  5  eos pad pad
        loss_mask: 0  0  1  1  1   1   0   0
        target:    2  3  4  5 eos pad pad 
        loss_mask: 0  1  1  1  1   0   0
        attn_mask: 1  1  1  1  1   1   0   0

        convert to numpy array as it is more packed and default collate will convert them into tensors
        """

        sample = {
            'input_tokens': np.array(tokens[:-1], dtype=np.int32),
            'targets': np.array(tokens[1:], dtype=np.int32),
            'loss_mask': np.array(loss_mask[1:], dtype=np.float16),
            'truncated': int(truncated),
            'attn_mask': np.array(attn_mask[:-1], dtype=np.int8),
            'orig_token_length': orig_token_length

        }
        return sample

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str,
                       help='Path to input JSON')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")

    group.add_argument('--no-append-eod', action='store_true',
                       help='disable append eod for the last sentence and use loss weight 1')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output_path', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=10000,
                       help='Interval between progress updates')
    group.add_argument('--max-seq-length', default=-1, type=int, help="Maximum sequence length")
    args = parser.parse_args()
    args.keep_empty = False

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    encoder = Encoder(args)
   
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_samples = pool.imap(encoder.encode, fin, 64)
    #encoded_docs = map(encoder.encode, fin)

    print(f"Output prefix: {args.output_path}")

    startup_end = time.time()
    proc_start = time.time()
    
    print("Time to startup:", startup_end - startup_start)
    import scipy
    token_length_stats= []
    output_samples = []
    total_proc, skip_proc = 0, 0
    for i, sample in enumerate(encoded_samples, start=1):
        if sample is None:
            skip_proc+=1
            continue

        has_loss_mask = np.sum(sample['loss_mask']==0)!=sample['loss_mask'].shape[0]        
        if not has_loss_mask:
            skip_proc+=1
            continue
        total_proc += 1
        orig_token_length = sample.pop('orig_token_length')
        token_length_stats.append(orig_token_length)
        output_samples.append(sample)

        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            print(f"Processed {i} documents",
                f"({i/elapsed} docs/s).",
                file=sys.stderr)

    print(f"total_proc = {total_proc}, skip_proc = {skip_proc}")
    print(f"total_tokens = {sum(token_length_stats)}")
    bins = list(range(0,args.max_seq_length,256))
    bins.append(int(1e10))
    hist, bin_edges = np.histogram(token_length_stats,bins=bins)

    # Checking the results
    print ("No. of points in each bin : ", hist)
    print ("Size of the bins          : ", bin_edges)

    np.save(args.output_path, output_samples)


if __name__ == '__main__':
    main()