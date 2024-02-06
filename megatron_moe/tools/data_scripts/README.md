
### PAI上进行数据处理
这个文件包含了在PAI上进行分布式数据处理所用到的脚本。

利用PAI DLC任务的分布式能力，一般32C-256G的instance申请48个，可以在2个小时内产出1T+的token。

一共有3步：

1. 将jsonl数据放在cpfs上

结构如下：
不同的category下有多个jsonl的file，每个最好不要超过2G
```
wiki/
    0001.jsonl
    0002.jsonl

book/
    0001.jsonl
    0002.jsonl
    0003.jsonl
```

2. 二进制化数据：
使用tokenizer，将数据二进制化，该脚本会启动X个instance，每个instance会处理自己对应的数据。
产出的bin和idx会写入output_path。

"""
cd MEGATRON-PATH

pip install sentencepiece
set -ex
# world_size
export RC_WORLD_SIZE=X

DATA_ROOT=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/ryan/data/oss_output_webtext_1108_jsonled;
TOKENIZER=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/jiaran/tokenizer_v3/
OUTPUT_PATH=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/jiaran/data/tmp_1111
mkdir -p $OUTPUT_PATH
python tools/data_scripts/pai_preprocess_data.py --data_root ${DATA_ROOT} --tokenizer $TOKENIZER --category book-zlib_pdf_1108  webtext-quora  webtext-redpajama_v2  webtext-skypile-150b  --output_path $OUTPUT_PATH --v v_1111
"""

输出格式如下：
```
格式如下：
wiki/
    0001_v_1111.bin
    0001_v_1111.idx
    0002_v_1111.bin
    0002_v_1111.idx
book/
    0001_v_1111.bin
    0001_v_1111.idx
    ...

```

3.  Merge
将2所产出的每个category的分片数据，进行merge，最终输出到output path。每个category一个文件，最终即C个文件。这个也是megatron训练的输入。

此任务一个worker就够了。

Example:
"""
cd MEGATRON-PATH

pip install sentencepiece
set -ex
DATA_ROOT=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/jiaran/data/tmp_1028
TOKENIZER=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/jiaran/tokenizer_v3/
OUTPUT_PATH=/cpfs01/projects-HDD/AI4S_public_queue_1_HDD/public/jiaran/data/data_v1111

export RC_WORLD_SIZE=1

python tools/data_scripts/pai_merge_data.py --data_root $DATA_ROOT  --category book-zlib_pdf_1108  webtext-quora  webtext-redpajama_v2  webtext-skypile-150b   --v v_1111 --output_path $OUTPUT_PATH --workers 32
"""
