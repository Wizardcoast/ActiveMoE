# SFT

## Prepare Data

1. 原数据格式

{"text": ["bad film you thought was going to be really awful Answer:", "0"], "loss_weight": [1, 100]}

text有2个list，loss_weight有2个list。二者一定要一致。
预处理会将数据拼好，同时确认好loss mask
```
text:       bad film you thought was going to be really awful Answer:0
loss-weight: 1   1    1    1      1    1    1  1   1      1    1     100
```
最终loss-weight会* loss-weight-factor，默认为0.01.

2. 二进制化
为了最大复用megatron的逻辑，也将数据进行二进制化。SFT的二进化目前会pad或者truncate到max-sequence-length。
区别有2:
- 为了方便debug和修改，采用npy。可以直接加载不用依赖框架的mmap dataset
- 准备attention mask和loss mask，去标明哪些token需要按照多少weight进行训练
```
python tools/preprocess_data_sft.py --input /home/jiaran/sft/data/sst_train2.jsonl     --tokenizer-name-or-path /home/jiaran/Megatron-LM/tests/unit_tests/data/tokenizer_v2/     --output_path sst2.npy     --max-seq-length 128     --workers 2 
```

注意：如果有些情况，loss-weight为0，比如需要train的token在最后被截断了，预处理会直接丢弃。因为mlm不容忍loss为0的情况。


## Run SFT

按照上述准备好数据之后，直接运行训练。具体配置可以参考sft_gpt.sh里的内容。

注意：sft要求必须自行制定train、valid、test。

dataloader type
* cyclic：框架会假设数据只有1个epoch，会自动循环+shuffle
* single：框架会假设数据shuffle和拼接均由上游完成，复制dataset到num-samples保证不报错，除此之外不做其他任何改动

``` 
sh examples/sft_gpt.sh
```


## RoadMap

为了提速，会陆续支持pack到一起的sft：
1）支持flash-attn compatible的attention-mask