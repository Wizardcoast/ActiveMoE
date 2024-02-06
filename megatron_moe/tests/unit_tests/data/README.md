### test env prepare

megatron-lm的运行需要编译，make sure你的环境是ok的。

### test corpus data prepare:


``` 

wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

unzip wikitext-103-raw-v1.zip

# get tokenizer_v2

python tools/preprocess_data.py --input /data3/jiaran/wikitext-103-raw/wiki.test.txt --output-prefix test_corpus --tokenizer-type PretrainedFromHF --tokenizer-name-or-path tokenizer_v2 --append-eod --workers=5 

# save test_corpus_text_document to the same dir with test_data.py

```

### test pad data prepare:
用来测试pad，直接写如下test_pad.jsonl：
{"text":"1234"}
{"text":"123457890123457890123457890123"}

这个数据一个padding，一个截断。都会被限制到20

```
python tools/preprocess_data.py --input test_pad.jsonl --output-prefix test_pad --tokenizer-type PretrainedFromHF --tokenizer-name-or-path tokenizer_v2  --append-eod --workers=1
--fixed-length=20

# save test_pad_text_document to the same dir with test_data.py


```

### run unit-test:
```
pytest test_data.py 
```


### functional test：

```
# 整体test，这个脚本会最小功能集的跑一个训练任务;
# 根据需要修改input data, 然后print数据、loss mask等信息。看一看在整体运行的时候，能否做对。
# 也许会跑不同的模型，cp到相关的目录下
cp examples/mvp_test.sh patch/megatron/examples/gptx
```