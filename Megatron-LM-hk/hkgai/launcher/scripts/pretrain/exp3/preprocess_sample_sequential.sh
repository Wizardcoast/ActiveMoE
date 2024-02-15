python -m pip install nltk 

python tools/preprocess_data.py \
       --input /workspace/dataset/test/sample.json \
       --output-prefix debug-sample-data-sequential \
       --tokenizer-model /workspace/megatron/baichuan.tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --workers 128 \
       --partitions 8 \
       --keep-sequential-samples \
       --append-eod 
