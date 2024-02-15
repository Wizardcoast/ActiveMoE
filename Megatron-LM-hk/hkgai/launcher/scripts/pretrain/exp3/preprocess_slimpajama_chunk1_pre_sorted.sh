python -m pip install nltk 
DATA_ROOT=/workspace/rawdata/slimpajama/slimpajama/in_context_pretraining/sorting_output
JSONL_NAME=chunk1_pre_sorted.jsonl

python tools/preprocess_data.py \
       --input $DATA_ROOT/$JSONL_NAME \
       --output-prefix slimpajama-icp-chunk1_pre_sorted-8node \
       --tokenizer-model /workspace/megatron/baichuan.tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --workers 128 \
       --partitions 8 \
       --append-eod 
