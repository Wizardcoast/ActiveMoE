python -m pip install nltk 
DATA_ROOT=/workspace/rawdata/slimpajama/slimpajama/in_context_pretraining/sorting_output
JSONL_NAME=final_merged_sorted.jsonl

python tools/preprocess_data.py \
       --input $DATA_ROOT/$JSONL_NAME \
       --output-prefix slimpajama-icp-final_merged_sorted-8node \
       --tokenizer-model /workspace/megatron/baichuan.tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --workers 224 \
       --partitions 8 \
       --keep-sequential-samples \
       --append-eod 
