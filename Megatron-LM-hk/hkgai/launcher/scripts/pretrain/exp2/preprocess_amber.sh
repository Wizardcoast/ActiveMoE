
python -m pip install nltk 
python tools/preprocess_data_amber.py \
       --input /workspace/dataset/filtered_amber_full.jsonl \
       --output-prefix full-amber-8node \
       --tokenizer-model /workspace/megatron/amber.tokenizer.model \
       --tokenizer-type Llama2Tokenizer \
       --vocab-file /workspace/dataset/llama.vocab.json \
       --json-keys token_ids \
       --workers 224 \
       --partitions 8 \
       --append-eod 
