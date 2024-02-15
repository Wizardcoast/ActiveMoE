python -m pip install nltk 
python tools/preprocess_data.py \
       --input /workspace/data/slimpajama.jsonl \
       --output-prefix full-gpt2-8node \
       --tokenizer-model /workspace/megatron/tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --workers 64 \
       --partitions 8 \
       --append-eod 
python -m pip install nltk 
python tools/preprocess_data.py \
       --input /workspace/dataset/sample.json \
       --output-prefix eod-gpt2 \
       --tokenizer-model /workspace/megatron/tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --workers 64 \
       --partitions 8 
       # --append-eod 

# python tools/preprocess_data.py \
#        --input /aifs4su/data/rawdata/yubo_slimpajama/slimpajama.jsonl \
#        --output-prefix /aifs4su/data/rawdata/yubo_slimpajama/real_baichuan2 \
#        --tokenizer-type Llama2Tokenizer \
#        --tokenizer-model /home/yubowang/PTM/Baichuan2-7B-Chat/tokenizer.model \
#        --vocab-file /home/yubowang/PTM/Baichuan2-7B-Chat/vocab.json \
#        --workers 64 \
#        --partitions 4 \
#        --split-sentences