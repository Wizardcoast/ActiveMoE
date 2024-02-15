#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN

CHECKPOINT_PATH=/workspace/megatron/checkpoints/save
VOCAB_FILE=/workspace/megatron/checkpoints/gpt2/gpt2-vocab.json
MERGE_FILE=/workspace/megatron/checkpoints/gpt2/gpt2-merges.txt
DATA_PATH=my-gpt2_text_document

GPT_ARGS="
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 1 \
    --global-batch-size 2 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

torchrun pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH