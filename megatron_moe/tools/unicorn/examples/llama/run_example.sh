#!/bin/bash
set -e

MEGATRON_PATH=< path to megatron-lm >
CODE_ROOT="${MEGATRON_PATH}"
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=12345

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

custom_options="--disable-bias-linear \
                --swiglu \
                --untie-embeddings-and-output-weights \
                --swiglu-make-ffn-hidden-size-divisible-by 256 \
                --position-embedding-type rope \
                --normalization RMSNorm \
                --norm-epsilon 1e-5 \
                --init-method-std 0.02 \
                --disable-scaled-init-method \
                "

# Llama tokenizers and use NullTokenizer
VOCAB_SIZE=$(( 32000 - 1 ))
NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=128
LR=3e-4
MIN_LR=3e-5
SEQ_LEN=1024
PR=fp16
TP=2
PP=2
AC="sel"
DO=true
FL=true
SP=false
SAVE_INTERVAL=1000000

DATA_ROOT="${MEGATRON_PATH}/tools/unicorn/tests/data/"
TOKENIZER_NAME_OR_PATH="/path/to/tokenizer"
DATASET_PATH=" \
    ${DATA_ROOT}/sample_llama_text_document \
    "
PRETRAIN_CHECKPOINT_PATH=none
TRAIN_TOKENS=10000000000    # 10B tokens
LR_DECAY_TOKENS=10000000000 # 10B tokens
WARMUP_TOKENS=$(( 10 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} )) # iter-2000

OUTPUT_BASEPATH="${MEGATRON_PATH}/test"
if [ $PRETRAIN_CHECKPOINT_PATH != none ]; then
    load_options=" \
        --load $PRETRAIN_CHECKPOINT_PATH"
fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
        --recompute-method uniform \
        --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
                    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
        --fp16 \
        --initial-loss-scale 65536"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
fi

if [ $DO = true ]; then
    do_options=" \
        --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
        --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
        --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi

TRAIN_ITERS=$(( ${TRAIN_TOKENS} / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${LR_DECAY_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME=""
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
LOAD_PATH="${MEGATRON_PATH}/tools/unicorn/llama-megatron"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --load ${LOAD_PATH} \
        --split 99.5,0.5,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-5 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --log-interval 1 \
        --eval-interval ${SAVE_INTERVAL} \
        --eval-iters 50 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 8 \
        --seed 888 \
        --tokenizer-type NullTokenizer \
        --vocab-size ${VOCAB_SIZE} \
        "

cd ${MEGATRON_PATH}

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${CODE_ROOT}/pretrain_gpt.py \
         ${megatron_options} \
         ${activation_checkpoint_options} \
         ${do_options} \
         ${pr_options} \
         ${sp_options} \
         ${flash_options} \
         ${load_options} \
         ${custom_options} \
         "

echo ${run_cmd}
eval ${run_cmd}

