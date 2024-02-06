#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_LAUNCH_BLOCKING=1

MASTER_ADDR=localhost
MASTER_PORT=12345 # $(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=8

ROOT=< path to the root directory of megatron >
MEGATRON_PATH="${ROOT}/Megatron-LM"
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

custom_options=" \
               "

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

# Hyper-parameters
BATCH_SIZE=1
GLOBAL_BATCH_SIZE=32
LR=3e-5
MIN_LR=3e-6
# GPTx-13B
NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
SEQ_LEN=256
PR=bf16
TP=2
PP=2
AC="sel"
DO=true
FL=true
SP=true
SAVE_INTERVAL=20
DATASET_ROOT=< path to the root of dataset >
DATASET_PATH=${DATASET_ROOT}/< path to dataset >
PRETRAIN_CHECKPOINT_PATH=none
TRAIN_TOKENS=$(( (${SAVE_INTERVAL} * 5 - 1) * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} ))
WARMUP_TOKENS=$(( 10 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} ))
OUTPUT_BASEPATH="finetune"

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
        --fp16"
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
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

NAME="test"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"

TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --load ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 98,2,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style linear \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.006 \
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
        --eval-interval 100 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 8 \
        --seed 1234 \
        --position-embedding-type rope \
        --tokenizer-type NullTokenizer \
        --vocab-size 81919 \
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS ${MEGATRON_PATH}/pretrain_gpt.py \
         ${megatron_options} \
         ${activation_checkpoint_options} \
         ${do_options} \
         ${pr_options} \
         ${sp_options} \
         ${flash_options} \
         ${load_options} \
         ${custom_options}"

echo ${run_cmd}
eval ${run_cmd}

set +x
