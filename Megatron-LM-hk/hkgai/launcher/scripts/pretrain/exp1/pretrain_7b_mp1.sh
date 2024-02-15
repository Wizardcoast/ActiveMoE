#!/bin/bash

# Runs the "HKGAI-7B" parameter model
SCRIPT_DIR=$(dirname $0)
MEGATRON_DIR=$(realpath ${SCRIPT_DIR}/../../../..)
echo $MEGATRON_DIR
export PYTHONPATH=$PYTHONPATH:$MEGATRON_DIR
echo $PYTHONPATH
export NCCL_SOCKET_IFNAME=ibp
export NCCL_IB_HCA=mlx5
# export NCCL_IB_GID_INDEX=3
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export GLOO_SOCKET_IFNAME=bond0

export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-1}
# WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

export GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=10.33.4.160
export MASTER_PORT=6000
export NNODES=2
export NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

export DP_SIZE=$((WORLD_SIZE / PP_SIZE / TP_SIZE))
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-1}
export GLOBAL_BATCH=$((DP_SIZE * MICRO_BATCH * GRAD_ACC_STEPS))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

HKG_MODELING_ARGS="
    --use-mcore-models \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --max-position-embeddings 4096 \
    --group-query-attention \
    --num-query-groups ${NUM_KV_HEADS:-4} \
    --swiglu \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --no-position-embedding \
    --attention-dropout 0 \
    --hidden-dropout 0\
    --disable-bias-linear
"

HKG_HYPER_PARAM_ARGS="
    --seed ${SEED:-42} \
    --seq-length 4096 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16 \
    --eod-mask-loss \
    --norm-epsilon 1e-5 \
    --lr 3e-4 \
    --min-lr 3e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 2000 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --init-method-std 0.02 \
    --override-opt_param-scheduler
"

HKG_TRAINING_ARGS="
    --num-workers 8 \
    --distributed-backend nccl \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --expert-model-parallel-size ${EP_SIZE:-1} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --optimizer adam \
    --train-iters ${TRAIN_ITERS:-793043} \
    --exit-interval ${EXIT_ITERS:-${TRAIN_ITERS:-793043}}
"

echo "[pretrain], begin..."
echo "[pretrain], WORLD_SIZE: $WORLD_SIZE, GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo "[pretrain], DP_SIZE: $DP_SIZE, TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE"
echo "[pretrain], Global batch size: $GLOBAL_BATCH, micro batch size: $MICRO_BATCH"

TASK_ID=${TASK_ID:-"Pretrain"}
JOB_NAME=hkg_7b_nl${NUM_LAYERS}_tp${TP_SIZE}_pp${PP_SIZE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}
OUTPUT_HOME="out/$JOB_NAME/$TASK_ID"
# CHECKPOINT_PATH="${OUTPUT_HOME}/checkpoint/"
CHECKPOINT_PATH=/workspace/megatron/checkpoints/save
WANDB_PATH="${OUTPUT_HOME}/"

DATA_PATH=my-gpt2_text_document
DATA_CACHE_PATH="my-gpt2_text_document"
TOKENIZER_MODEL_PATH="./tokenizer.model"

HKG_DATA_ARGS="
    --train-data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --split 1000,0,0
"
load_args=""
if [[ $(ls ${CHECKPOINT_PATH} 2> /dev/null | wc -l ) > 0 ]]; then
  load_args="--load ${CHECKPOINT_PATH}"
fi
CHECKPOINT_ARGS="
    --save $CHECKPOINT_PATH \
    $load_args
"
WANDB_ARGS="
    --wandb-project deepspeed \
    --wandb-exp-name frist_run \
    --wandb-save-dir ${WANDB_PATH} \
"

HKG_OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-iters 0 \
    --eval-interval 1000000 \
    --timing-log-level=0
"

torchrun $DISTRIBUTED_ARGS /workspace/megatron/hkgai/launcher/core/pretrain_gpt_new.py \
    $HKG_MODELING_ARGS \
    $HKG_HYPER_PARAM_ARGS \
    $HKG_TRAINING_ARGS \
    $HKG_DATA_ARGS \
    $HKG_OUTPUT_ARGS \
    $CHECKPOINT_ARGS \
    $WANDB_ARGS