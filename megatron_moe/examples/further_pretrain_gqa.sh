#!/bin/bash
set -e

# A100 on cf-a
ENV="dlc"

MEGATRON_PATH="< path to the root directory of Megatron-LM >"
DATASET_PATH="< path to the Data >"
CHECKPOINT_PATH="< path to the root directory of checkpoints to be loaded >"
# TOKENIZER_NAME_OR_PATH="< path to the tokenizer model >"  # If not set, should use NullTokenizer and set vocab-size.

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

# For local test.
export WORLD_SIZE=1
export RANK=0
export KUBERNETES_CONTAINER_RESOURCE_GPU=8
export MASTER_ADDR=localhost
export MASTER_PORT=60000

NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU}

# Envs for cluster training.
#export NCCL_IB_TC=136
#export NCCL_IB_SL=5
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=mlx5
#export NCCL_IB_TIMEOUT=22
#export NCCL_IB_QPS_PER_CONNECTION=32
#export NCCL_NET_PLUGIN=none
#export CUDA_LAUNCH_BLOCKING=1

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"

custom_options="--position-embedding-type rope \
                --distributed-timeout-minutes 30 \
                --override-opt_param-scheduler \
                --reset-sample-and-iteration-stat \
                --tokenizer-type NullTokenizer \
                --vocab-size 81919
                "

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
BATCH_SIZE=1

GLOBAL_BATCH_SIZE=480 # Stage-2 1472 46x8

LR=3e-4
MIN_LR=3e-5
SEQ_LEN=4096
PR=bf16
TP=2
PP=1
AC="none"
DO=true
FL=true
SP=true
SAVE_INTERVAL=40


TRAIN_TOKENS=57000000000 # stage-2 570B
LR_DECAY_TOKENS=1024000000000 # 1.024T tokens

# Since the global_batch_size mighe be different between two stages,
# there might be value check failed, you should
# use --override-opt_param-scheduler
WARMUP_TOKENS=$(( ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} * 2000 ))

OUTPUT_BASEPATH="${MEGATRON_PATH}/cpfs_test_p2"
echo "Saving to ${OUTPUT_BASEPATH} ..."

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

mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
#current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p ${TENSORBOARD_DIR}

SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/"

megatron_options="  \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 99.5,0.5,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --weight-decay 0.1 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
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
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --num-workers 8 \
        --seed 42 \
        --load ${CHECKPOINT_PATH} \
        --finetune \
        --num-query-groups 8 \
        "

cd ${MEGATRON_PATH}

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS pretrain_gpt.py \
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
