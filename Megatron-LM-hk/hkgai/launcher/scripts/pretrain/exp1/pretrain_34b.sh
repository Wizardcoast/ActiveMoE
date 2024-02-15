#!/bin/bash

# Runs the "HKGAI-34B" parameter model
SCRIPT_DIR=$(dirname $0)
MEGATRON_DIR=$(realpath ${SCRIPT_DIR}/../../../..)
export PYTHONPATH=$PYTHONPATH:$MEGATRON_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ibp
export NCCL_IB_HCA=mlx5

# envs from volces/pai/arsenal/default
export GPUS_PER_NODE=${MLP_WORKER_GPU:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-8}}
export NNODES=${MLP_WORKER_NUM:-${WORLD_SIZE:-1}}
export NODE_RANK=${MLP_WORKER_RACK_RANK_INDEX:-${MLP_ROLE_INDEX:-${RANK:-0}}}
export MASTER_ADDR=${MLP_WORKER_0_HOST:-${MASTER_ADDR:-127.0.0.1}}
export MASTER_PORT=${MLP_WORKER_0_PORT:-${MASTER_PORT:-1234}}

WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))
export DP_SIZE=$((WORLD_SIZE / PP_SIZE / TP_SIZE))
export MICRO_BATCH=${MICRO_BATCH:-1}
export GRAD_ACC_STEPS=${GRAD_ACC_STEPS:-4}
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
    --num-layers ${NUM_LAYER:-60} \
    --hidden-size ${HIDDEN_SIZE:-7168} \
    --ffn-hidden-size ${FFN_HIDDEN_SIZE:-20480} \
    --num-attention-heads ${NUM_ATTN_HEAD:-56} \
    --max-position-embeddings ${SEQ_LEN:-4096} \
    --group-query-attention \
    --num-query-groups ${NUM_KV_HEADS:-8} \
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
    --seq-length ${SEQ_LEN:-4096} \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --bf16
    --norm-epsilon 1e-6 \
    --lr 1.5e-4 \
    --min-lr 1.5e-5 \
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
    --tensor-model-parallel-size ${TP_SIZE:-1} \
    --pipeline-model-parallel-size ${PP_SIZE:-1} \
    --sequence-parallel \
    --use-distributed-optimizer \
    --optimizer adam \
    --use-flash-attn \
    --train-iters ${TRAIN_ITERS:-793043}
    --exit-interval ${EXIT_ITERS:-${TRAIN_ITERS:-793043}}
"

HKG_DATA_ARGS="
    --train-data-path $DATA_PATH \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL_PATH} \
    --split 1000,0,0
"

echo "[pretrain], begin..."
echo "[pretrain], WORLD_SIZE: $WORLD_SIZE, GPUS_PER_NODE: $GPUS_PER_NODE, NNODES: $NNODES"
echo "[pretrain], DP_SIZE: $DP_SIZE, TP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE"
echo "[pretrain], Global batch size: $GLOBAL_BATCH, micro batch size: $MICRO_BATCH"

TASK_ID=${MLP_TASK_ID:-"Pretrain"}
JOB_NAME=hkg_nl${NUM_LAYER}_tp${TP_SIZE}_pp${PP_SIZE}_mb${MICRO_BATCH}_gb${GLOBAL_BATCH}
OUTPUT_HOME="out/$JOB_NAME/$TASK_ID"
CHECKPOINT_PATH="${OUTPUT_HOME}/checkpoint/"
WANDB_PATH="${OUTPUT_HOME}/"

# --save $CHECKPOINT_PATH \
# --load $CHECKPOINT_PATH
HKG_OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-iters 0 \
    --eval-interval 1000000 \
    --wandb-project test_megatron \
    --wandb-exp-name test_wandb \
    --wandb-save-dir ${WANDB_PATH} \
    --timing-log-level=0
"

torchrun $DISTRIBUTED_ARGS $MEGATRON_DIR/hkgai/launcher/core/pretrain_gpt.py \
    $HKG_MODELING_ARGS \
    $HKG_HYPER_PARAM_ARGS \
    $HKG_TRAINING_ARGS \
    $HKG_DATA_ARGS \
    $HKG_OUTPUT_ARGS
