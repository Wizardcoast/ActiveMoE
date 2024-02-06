#!/bin/bash
set -ex
ENV="dsw"
# cicd 入口就是megatron path
MEGATRON_PATH=$(pwd)
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=${1}
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $NNODES \
                  --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR \
                  --master_port $MASTER_PORT"
MODEL_SIZE="xB"
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=2048
LR=2e-4
MIN_LR=1e-5
SEQ_LEN=128
#bf16 fp16 etc
PR=${2} 
TP=${3}
PP=${4}
AC="none"
DO=${5}
FL=${6}
SP=false
SAVE_INTERVAL=5000

# sst2.npy is made by 
# python tools/preprocess_data_sft.py --input sst_train2.jsonl     
# --tokenizer-name-or-path tokenizer_v2/     
# --output_path sst2.npy     --max-seq-length 128     --workers 2

DATASET_PATH="${TEST_DATA_PATH}/sst2.npy"
TOKENIZER_NAME_OR_PATH="${TEST_DATA_PATH}/tokenizer_v2"
PRETRAIN_CHECKPOINT_PATH=none
TRAIN_TOKENS=7000000
WARMUP_TOKENS=102400
OUTPUT_BASEPATH="${MEGATRON_PATH}/save"

NUM_LAYERS=8
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=8


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
NAME="${ENV}-pretrain-megatron-llama-${MODEL_SIZE}-lr-${LR}-bs-${BATCH_SIZE}-seqlen-${SEQ_LEN}-pr-${PR}-tp-${TP}-pp-${PP}-ac-${AC}-do-${DO}-sp-${SP}-tt-${TRAIN_TOKENS}-wt-${WARMUP_TOKENS}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/"
mkdir -p "${OUTPUT_BASEPATH}/log/"
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard/${NAME}_${current_time}"
mkdir -p ${TENSORBOARD_DIR}
SAVED_PRETRAIN_CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/${NAME}"
megatron_options="  \
        --finetune
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 98,2,0 \
        --train-data-path ${DATASET_PATH} \
        --dataloader-type cyclic \
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
        --eval-interval ${SAVE_INTERVAL} \
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
        --seed 42 \
        --position-embedding-type rope \
        --tokenizer-type PretrainedFromHF \
        --tokenizer-name-or-path ${TOKENIZER_NAME_OR_PATH} \
        "

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS sft_gpt.py \
         ${megatron_options} \
         ${activation_checkpoint_options} \
         ${do_options} \
         ${pr_options} \
         ${sp_options} \
         ${flash_options} \
         ${load_options}"
echo ${run_cmd}
eval ${run_cmd}
set +x

