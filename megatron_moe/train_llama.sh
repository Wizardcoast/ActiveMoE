#!/bin/bash
set -x

MEGATRON_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/megatron/Megatron-Inf"
CODE_ROOT=${MEGATRON_PATH}

export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1  # 环境变量设置，开启Sequence parallel需打开

NNODES=${WORLD_SIZE} # 环境变量读取，无需手动设置，与DLC启动时设置的节点数量相关
NODE_RANK=${RANK}    # 环境变量读取，无需手动设置
GPUS_PER_NODE=${KUBERNETES_CONTAINER_RESOURCE_GPU} # 同上

# PAI平台推荐的NCCL环境变量设置，与通信相关，不建议修改
#export NCCL_IB_TC=136
#export NCCL_IB_SL=5
#export NCCL_IB_GID_INDEX=3
#export NCCL_SOCKET_IFNAME=bond0
#export NCCL_DEBUG=INFO
#export NCCL_IB_HCA=mlx5
#export NCCL_IB_TIMEOUT=22
#export NCCL_IB_QPS_PER_CONNECTION=32
#export NCCL_NET_PLUGIN=none

# 分布式启动参数，目前使用`python -m torch.distributed.launch`
DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

# 与LLaMA网络结构相关的配置
custom_options="--disable-bias-linear \
                --swiglu \
                --untie-embeddings-and-output-weights \
                --swiglu-make-ffn-hidden-size-divisible-by 256 \
                --position-embedding-type rope \
                --init-method-std 0.02 \
                --disable-scaled-init-method \
                --normalization RMSNorm \
                --norm-epsilon 1e-5 \
                "
# GPTx的配置可参考
# custom_options="--position-embedding-type rope \
#                 --normalization LayerNorm \
#                 --norm-epsilon 1e-5 \
#                 --init-method-std 0.02 \
#                 "

# 设置网络结构相关参数，这里是一个7B模型的设置
NUM_LAYERS=24
HIDDEN_SIZE=1024
FNN_HIDDEN_SIZE=4096
NUM_ATTN_HEADS=16

# micro-batchsize与global_batchsize设置，注意global_batchsize需被DP-size整除
BATCH_SIZE=8
GLOBAL_BATCH_SIZE=512

# 学习率
LR=3e-4
MIN_LR=3e-5

# 序列长度
SEQ_LEN=1024

# 混合精度训练设置
PR=bf16

# TP/PP设置，DP=GPU总数/TP/PP
TP=1  # 不超过8
PP=1  # 要能整除transformer-block的数量

AC="none" # Activation checkpointing
DO=true   # ZERO optimizer
FL=true   # Flash-attention
SP=false  # Sequence-parallel
SAVE_INTERVAL=1000  # Checkpoint保存的step间隔

# 设置数据路径
DATA_ROOT="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/data/refined_web_megatron/"
# 设置Tokenizer类型，常用：
# - "PretrainedFromHF": AutoTokenizer.from_pretrained(...), 设定tokenizer目录
# - "SentencePieceTokenizer": 设定tokenizer model路径
# - "NullTokenizer": 绕过tokenizer的真实路径设置，只需要给定词表大小；
#                    需在Megatron配置中额外加入`--vocab-size`
#                    这里词表大小要设置为真实词表"VOCAB_SIZE - 1", 因为NullTokenizer会自动pad一个token
TOKENIZER_TYPE = "<Tokenizer type>"
TOKENIZER_NAME_OR_PATH="<Path to tokenizer>"

# 数据路径的写法只需要写到basename即可，不用带扩展名.bin/.idx
# 先写tokens数，再写路径
DATASET_PATH=" \
    10  ${DATA_ROOT}/megatron_large_refined_web_0_text_document \
    20  ${DATA_ROOT}/megatron_large_refined_web_0_text_document \
    "

PRETRAIN_CHECKPOINT_PATH=none
TRAIN_TOKENS=130000000000     # 训练总token数
LR_DECAY_TOKENS=1000000000000 # 学习率decay的范围，1.0T tokens
WARMUP_TOKENS=$(( 2000 * ${GLOBAL_BATCH_SIZE} * ${SEQ_LEN} )) # warmup during 2000 iters

OUTPUT_BASEPATH="${MEGATRON_PATH}/save"  # 设置checkpoint/tensorboard的保存路径
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

# Megatron的配置
SEED=42
megatron_options="  \
        --continue-on-missing-checkpoint \
        --save ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --load ${SAVED_PRETRAIN_CHECKPOINT_PATH} \
        --split 100,0,0 \
        --data-path ${DATASET_PATH} \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --adam-eps 1e-8 \
        --clip-grad 1.0 \
        --weight-decay 0.1 \
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
        --seed ${SEED} \
        --vocab-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/tokenizer/gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/tokenizer/gpt2-merges.txt \
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