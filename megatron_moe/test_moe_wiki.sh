#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

TASK="WIKITEXT103"

VALID_DATA="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megatron_eval_data/wikitext-103/wiki.test.tokens"

#CHECKPOINT_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/checkpoint/trail-1"
CHECKPOINT_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/checkpoint/dmoe_LLaMA_1.3Bx8_top2_3.3t"
TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/public/jiaran/tokenizer_v3"

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 2048 \
                  --num-attention-heads 16 \
                  --seq-length 4096 \
                  --max-position-embeddings 4096 \
                  --fp16 \
                  --tokenizer-type ${TOKENIZER_TYPE} \
                  --tokenizer-name-or-path ${TOKENIZER_NAME_OR_PATH}"


DISTRIBUTED_ARGS="--nproc_per_node 8 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

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
MOE_ARGUMENTS="\
--moe-num-experts=8 \
--moe-loss-weight=0.1 \
--moe-top-k=2 \
--moe-capacity-factor 0 \
--mlp_type=glu \
--add_moe_bias \
--moe_num_layers 24"

python -m torch.distributed.launch $DISTRIBUTED_ARGS /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/megatron/Megatron-Inf/tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       $custom_options \
       $MOE_ARGUMENTS \
       --valid-data $VALID_DATA \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng