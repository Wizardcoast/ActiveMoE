#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=6000

# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/save/megatron_ckpt_1b3_8_top2_500b"
TOKENIZER_TYPE="PretrainedFromHF"
TOKENIZER_NAME_OR_PATH="/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/save_hf/tokenizer"

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
COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 2048 \
                  --num-attention-heads 16 \
                  --seq-length 4096 \
                  --max-position-embeddings 4096 \
                  --fp16 \
                  --tokenizer-type ${TOKENIZER_TYPE} \
                  --tokenizer-name-or-path ${TOKENIZER_NAME_OR_PATH}"
                  
MOE_ARGUMENTS="\
--moe-num-experts=8 \
--moe-loss-weight=0.1 \
--moe-top-k=2 \
--moe-capacity-factor 0 \
--mlp_type=glu \
--add_moe_bias"
#pip install flask-restful

torchrun -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       $custom_options \
       $COMMON_TASK_ARGS \
       $MOE_ARGUMENTS \
       --load ${CHECKPOINT_PATH}  \
       --micro-batch-size 1  \
       --seed 42 \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng