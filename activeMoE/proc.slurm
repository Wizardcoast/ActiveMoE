#!/bin/bash
#SBATCH -p megatron
#SBATCH -J Megatron-LM          # 队列名
#SBATCH --nodes 2  # 请求节点数
#SBATCH --cpus-per-task=64 # 任务所需cpu数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo SLURM_PROCID: $SLURM_PROCID
echo SLURM_NODEID: $SLURM_NODEID
export LOGLEVEL=INFO

PROJECT_ROOT=/home/songyanggao/ActiveMoE/activeMoE

set -x
# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

env;

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=${MASTER_ADDR} #localhost
MASTER_PORT=${MASTER_PORT} #6000
NNODES=2 #1
NODE_RANK=0    #0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DISTRIBUTED_ARGS="\
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS src/train_bash.py  \
    --deepspeed ds_zero3.json        --stage sft    \
    --finetuning_type full    --do_train    \
    --template default    --cutoff_len 3840    \
    --model_name_or_path /home/songyanggao/models/phi-2    \
    --dataset_dir data --dataset openorca    \
    --learning_rate 5e-5    \
    --overwrite_cache      \
    --num_train_epochs 1.0  \
    --ddp_find_unused_parameters False   \
    --plot_loss  \
    --overwrite_output_dir True  \
    --output_dir output_dense2moe/216train_phi_multinode  \
    --evaluation_strategy steps --eval_steps 5000  \
    --save_strategy steps --save_steps 5000 --save_total_limit 6   \
    --Dense2MoE False  --model_name phi   \
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/phi2_config.json\
    --per_device_train_batch_size 8    \
    --save_safetensors False    \
    --streaming True --dispatch_batches False \
    --max_step 30000    \
    --do_sample True  > logs/train_phi_multinode.log 2>&1 &

