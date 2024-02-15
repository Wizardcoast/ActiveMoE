#!/bin/bash
#SBATCH -p megatron
#SBATCH -J Megatron-LM          # 队列名
#SBATCH -N 4                    # 节点数量
#SBATCH -A root                 # root授权
#SBATCH --cpus-per-task=200       # 任务所需cpu数量
#SBATCH --mem=1024G      
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --exclude=dgx-067,dgx-071,dgx-095


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo SLURM_PROCID: $SLURM_PROCID
echo SLURM_NODEID: $SLURM_NODEID
export LOGLEVEL=INFO
export WANDB_API_KEY=YourOwnWandbAPIKey


PROJECT_ROOT=/aifs4su/code/

# remember to change TASK_ID
srun -l --ntasks-per-node=1 bash -c "docker run --net=host --ipc=host --rm --user $(id -u):$(id -g) --gpus all --shm-size=1024g --ulimit memlock=-1 --privileged \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e NCCL_SOCKET_IFNAME=ibp \
     -e NCCL_IB_HCA=mlx5 \
     -e NCCL_DEBUG_SUBSYS=ALL \
     -e GPUS_PER_NODE=8 \
     -e MASTER_ADDR=$(echo $head_node_ip) \
     -e MASTER_PORT=6000 \
     -e NODE_RANK=\$SLURM_PROCID \
     -e NNODES=4 \
     -e GRAD_ACC_STEPS=2 \
     -e TASK_ID=exp3.2 \
     -e CUDA_DEVICE_MAX_CONNECTIONS=10 \
     -e OMP_NUM_THREADS=10 \
     -e WANDB_API_KEY=${WANDB_API_KEY} \
     -e DATA_PATH=/workspace/megatron/slimpajama-icp-chunk1_pre_sorted-8node_text_document \
     -e DATA_CACHE_PATH=/workspace/megatron/slimpajama-icp-chunk1_pre_sorted-8node_text_document \
     -e TOKENIZER_MODEL_PATH=/workspace/megatron/baichuan.tokenizer.model \
     -e ENABLE_SHUFFLE=true \
     -v $(echo $PROJECT_ROOT)/Megatron-LM:/workspace/megatron \
     -v $(echo $PROJECT_ROOT)/dataset:/workspace/dataset \
     -v /run/mellanox/drivers:/run/mellanox/drivers:shared \
     -v /etc/network:/etc/network \
     -v /etc:/host/etc \
     -v /lib/udev:/host/lib/udev \
     -v $(echo $PROJECT_ROOT)/checkpoints:/workspace/checkpoints \
     -w /workspace/megatron \
     registry-intl.cn-hongkong.aliyuncs.com/sixpublic/pytorch:23.10-py3 \
     bash hkgai/launcher/scripts/pretrain/exp3/pretrain.sh"
