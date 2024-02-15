#!/bin/bash
#SBATCH -p megatron
#SBATCH -J Megatron-LM          # 队列名
#SBATCH --nodes 1  # 请求节点数
#SBATCH --cpus-per-task=64 # 任务所需cpu数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --exclusive
#SBATCH --exclude=dgx-052


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

srun -l --ntasks-per-node=1 bash -c "docker run --net=host --ipc=host --rm --user root --shm-size=1024g --ulimit memlock=-1 --privileged \
     -e NVIDIA_VISIBLE_DEVICES=all \
     -e NCCL_SOCKET_IFNAME=ibp \
     -e NCCL_IB_HCA=mlx5 \
     -e NCCL_DEBUG=INFO \
     -e NCCL_DEBUG_SUBSYS=ALL \
     -e GPUS_PER_NODE=8 \
     -e MASTER_ADDR=$(echo $head_node_ip) \
     -e MASTER_PORT=6000 \
     -e NODE_RANK=\$SLURM_PROCID \
     -e NNODES=4 \
     -e NCCL_DEBUG=INFO \
     -e CUDA_DEVICE_MAX_CONNECTIONS=10 \
     -e OMP_NUM_THREADS=10 \
     -e WANDB_API_KEY=${WANDB_API_KEY} \
     -v $(echo $PROJECT_ROOT)/Megatron-LM:/workspace/megatron \
     -v $(echo $PROJECT_ROOT)/dataset:/workspace/dataset \
     -v /aifs4su/data/rawdata/yubo_slimpajama/:/workspace/data/ \
     -v /run/mellanox/drivers:/run/mellanox/drivers:shared \
     -v /etc/network:/etc/network \
     -v /etc:/host/etc \
     -v /lib/udev:/host/lib/udev \
     -v $(echo $PROJECT_ROOT)/checkpoints:/workspace/checkpoints \
     -w /workspace/megatron \
     registry-intl.cn-hongkong.aliyuncs.com/sixpublic/pytorch:23.10-py3 \
     bash process_data.sh"