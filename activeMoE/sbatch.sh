#!/bin/bash
#SBATCH --job-name=activemoe
#SBATCH --nodes=1           # Rquest 2 DGX
#SBATCH --ntasks-per-node=1 # Number of "tasks" per node
#SBATCH --gres=gpu:8        # Request 8 GPU
#SBATCH --cpus-per-task=8   # Request 8 CPU cores
#SBATCH --mem=1000G          # Request 128 GB of memory
#SBATCH --time=1:00:00      # Set the maximum runtime for your job
#SBATCH --exclusive
#SBATCH --output=%j.out
#SBATCH --error=%j.err

srun --partition=debug --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --job-name=activemoe  deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_zero3.json\
    --stage pt\
    --finetuning_type full\
    --do_train \
    --template default\
    --cutoff_len 4096\
    --model_name_or_path /home/songyanggao/models/phi-2 \
    --dataset_dir data \
    --dataset wiki_demo \
    --learning_rate 5e-5\
    --overwrite_cache \
    --num_train_epochs 1.0\
    --ddp_find_unused_parameters False\
    --plot_loss\
    --overwrite_output_dir True\
    --output_dir output_dense2moe\
    --Dense2MoE True\
    --model_name phi \
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/phi_config.json \
    --per_device_train_batch_size 1 \
    --do_sample True > logs/train_phi_test.log 2>&1 &
