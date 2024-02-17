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


srun --partition=ocr --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --job-name=activemoe deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py  \
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
    --flash_attn \
    --overwrite_output_dir True  \
    --output_dir output_dense2moe/216train_phisft_orca_lr5e5  \
    --evaluation_strategy steps --eval_steps 5000  \
    --save_strategy steps --save_steps 5000 --save_total_limit 6   \
    --Dense2MoE True  --model_name phi   \
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/phi2_nomoe_config.json\
    --per_device_train_batch_size 8    \
    --save_safetensors False    \
    --streaming True --dispatch_batches False \
    --max_step 30000    \
    --do_sample True  > logs/train_phisft_orca.log 2>&1 &


# train phi moe
srun --partition=ocr --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --job-name=activemoe deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py  \
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
    --flash_attn \
    --bf16 \
    --overwrite_output_dir True  \
    --output_dir output_dense2moe/216train_phi2moepreserve_orca_lr5e5_run2  \
    --evaluation_strategy no  \
    --save_strategy steps --save_steps 10000 --save_total_limit 6   \
    --Dense2MoE True  --model_name phi   \
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/phi2_config.json\
    --per_device_train_batch_size 2    \
    --save_safetensors False    \
    --streaming True --dispatch_batches False \
    --max_step 60000    \
    --do_sample True  > logs/train_phimoe_orca_preserve_run2.log 2>&1 &

#train orca sft
srun --partition=ocr --nodes=1 --gres=gpu:8 --ntasks-per-node=1 --job-name=activemoe deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py  \
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
    --flash_attn \
    --bf16 \
    --overwrite_output_dir True  \
    --output_dir output_dense2moe/216train_phisft_orca_lr5e5_run2  \
    --evaluation_strategy no --eval_steps 10000  \
    --save_strategy steps --save_steps 8000 --save_total_limit 6   \
    --Dense2MoE False  --model_name phi   \
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/phi2_config.json\
    --per_device_train_batch_size 4    \
    --save_safetensors False    \
    --streaming True --dispatch_batches False \
    --max_step 40000    \
    --do_sample True  > logs/train_phisft_orca_lr5e5_run2.log 2>&1 &