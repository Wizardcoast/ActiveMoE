export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# output_dir=output_moe
# if [ ! -d ${output_dir} ];then
#     mkdir -p ${output_dir}
# fi


### pt demo

### 多机多卡，仍然有bug

# deepspeed --hostfile hostfile --num_nodes 2  --num_gpus 8 --master_addr 172.20.39.22  --master_port=9901 src/train_bash.py \
#     --deepspeed ds_zero3.json\
#     --stage pt\
#     --finetuning_type full\
#     --do_train \
#     --template default\
#     --cutoff_len 4096\
#     --model_name_or_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/.cache/Llama-2-7b-hf\
#     --dataset_dir data\
#     --dataset wiki_demo \
#     --learning_rate 5e-5\
#     --overwrite_cache \
#     --num_train_epochs 1.0\
#     --ddp_find_unused_parameters False\
#     --plot_loss\
#     --overwrite_output_dir True\
#     --output_dir output_dense2moe\
#     --Dense2MoE True\
#     --MoE_config_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/configs/config.json\
#     --per_device_train_batch_size 1



# deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
#     --deepspeed ds_zero3.json\
#     --stage pt\
#     --finetuning_type full\
#     --do_train \
#     --template default\
#     --cutoff_len 4096\
#     --model_name_or_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/.cache/Llama-2-7b-hf\
#     --dataset_dir data\
#     --dataset wiki_demo \
#     --learning_rate 5e-5\
#     --overwrite_cache \
#     --num_train_epochs 1.0\
#     --ddp_find_unused_parameters False\
#     --plot_loss\
#     --overwrite_output_dir True\
#     --output_dir output_dense2moe\
#     --Dense2MoE True\
#     --MoE_config_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/configs/config.json\
#     --per_device_train_batch_size 1

# ### sft demo

# deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
#     --deepspeed ds_zero2.json\
#     --stage sft\
#     --finetuning_type full\
#     --do_train \
#     --template default\
#     --cutoff_len 4096\
#     --model_name_or_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/.cache/Llama-2-7b-hf\
#     --dataset_dir data\
#     --dataset alpaca_gpt4_en \
#     --learning_rate 5e-5\
#     --num_train_epochs 1.0\
#     --ddp_find_unused_parameters False\
#     --plot_loss\
#     --output_dir ${output_dir}\

