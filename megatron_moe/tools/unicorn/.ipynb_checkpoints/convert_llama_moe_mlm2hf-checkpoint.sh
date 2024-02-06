#!/bin/bash
python main.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/megatron/Megatron-Inf \
  --load-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/chengs18/trainingInfo/checkpoint/dmoe_LLaMA_1.3Bx8_top2_3.3t/iter_0145000  \
  --save-path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/megablock/megablock_0_0/save_hf/moe_1b3_8_top2_600b \
  --model-name llama_moe \
  --template-name llama_moe \
  --print-checkpoint-structure \
  --target_params_dtype bf16