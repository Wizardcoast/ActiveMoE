# Dense2MoE training codebase

## demo

### pretraining 

注意之后需要修改 dataloader，现在只用了 demo 的小数据；目前测得在单机8卡上 `per_device_train_batch_size=1` 的情景下，可以跑到从 16 层开始每隔一层设置一层 8 个 experts 的 MoEs（总参数量 22B），不过过程中一直报 `high memory pressure` 的 warning，完整的 Mixtral 8*7B 的设置应该需要两台机器才能跑。

```bash
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_zero3.json\
    --stage pt\
    --finetuning_type full\
    --do_train \
    --template default\
    --cutoff_len 4096\
    --model_name_or_path /home/songyanggao/models/Llama-2-7b-chat-hf \
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
    --MoE_config_path /home/songyanggao/ActiveMoE/activeMoE/configs/config.json \
    --per_device_train_batch_size 1


```

### sft

```sh
deepspeed --num_gpus 8 --master_port=9901 src/train_bash.py \
    --deepspeed ds_zero2.json\
    --stage sft\
    --finetuning_type full\
    --do_train \
    --template default\
    --cutoff_len 4096\
    --model_name_or_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/.cache/Llama-2-7b-hf\
    --dataset_dir data\
    --dataset alpaca_gpt4_en \
    --learning_rate 5e-5\
    --num_train_epochs 1.0\
    --ddp_find_unused_parameters False\
    --plot_loss\
    --output_dir sft_demo\
    --Dense2MoE True\
    --MoE_config_path /cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/configs/config.json\
    --per_device_train_batch_size 1\
```

其它参数设置请参考原始 repo。
MoEs 相关的参数请在 `/configs/config.json` 中对应修改。


## 修改说明

### 模型相关

相关将 Llama2MoEs 的 class 位于： `/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/src/llmtuner/model/MoLlama`

主要的修改是载入模型的过程中

`/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/src/llmtuner/model/loader.py`

```python
if model is None:
    if model_args.Dense2MoE:
        from llmtuner.model.MoLlama import MoLlamaForCausalLM, MoLlamaConfig
        config = MoLlamaConfig.from_pretrained(model_args.MoE_config_path)
        model = MoLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )
        model.model.reinit_mlp()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
            **config_kwargs
        )
```

支持了添加 MoLlama 类，对应的设置在 configs 里面

### configs 说明

```json
  "num_experts": 8, //每一层复制的 experts 数
  "expert_top_k": 2, //每一层选取的 topk
  "moe_start_layer": 16,  //从多少层开始转成 MoEs
  "moe_layer_freq":2,  //每隔多少层转 MoEs
  "gate_type": "linear", // gate 类型，支持 "linear". "mlp", 'gmm'  
  "aux_loss_type": "mi", // 补充 loss 的类型，支持 'mi' 和 'switch'
  "aux_loss_weight": 1e-2 // switch 推荐的 weight
```

### 训练相关

目前 gate loss 直接添加在了 forward 过程中，避免需要在训练 loop 中手动提取、添加 gate loss 导致需要重头训 gate



## TODOs:

### bug-check

一个不确定的点：`/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/LLaMA-Factory/src/llmtuner/model/MoLlama/modeling_MoLlama_hf.py` 中 860 行的位置，
transformers 中 from_pretrained 功能在使用 deepspeed zero3 的时候会对 weight 做拆分处理，而后如果直接 copy weight 初始化 MoEs 会出错，因此首先使用 
`deepspeed.zero.GatheredParameters` 将相关参数整合后再初始化 experts；不太确定这样做是否会造成参数冗余（也即复制添加的 experts 不能正确进行模型并行）

### Deepspeed MoEs 支持

确定 Deepspeed MoEs 中 gate 相关 configs 后优化

### 其它 MoEs 的 feature

Residual Mixture of Experts: 每一层设置一个 share experts + top1 选择 experts。Motivation 为将 experts 的输出视作 share experts 的修正

PMoE：不同层的 experts 不同，底层少顶层高（这与提供的 `moe_start_layer, moe_layer_freq` configs 相关）