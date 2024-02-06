### Unicorn

#### Introduction
A tool to convert checkpoints between Megatron-LM and Huggingface transformers.
* Still in developing, currently supported network architectures
  * GPTx
  * LLaMA
  * Qwen
* Check `examples` for more details.

#### Usage
* Convert Megatron-LM checkpoint to Huggingface model.
```shell
python main.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path < MEGATRON_PATH > \
  --load-path < path to megatron checkpoint >  \
  --save-path < path to save converted hf model > \
  --model-name < gptx|llama|qwen > \
  --template-name < gptx|llama|qwen > \
  --print-checkpoint-structure \
  --target_params_dtype < bf16|fp16 >
```

* Make Megatron-LM checkpoint from Huggingface model.
```shell
python main.py \
  --megatron-path < MEGATRON_PATH > \
  --load-path < path to hf model > \
  --save-path < output dirctory to save converted megatron checkpoint > \
  --model-name < gptx|llama|qwen > \
  --template-name < gptx|llama|qwen > \
  --print-checkpoint-structure \
  --target_tensor_model_parallel_size < target TP size > \
  --target_pipeline_model_parallel_size < target PP size > \
  --target_params_dtype < bf16|fp16 >
```

#### Notebook
You can check notebooks in `tools/unicorn/examples/llama/` for more details.
