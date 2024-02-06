#### test_convert_ckpt.py

1. Use Megatron-LM checkpoints or unicorn converted version of Huggingface model. Below is the second option:
```shell
set -x

MEGATRON_PATH=< path to Megatron-LM >
cd ${MEGATRON_PATH}/tools/unicorn

python main.py \
  --megatron-path ${MEGATRON_PATH} \
  --load-path < path to gptx huggingface model > \
  --save-path gptx-megatron \
  --model-name gptx-13b \
  --template-name gptx_legacy \
  --print-checkpoint-structure \
  --target_tensor_model_parallel_size 2 \
  --target_pipeline_model_parallel_size 2 \
  --target_params_dtype bf16 \
  --set-iteration 1 
```

2. Convert Megatron checkpoint to TP=1 & PP=1 using `tools/checkpoint`
```shell
MEGATRON_PATH=< path to Megatron-LM >
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}

cd ${MEGATRON_PATH}/tests/unit_tests/

python test_convert_ckpt.py \
  --model-type GPT \
  --megatron-checkpoint ./conveted \
  --huggingface-model < path to gptx huggingface model > \
  --pre-convert-megatron \
  --pre-megatron-checkpoint ${MEGATRON_PATH}/tools/unicorn/gptx-megatron
```

3. compare model weights and forwarding precision
* You can use the precision aligned huggingface version of GPTx in `tools/checkpoint/transformers/models/gptx` 
```shell
MEGATRON_PATH=< path to Megatron-LM >
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}
cd ${MEGATRON_PATH}/tests/unit_tests/

python test_convert_ckpt.py \
  --model-type GPT \
  --megatron-checkpoint ./conveted \
  --huggingface-model < path to gptx huggingface model or precision aligned one > \
  --compare-forward \
  --compare-weight
```
