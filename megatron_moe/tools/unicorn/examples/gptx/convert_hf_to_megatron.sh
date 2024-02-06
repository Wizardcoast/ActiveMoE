set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}/tools/unicorn

python main.py \
  --megatron-path ${MEGATRON_PATH} \
  --load-path < path to gptx-pre-13b hf model > \
  --save-path gptx-megatron \
  --model-name gptx-13b \
  --template-name gptx \
  --print-checkpoint-structure \
  --target_tensor_model_parallel_size 2 \
  --target_pipeline_model_parallel_size 2 \
  --target_params_dtype bf16
