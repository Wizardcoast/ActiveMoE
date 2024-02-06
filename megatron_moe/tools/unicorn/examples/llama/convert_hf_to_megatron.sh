set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}/tools/unicorn

python main.py \
  --megatron-path ${MEGATRON_PATH} \
  --load-path < path to llama2 hf model > \
  --save-path llama-megatron \
  --model-name llama2-13b \
  --template-name llama \
  --print-checkpoint-structure \
  --target_tensor_model_parallel_size 2 \
  --target_pipeline_model_parallel_size 2 \
  --target_params_dtype fp16
