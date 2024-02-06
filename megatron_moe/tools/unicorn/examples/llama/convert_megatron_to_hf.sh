set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}/tools/unicorn

python main.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path ${MEGATRON_PATH} \
  --load-path llama-megatron/release/ \
  --save-path llama-hf \
  --model-name llama \
  --template-name llama \
  --print-checkpoint-structure \
  --target_params_dtype fp16
