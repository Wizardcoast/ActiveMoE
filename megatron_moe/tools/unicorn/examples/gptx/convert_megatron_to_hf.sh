set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}/tools/unicorn

python main.py \
  --convert_checkpoint_from_megatron_to_transformers \
  --megatron-path ${MEGATRON_PATH} \
  --load-path gptx-megatron/release/ \
  --save-path gptx-hf \
  --model-name gptx \
  --template-name gptx \
  --print-checkpoint-structure \
  --target_params_dtype bf16
