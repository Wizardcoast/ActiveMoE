set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}/tests/unicorn

python compare_hf_model.py \
  --src-model < path to gptx-pre-13b hf model > \
  --dst-model < path to converted hf model from megatron >
