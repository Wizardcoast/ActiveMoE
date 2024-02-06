set -x

python util.py \
  --model-type GPT \
  --load-dir <path of megatron checkpoint to convert> \
  --save-dir <path of converted megatron checkpoint> \
  --megatron-path <root directory of Megatron-LM> \
  --target-tensor-parallel-size <target TP size> \
  --target-pipeline-parallel-size <target PP size> \
  --bf16
