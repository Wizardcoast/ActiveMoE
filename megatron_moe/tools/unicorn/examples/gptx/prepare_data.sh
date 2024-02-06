set -x

MEGATRON_PATH=< path to megatron >
cd ${MEGATRON_PATH}

PYTHONPATH=${MEGATRON_PATH} python tools/preprocess_data.py \
  --input tests/unicorn/data/sample.jsonl \
  --json-keys text \
  --tokenizer-type PretrainedFromHF \
  --tokenizer-name-or-path < path to tokenizer_v2 directory > \
  --append-eod \
  --output-prefix tests/unicorn/data/sample_gptx \
  --workers 4
