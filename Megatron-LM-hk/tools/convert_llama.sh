MEGATRON_SAVE_ROOT=/workspace/checkpoints
HF_SAVE_ROOT=/workspace/megatron/hf_ckpt

ITER=${1:-490000}
HF_SAVE_PATH=${2:-null}
MEGATRON_PATH=${3:-${MEGATRON_SAVE_ROOT}/hkg_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0${ITER}}

HF_SAVE_PATH=${HF_SAVE_PATH}/hf_${ITER}

echo "ITER: ${ITER}"
echo "MEGATRON_PATH: ${MEGATRON_PATH}"
echo "HF_SAVE_PATH: ${HF_SAVE_PATH}"

mkdir -p ${HF_SAVE_PATH}

python convert_megatron_core_llama2hf.py \
    --input-dir ${MEGATRON_PATH} \
    --output-dir ${HF_SAVE_PATH} \
    --vocab-size 125699 
