Aligned the computation precision of forwarding between TP=1 & PP=1 Megatron-LM and HuggingFace transformers.
BUG: MixedFusedLayerNorm (borrowed from apex) cannot work properly with transformers on multiple-GPU devices,
     you must use `CUDA_VISIBLE_DEVICES=0 python ...` to force the code running only on one device.