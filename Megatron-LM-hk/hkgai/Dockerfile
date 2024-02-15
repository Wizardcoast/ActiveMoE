FROM  nvcr.io/nvidia/pytorch:23.10-py3

RUN apt update && apt install infiniband-diags -y && \
    pip install sentencepiece wandb  && \
    pip install flash-attn
