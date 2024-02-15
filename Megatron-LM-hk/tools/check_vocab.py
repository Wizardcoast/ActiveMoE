import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/megatron/Baichuan2-7B-Chat", trust_remote_code=True)
vocab_size = tokenizer.vocab_size

print("Vocabulary size:", vocab_size)