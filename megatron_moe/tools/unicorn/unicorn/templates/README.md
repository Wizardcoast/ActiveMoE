##########
# KeyMap
##########
Qwen:
transformer.wte.weight,

transformer.h.0.ln_1.weight,
transformer.h.0.attn.c_attn.weight,
transformer.h.0.attn.c_attn.bias,
transformer.h.0.attn.c_proj.weight,
transformer.h.0.ln_2.weight,
transformer.h.0.mlp.w1.weight,
transformer.h.0.mlp.w2.weight,
transformer.h.0.mlp.c_proj.weight,

transformer.ln_f.weight,

lm_head.weight

-----------------------------------------
LLaMA2:
model.embed_tokens.weight

model.layers.0.input_layernorm.weight,
model.layers.0.mlp.down_proj.weight,
model.layers.0.mlp.gate_proj.weight,
model.layers.0.mlp.up_proj.weight,
model.layers.0.post_attention_layernorm.weight,
model.layers.0.self_attn.k_proj.weight,
model.layers.0.self_attn.o_proj.weight,
model.layers.0.self_attn.q_proj.weight,
model.layers.0.self_attn.rotary_emb.inv_freq,
model.layers.0.self_attn.v_proj.weight

model.norm.weight

lm_head.weight

-----------------------------------------
GPTx:
word_embeddings.weight

h.0.input_layernorm.weight,
h.0.input_layernorm.bias,
h.0.self_attention.query_key_value.weight,
h.0.self_attention.query_key_value.bias,
h.0.self_attention.dense.weight,
h.0.self_attention.dense.bias,
h.0.self_attention.rotary_emb.inv_freq,
h.0.post_attention_layernorm.weight,
h.0.post_attention_layernorm.bias,
h.0.mlp.dense_h_to_4h.weight,
h.0.mlp.dense_h_to_4h.bias,
h.0.mlp.dense_4h_to_h.weight,
h.0.mlp.dense_4h_to_h.bias,

ln_f.weight,
ln_f.bias

#################
# Configuration
#################
GPTx
{
  "apply_residual_connection_post_layernorm": false,
  "attention_dropout": 0.0,
  "hidden_dropout": 0.0,
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "layer_norm_epsilon": 1e-05,
  "model_type": "GPTx",
  "n_head": 40,
  "n_layer": 40,
  "pretraining_tp": 4,
  "slow_but_exact": false,
  "transformers_version": "4.27.2",
  "use_cache": true,
  "vocab_size": 81920
}

LLaMA
{
  "_name_or_path": "meta-llama/Llama-2-13b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 13824,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 40,
  "num_hidden_layers": 40,
  "num_key_value_heads": 40,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.32.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}

Qwen
{
  "apply_residual_connection_post_layernorm": false,
  "architectures": [
    "QWenLMHeadModel"
  ],
  "auto_map": {
    "AutoConfig": "configuration_qwen.QWenConfig",
    "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
  },
  "attn_pdrop": 0.0,
  "bf16": false,
  "fp16": false,
  "fp32": false,
  "bias_dropout_fusion": true,
  "bos_token_id": 151643,
  "embd_pdrop": 0.0,
  "eos_token_id": 151643,
  "ffn_hidden_size": 22016,
  "initializer_range": 0.02,
  "kv_channels": 128,
  "layer_norm_epsilon": 1e-06,
  "model_type": "qwen",
  "n_embd": 4096,
  "n_head": 32,
  "n_layer": 32,
  "n_positions": 8192,
  "no_bias": true,
  "onnx_safe": null,
  "padded_vocab_size": 151936,
  "params_dtype": "torch.bfloat16",
  "pos_emb": "rotary",
  "resid_pdrop": 0.1,
  "rotary_emb_base": 10000,
  "rotary_pct": 1.0,
  "scale_attn_weights": true,
  "seq_length": 2048,
  "tie_word_embeddings": false,
  "tokenizer_type": "QWenTokenizer",
  "transformers_version": "4.31.0",
  "use_cache": true,
  "use_flash_attn": "auto",
  "vocab_size": 151936,
  "use_dynamic_ntk": true,
  "use_logn_attn": true
}


