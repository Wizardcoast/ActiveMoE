import os
import sys
import types
import torch


def _guess_dtype(megatron_args, prefix=""):
    dtype = None
    if getattr(megatron_args, "bf16", None):
        dtype = "bfloat16"
    elif getattr(megatron_args, "fp16", None):
        dtype = "float16"
    else:
        dtype = "float32"
    return prefix + dtype


def _guess_megatron_args_dtype(torch_dtype):
    dtype_dict = {}
    if torch_dtype is torch.float16:
        dtype_dict["fp16"] = True
    elif torch_dtype == torch.bfloat16:
        dtype_dict["bf16"] = True
    return dtype_dict


def _dumps_GPTxConfig(megatron_args):
    from .transformers.models.gptx import GPTxConfig
    config = GPTxConfig(
        apply_residual_connection_post_layernorm=False,
        attention_dropout=megatron_args.attention_dropout,
        hidden_dropout=megatron_args.hidden_dropout,
        hidden_size=megatron_args.hidden_size,
        max_position_embeddings=megatron_args.seq_length,
        initializer_range=megatron_args.init_method_std,
        layer_norm_epsilon=megatron_args.norm_epsilon,
        model_type="GPTx",
        n_head=megatron_args.num_attention_heads,
        n_layer=megatron_args.num_layers,
        pretraining_tp=megatron_args.tensor_model_parallel_size,
        slow_but_exact=False,  # NOTE: disable exact mode
        use_cache=True,
        vocab_size=megatron_args.padded_vocab_size
    )

    return config


def _dumps_GPTxLegacyConfig(megatron_args):
    from .transformers.models.gptx import GPTxConfig
    config = GPTxConfig(
        apply_residual_connection_post_layernorm=False,
        attention_dropout=megatron_args.attention_dropout,
        hidden_dropout=megatron_args.hidden_dropout,
        hidden_size=megatron_args.hidden_size,
        max_position_embeddings=megatron_args.seq_length,
        initializer_range=megatron_args.init_method_std,
        layer_norm_epsilon=megatron_args.layernorm_epsilon,
        model_type="GPTx",
        n_head=megatron_args.num_attention_heads,
        n_layer=megatron_args.num_layers,
        pretraining_tp=megatron_args.tensor_model_parallel_size,
        slow_but_exact=False,  # NOTE: disable exact mode
        use_cache=True,
        vocab_size=megatron_args.padded_vocab_size
    )

    return config


def _dumps_LlamaConfig(megatron_args):
    from transformers import LlamaConfig
    config = LlamaConfig(
        architectures=[
            "LlamaForCausalLM"
        ],
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=megatron_args.hidden_size,
        initializer_range=megatron_args.init_method_std,
        intermediate_size=megatron_args.ffn_hidden_size,
        max_position_embeddings=megatron_args.seq_length,
        model_type="llama",
        num_attention_heads=megatron_args.num_attention_heads,
        num_hidden_layers=megatron_args.num_layers,
        num_key_value_heads=megatron_args.num_attention_heads,  # TODO: support GQA
        pretraining_tp=1,  # megatron_args.tensor_model_parallel_size,
        rms_norm_eps=megatron_args.norm_epsilon,
        rope_scaling=None,
        tie_word_embeddings=False if megatron_args.untie_embeddings_and_output_weights else True,
        torch_dtype=_guess_dtype(megatron_args),
        use_cache=True,
        vocab_size=megatron_args.padded_vocab_size
    )

    return config

def _dumps_Llama_MoeConfig(megatron_args):
    from .transformers.models.llama_moe import LlamaMoeConfig
    
    #load old model 
    try:
        moe_num_layers = megatron_args.moe_num_layers
    except:
        moe_num_layers = megatron_args.num_layers
        
    try:
        add_moe_share_expert = megatron_args.add_moe_share_expert
    except:
        add_moe_share_expert = False
        
    try:
        moe_share_expert_ffn_hidden_size = megatron_args.moe_share_expert_ffn_hidden_size
    except:
        moe_share_expert_ffn_hidden_size = megatron_args.ffn_hidden_size
        
    try:
        moe_expert_ffn_hidden_size = megatron_args.moe_expert_ffn_hidden_size
    except:
        moe_expert_ffn_hidden_size = megatron_args.ffn_hidden_size
        
    try:
        share_weight = megatron_args.share_weight
    except:
        share_weight = 1.0
        
    try:
        moe_weight = megatron_args.moe_weight
    except:
        moe_weight = 1.0
        
    try:
        add_moe_bias = megatron_args.add_moe_bias
    except:
        add_moe_bias = True
    
    config = LlamaMoeConfig(
        architectures=[
            "LlamaForCausalLM"
        ],
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=megatron_args.hidden_size,
        initializer_range=megatron_args.init_method_std,
        intermediate_size=megatron_args.ffn_hidden_size,
        max_position_embeddings=megatron_args.seq_length,
        model_type="llama_moe",
        num_attention_heads=megatron_args.num_attention_heads,
        num_hidden_layers=megatron_args.num_layers,
        num_key_value_heads=megatron_args.num_attention_heads,  # TODO: support GQA
        pretraining_tp=1,  # megatron_args.tensor_model_parallel_size,
        rms_norm_eps=megatron_args.norm_epsilon,
        rope_scaling=None,
        tie_word_embeddings=False if megatron_args.untie_embeddings_and_output_weights else True,
        torch_dtype=_guess_dtype(megatron_args),
        use_cache=True,
        vocab_size=megatron_args.padded_vocab_size,
        num_experts_per_tok=megatron_args.moe_top_k,
        num_local_experts=megatron_args.moe_num_experts,
        output_router_logits=False,
        router_aux_loss_coef=megatron_args.moe_loss_weight,
        moe_num_layers=moe_num_layers,
        add_moe_share_expert=add_moe_share_expert,
        moe_share_expert_intermediate_size=moe_share_expert_ffn_hidden_size,
        moe_expert_intermediate_size=moe_expert_ffn_hidden_size,
        share_weight=share_weight,
        moe_weight=moe_weight,
        add_moe_bias=add_moe_bias
    )

    return config


def _dumps_GPTNeoXConfig(megatron_args):
    from transformers import GPTNeoXConfig
    config = GPTNeoXConfig(
        architectures=[
            "GPTNeoXForCausalLM"
        ],
        model_type="gpt_neox",
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=megatron_args.hidden_size,
        intermediate_size=4*megatron_args.hidden_size,
        initializer_range=megatron_args.init_method_std,
        num_hidden_layers=megatron_args.num_layers,
        num_attention_heads=megatron_args.num_attention_heads,
        layer_norm_eps=megatron_args.norm_epsilon,
        max_position_embeddings=megatron_args.seq_length,
        hidden_act="gelu_fast",
        classifier_dropout=0.0,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        rotary_emb_base=10000,
        rotary_pct=1.0,
        use_parallel_residual=False,
        use_cache=True,
        vocab_size=megatron_args.padded_vocab_size,
        tie_word_embeddings=True,
    )

    return config


def _dumps_QWenConfig(megatron_args):
    from .transformers.models.qwen import QWenConfig
    config = QWenConfig(
        apply_residual_connection_post_layernorm=False,
        architectures=[
            "QWenLMHeadModel"
        ],
        auto_map={
            "AutoConfig": "configuration_qwen.QWenConfig",
            "AutoModelForCausalLM": "modeling_qwen.QWenLMHeadModel"
        },
        attn_pdrop=0.0, #megatron_args.attention_dropout,
        bf16=False,
        fp16=False,
        fp32=False,
        # bias_dropout_fusion
        embd_pdrop=0.0,
        ffn_hidden_size=megatron_args.ffn_hidden_size,
        initializer_range=megatron_args.init_method_std,
        kv_channels=megatron_args.kv_channels,
        layer_norm_epsilon=megatron_args.norm_epsilon,  # NOTE: compatible for new mlm
        model_type="qwen",
        n_embd=megatron_args.hidden_size,
        n_head=megatron_args.num_attention_heads,
        n_layer=megatron_args.num_layers,
        # n_positions
        no_bias=True,
        onnx_safe=None,
        padded_vocab_size=megatron_args.padded_vocab_size,
        params_dtype=_guess_dtype(megatron_args, "torch."),
        pos_emb="rotary",  # TODO: customization by --position-embedding-type
        rotary_emb_base=10000,
        rotary_pct=1.0,
        scale_attn_weights=True,
        seq_length=megatron_args.seq_length,
        tie_word_embeddings=False if megatron_args.untie_embeddings_and_output_weights else True,
        tokenizer_type="QWenTokenizer",
        use_cache=True,
        use_flash_attn="auto",
        vocab_size=megatron_args.padded_vocab_size,
        use_dynamic_ntk=True,
        use_logn_attn=True
    )

    return config


def get_config_from_megatron_args(model_type, megatron_args):
    model_type = model_type.lower()
    if model_type.startswith("gptx_legacy"):
        config = _dumps_GPTxLegacyConfig(megatron_args)
    elif model_type.startswith("gptx"):
        config = _dumps_GPTxConfig(megatron_args)
    elif model_type.startswith("llama_moe"):
        config = _dumps_Llama_MoeConfig(megatron_args)
    elif model_type.startswith("llama"):
        config = _dumps_LlamaConfig(megatron_args)
    elif model_type.startswith("qwen"):
        config = _dumps_QWenConfig(megatron_args)
    elif model_type.startswith("gpt_neox"):
        config = _dumps_GPTNeoXConfig(megatron_args)
    else:
        raise NotImplementedError(f"=> Configuration for {model_type} is not supported!")

    return config


def _loads_megatron_args_for_gptx(args, config_file):
    from .transformers.models.gptx import GPTxConfig
    config = GPTxConfig.from_pretrained(config_file)

    magatron_args = {
        "orig_vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "norm_epsilon": config.layer_norm_epsilon,
        "attention_dropout": config.attention_dropout,
        "hidden_dropout": config.hidden_dropout,
        "init_method_std": config.initializer_range,
        "seq_length": config.max_position_embeddings,  # TODO: patch to config.json
        "untie_embeddings_and_output_weights": not config.tie_word_embeddings,
        "bf16": True,  # TODO: use torch_dtype instead?
        "padded_vocab_size": config.vocab_size,
        "position_embedding_type": "rope",
        "normalization": "LayerNorm",
        "max_position_embeddings": config.max_position_embeddings,

        # training related
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "NullTokenizer",  # TODO: it is suitable
    }

    return magatron_args, config


def _loads_megatron_args_from_llama(args, config_file):
    from transformers import LlamaConfig
    config = LlamaConfig.from_pretrained(config_file)

    megatron_args = {
        "orig_vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.intermediate_size,
        "num_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "kv_channels": config.hidden_size // config.num_key_value_heads,
        "norm_epsilon": config.rms_norm_eps,
        "init_method_std": config.initializer_range,
        "seq_length": config.max_position_embeddings,
        "untie_embeddings_and_output_weights": not config.tie_word_embeddings,
        "padded_vocab_size": config.vocab_size,
        "position_embedding_type": "rope",
        "normalization": "RMSNorm",
        "swiglu": True,
        "max_position_embeddings": config.max_position_embeddings,
        "add_bias_linear": False,

        # training related
        "tensor_model_parallel_size": args.target_tensor_model_parallel_size,
        "pipeline_model_parallel_size": args.target_pipeline_model_parallel_size,
        "data_parallel_size": args.target_data_parallel_size,
        "make_vocab_size_divisible_by": args.make_vocab_size_divisible_by,
        "rank": 0,
        "tokenizer_type": "NullTokenizer",  # TODO: it is suitable
    }

    dtype_dict = _guess_megatron_args_dtype(config.torch_dtype)
    megatron_args.update(dtype_dict)
    return megatron_args, config


def _loads_megatron_args_from_qwen(args, config_file):
    from .transformers.models.qwen import QWenConfig
    config = QWenConfig.from_pretrained(config_file)

    # TODO: pay attention to bias item in query_key_value.weight
    #       it seems not implemented in MLM master branch.
    megatron_args = dict(
        orig_vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        ffn_hidden_size=config.ffn_hidden_size,
        num_layers=config.n_layer,
        num_attention_heads=config.n_head,
        kv_channels=config.kv_channels,
        norm_epsilon=config.layer_norm_epsilon,
        init_method_std=config.initializer_range,
        seq_length=config.seq_length,
        untie_embeddings_and_output_weights=not config.tie_word_embeddings,
        padded_vocab_size=config.vocab_size,
        position_embedding_type="rope",
        normalization="RMSNorm",
        swiglu=True,
        max_position_embeddings=config.seq_length,
        add_bias_linear=False,
        bias_attn_linear=True,

        # training related
        tensor_model_parallel_size=args.target_tensor_model_parallel_size,
        pipeline_model_parallel_size=args.target_pipeline_model_parallel_size,
        data_parallel_size=args.target_data_parallel_size,
        make_vocab_size_divisible_by=args.make_vocab_size_divisible_by,
        rank=0,
        tokenizer_type="NullTokenizer",  # TODO: it is suitable
    )

    dtype_dict = _guess_megatron_args_dtype(config.params_dtype)
    megatron_args.update(dtype_dict)
    return megatron_args, config


def get_megatron_args_from_config(model_type, args, config_file):
    model_type = model_type.lower()
    if model_type.startswith("gptx"):
        megatron_args, config = _loads_megatron_args_for_gptx(args, config_file)
    elif model_type.startswith("llama"):
        megatron_args, config = _loads_megatron_args_from_llama(args, config_file)
    elif model_type.startswith("qwen"):
        megatron_args, config = _loads_megatron_args_from_qwen(args, config_file)
    else:
        raise NotImplementedError(f"=> Configuration for {model_type} is not supported!")
    return megatron_args, config


def _save_config(config, save_path):
    print(f"=> Saving {type(config)} to {save_path} ...")
    config.save_pretrained(save_path)
