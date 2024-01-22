from typing import TYPE_CHECKING, Optional, Tuple
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version
from trl import AutoModelForCausalLMWithValueHead

from src.llmtuner.extras.logging import get_logger
from src.llmtuner.extras.misc import count_parameters, get_current_device, try_download_model_from_ms
from src.llmtuner.model.adapter import init_adapter
from src.llmtuner.model.patcher import patch_config, patch_tokenizer, patch_model, patch_valuehead_model
from src.llmtuner.model.utils import load_valuehead_params, register_autoclass


MoE_config_path = ''
model_name_or_path = ''

from llmtuner.model.MoLlama import MoLlamaForCausalLM, MoLlamaConfig
config = MoLlamaConfig.from_pretrained(MoE_config_path)
model = MoLlamaForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
    **config_kwargs
)
model.model.reinit_mlp()