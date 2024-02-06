import torch
import transformers
import sys
import os

current_path = os.getcwd()
sys.path.append(current_path)
sys.path.append('/cpfs01/projects-HDD/cfff-23ba4487e9df_HDD/wangzili/moe_yuan/transformers/models/llama_moe/')

from llama_moe_modeling import LlamaModel