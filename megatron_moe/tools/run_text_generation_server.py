# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer,MegatronGenerate,MegatronGenerate_local
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
from megatron.text_generation import generate_and_post_process,generate
from megatron.text_generation import beam_search_and_post_process

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    config = core_transformer_config_from_args(get_args())

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    return parser


if __name__ == "__main__":
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        #server = MegatronServer(model)
        #server.run("0.0.0.0",port=args.port)
        server = MegatronGenerate_local(model)

    #{"prompts":["Hello world"], "tokens_to_generate":1}
    input = {}
    input['prompts']=["Hello world"]
    input['tokens_to_generate']=0
    

    x = generate_and_post_process(model,
                              prompts=input['prompts'],
                              tokens_to_generate=0,
                              return_output_log_probs=False,
                              top_k_sampling=0,
                              top_p_sampling=0.0,
                              top_p_decay=0.0,
                              top_p_bound=0.0,
                              temperature=1.0,
                              add_BOS=False,
                              use_eod_token_for_early_termination=True,
                              stop_on_double_eol=False,
                              stop_on_eol=False,
                              prevent_newline_after_colon=False,
                              random_seed=-1,
                              return_logits=False)
 

    print(x)


    '''
    while True:
        choice = torch.tensor([1], dtype=torch.long, device='cuda')
        torch.distributed.broadcast(choice, 0)
        if choice[0].item() == 0:
            try:
                generate_and_post_process(model)
            except ValueError as ve:
                pass
        elif choice[0].item() == 1:
            try:
                beam_search_and_post_process(model)
            except ValueError as ve:
                pass
    '''

