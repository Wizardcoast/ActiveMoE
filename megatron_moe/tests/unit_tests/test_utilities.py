import os
import torch
import megatron.core.parallel_state as ps

class Utils:
    world_size = torch.cuda.device_count()
    rank = int(os.getenv('LOCAL_RANK', 0))

    @staticmethod
    def initialize_distributed():
        print(f'Initializing torch.distributed with rank: {Utils.rank}, world_size: {Utils.world_size}')
        torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
        init_method = 'tcp://'
        master_ip = os.getenv('MASTER_ADDR', 'localhost')
        master_port = os.getenv('MASTER_PORT', '6000')
        init_method += master_ip + ':' + master_port
        torch.distributed.init_process_group(backend='nccl', world_size=Utils.world_size, rank=Utils.rank,
                                             init_method=init_method)

    @staticmethod
    def destroy_model_parallel():
        ps.destroy_model_parallel()
        torch.distributed.barrier()

    @staticmethod
    def initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
                                  virtual_pipeline_model_parallel_size=None, pipeline_model_parallel_split_rank=None):
        ps.destroy_model_parallel()
        if not torch.distributed.is_initialized():
            Utils.initialize_distributed()
        ps.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size,
                                     virtual_pipeline_model_parallel_size, pipeline_model_parallel_split_rank)

    @staticmethod
    def _clean_global_vars():
        import megatron.global_vars
        from megatron.global_vars import set_args

        set_args(None)
        megatron.global_vars._GLOBAL_NUM_MICROBATCHES_CALCULATOR = None
        megatron.global_vars._GLOBAL_TOKENIZER = None
        megatron.global_vars._GLOBAL_TIMERS = None

    @staticmethod
    def gpt_load_checkpoint(margs, load_scheduler=False):
        # mock args
        import sys

        from megatron import get_args
        from megatron import print_rank_0
        from megatron.model.gpt_model import GPTModel
        from megatron.core.enums import ModelType
        from megatron.core import mpu
        from megatron.checkpointing import load_args_from_checkpoint, load_checkpoint
        from megatron.arguments import parse_args, validate_args
        from megatron.global_vars import set_args, set_global_variables
        from megatron.arguments import core_transformer_config_from_args

        def model_provider(pre_process=True, post_process=True):
            """Build the model."""
            args = get_args()
            print_rank_0('building GPT model ...')
            config = core_transformer_config_from_args(get_args())

            model = GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
            return model

        # NOTE: load_args_from_checkpoint -> _load_base_checkpoint ->
        #       find_checkpoint_rank_0 -> get_checkpoint_name ->
        #       get_args(), so _GLOBAL_ARGS must be initialized.
        # A temporary workaround
        set_args(margs)
        margs, checkpoint_args = load_args_from_checkpoint(margs)

        # NOTE: set_global_variables need to _ensure_var_is_not_initialized,
        #       so set _GLOBAL_ARGS back to None.
        set_args(None)
        margs.model_type = ModelType.encoder_or_decoder

        # mock mpu
        margs.world_size = margs.tensor_model_parallel_size * \
                           margs.pipeline_model_parallel_size
        mpu.set_tensor_model_parallel_world_size(
            margs.tensor_model_parallel_size)
        mpu.set_pipeline_model_parallel_world_size(
            margs.pipeline_model_parallel_size)
        mpu.set_virtual_pipeline_model_parallel_world_size(1)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
        mpu.set_pipeline_model_parallel_rank(0)
        mpu.set_tensor_model_parallel_rank(0)

        #
        validate_args(margs)

        # params_dtype
        args_to_keep = ['lr', 'min_lr', 'lr_decay_iters', 'lr_decay_style',
                        'start_weight_decay', 'end_weight_decay', 'weight_decay_incr_style',
                        'use_checkpoint_opt_param_scheduler',
                        'override_opt_param_scheduler', 'lr_warmup_iters', 'weight_decay_incr_style']
        for arg, value in vars(checkpoint_args).items():
            if arg not in args_to_keep:
                continue
            if not hasattr(margs, arg):
                print(
                    f"Checkpoint had argument {arg} but new arguments does not have this.")
                continue
            if getattr(margs, arg) != value:
                print(
                    f"Overwriting default {arg} value {getattr(margs, arg)} with value from checkpoint {value}.")
                setattr(margs, arg, value)
        # fused_kernels.load(margs)

        set_global_variables(margs)
        set_args(margs)

        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        this_model = model_provider(
            pre_process=pre_process,
            post_process=post_process
        )

        from megatron.model import Float16Module

        """
        in most cases, megatron use param dtype for layers
        but for some fused layer such as layernorm, it needs to be converted after initialization
        """
        if margs.bf16 or margs.fp16:
            this_model = Float16Module(this_model, margs)

        scheduler = None

        if load_scheduler:
            from megatron.training import get_optimizer_param_scheduler

            # mock optimizer. currently not testing load optimizer states.only the scheduler
            class OPT:
                param_groups = [{'lr': 0, 'weight_decay': 0, 'lr_mult': 0, 'wd_mult': 0}]

            optimizer = OPT()
            scheduler = get_optimizer_param_scheduler(optimizer)

        # this is required before load
        margs.consumed_train_samples = 0
        margs.consumed_valid_samples = 0
        iteration = load_checkpoint([this_model], None, scheduler)

        return this_model, iteration, scheduler
