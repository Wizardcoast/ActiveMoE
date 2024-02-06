# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import math

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, List
from logging import getLogger

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron import get_args
from megatron.core.utils import get_attr_wrapped_model, get_model_config

from . import parallel_state
from .transformer.module import MegatronModule
from .transformer.transformer_config import TransformerConfig


logger = getLogger(__name__)


def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into dp_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)] for r in range(data_parallel_world_size)
    ]
    return sharded_buffer

class Bucket:
    """
    Bucket to keep track of a subset of the model's gradients. Provides functionality to register
    when params in the bucket have grads ready to be synced; an asynchronous communication call
    is automatically launched when _all_ params in the bucket have grads ready.
    Arguments:
        params: List of parameters whose gradients are collated in this bucket.
        data: View in larger GradBuffer that this bucket is responsible for.
        offset: Offset of this bucket's view in the larger GradBuffer.
        true_numel: The numel of data before pad to divisible by DP group size.
        data_parallel_group: Data-parallel process group.
        data_parallel_world_size: World size using the data-parallel group group.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        data: torch.Tensor,
        offset: int,
        true_numel: int,
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_world_size: int,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):
        # State for bookkeeping: params is the set of parameters this bucket is
        # responsible for, params_with_grad is the set of parameters with grads
        # available. When overlap_grad_reduce is True, communication (all-reduce
        # or reduce-scatter) is issued when params_with_grad equals params.
        self.params_list = params
        self.params = set(params)
        self.params_with_grad = set()
        self.data = data
        # The distributed optimizer needs to keep track of this bucket's offset
        # within the full grad_buffer.
        self.offset = offset
        self.true_numel = true_numel
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = data_parallel_world_size
        self.data_parallel_rank = torch.distributed.get_rank(group=data_parallel_group)
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        self.reset()

    def reset(self):
        """
        Reset metadata in bucket in preparation for the next iteration of training.
        """
        self.params_with_grad = set()
        self.communication_handle = None
        self.communication_issued = False

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.
        When overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.communication_handle is None and not self.communication_issued
        ), 'Should not have multiple communication calls in flight at once'

        self.data /= self.data_parallel_world_size
        # Use async_op only when overlap_grad_reduce is True.
        if self.use_distributed_optimizer:
            args = get_args()
            # TODO: check the correctness of expert parallelism.
            if args.pipeline_model_parallel_size > 1 and \
                (not args.untie_embeddings_and_output_weights or args.pipeline_model_parallel_split_rank is not None):
                self.communication_handle = torch.distributed.all_reduce(
                    self.data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
                )
            else:
                local_data_view = shard_buffer(self.data, self.data_parallel_world_size)[self.data_parallel_rank]
                self.communication_handle = torch.distributed._reduce_scatter_base(
                    local_data_view,
                    self.data,
                    group=self.data_parallel_group,
                    async_op=self.overlap_grad_reduce,
                )
        else:
            self.communication_handle = torch.distributed.all_reduce(
                self.data, group=self.data_parallel_group, async_op=self.overlap_grad_reduce
            )
        self.communication_issued = True

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operation
        for this bucket.
        When overlap_grad_reduce is set to True, waits for asynchronous communication
        call to complete. When overlap_grad_reduce is set to False, makes synchronous call.
        """
        # If overlap_grad_reduce is False, start (and finish) synchronous communication call here.
        if not self.overlap_grad_reduce:
            self.start_grad_sync()
            return
        assert self.communication_handle is not None and self.communication_issued, (
            f'Communication call has not been issued for this bucket '
            f'({len(self.params_with_grad)}/{len(self.params)} params have grad available)'
        )
        self.communication_handle.wait()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.
        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert param in self.params, 'Param is not in the bucket'
        assert param not in self.params_with_grad, 'Cannot set grad twice'
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should be called only when overlapping grad reduce'
        self.params_with_grad.add(param)
        # If all params in bucket have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()


class GradBuffer:
    """
    Groups gradients into a contiguous buffer, and then breaks the buffer into buckets with
    roughly `bucket_size` parameters each.

    Arguments:
        dtype: Type of underlying tensor.
        params: List of parameters whose gradients are collated in the underlying tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
    ):

        # Check that params are unique.
        unique_params = set()
        for param in params:
            assert param not in unique_params
            unique_params.add(param)
        del unique_params

        # Store attributes that will be needed later.
        self.dtype = dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = torch.distributed.get_world_size(
            group=self.data_parallel_group
        )
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer
        self.is_last_microbatch = True

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad_if_needed(data_index: int):
            """Pads data indices if using distributed optimizer (to ensure uniform sharding)."""
            if use_distributed_optimizer:
                return (
                    int(math.ceil(data_index / self.data_parallel_world_size))
                    * self.data_parallel_world_size
                )
            return data_index

        # First, figure out how many elements should be in the underlying buffer storage.
        # Note that if we need to split the buffer into smaller buckets, each of these
        # might need to be padded as well (if using the distributed optimizer).
        data_start_index = 0
        bucket_data_start_index = data_start_index
        bucket_params = set()
        self.bucket_indices = []
        bucket_id = 0
        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order,
            # and skip parameters that don't require gradients.
            if not param.requires_grad:
                continue
            this_numel = param.data.nelement()
            data_end_index = data_start_index + this_numel
            self.param_index_map[param] = (
                data_start_index,
                data_end_index,
                bucket_id,
            )
            bucket_params.add(param)

            # If we have enough elements already, form a new bucket.
            # If bucket_size is None, accumulate everything into a single bucket.
            if bucket_size is not None:
                if (data_end_index - bucket_data_start_index) >= bucket_size:
                    true_numel = data_end_index - bucket_data_start_index
                    data_end_index = _pad_if_needed(data_end_index)
                    self.bucket_indices.append((bucket_data_start_index, data_end_index, true_numel))
                    bucket_data_start_index = data_end_index
                    bucket_params = set()
                    bucket_id += 1
            data_start_index = data_end_index

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            true_numel = data_end_index - bucket_data_start_index
            data_end_index = _pad_if_needed(data_end_index)
            self.bucket_indices.append((bucket_data_start_index, data_end_index, true_numel))

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = data_end_index
        if use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
        self.data = torch.zeros(
            self.numel, dtype=self.dtype, device=torch.cuda.current_device(), requires_grad=False,
        )

        # Finally, map main_grad fields for each parameter with a .grad field.
        bucket_params = set()
        bucket_data_start_index = 0
        cur_bucket_id = 0
        for param in params[::-1]:
            if not param.requires_grad:
                continue
            data_start_index, data_end_index, bucket_id = self.param_index_map[param]
            param.main_grad = self._get(param.data.shape, data_start_index)
            if bucket_id != cur_bucket_id:
                bucket_data_end_index = _pad_if_needed(data_start_index)
                self._set_bucket(
                    bucket_params, bucket_data_start_index, bucket_data_end_index, cur_bucket_id
                )
                bucket_data_start_index = bucket_data_end_index
                bucket_params = set()
                assert cur_bucket_id + 1 == len(self.buckets)
                assert bucket_id == cur_bucket_id + 1
                cur_bucket_id = bucket_id
            bucket_params.add(param)

        # Add remaining params to a new bucket.
        if len(bucket_params) > 0:
            bucket_data_end_index = _pad_if_needed(data_end_index)
            self._set_bucket(
                bucket_params, bucket_data_start_index, bucket_data_end_index, cur_bucket_id
            )

        if not overlap_grad_reduce:
            assert len(bucket_params) == len(
                params
            ), 'All params should be in one bucket when overlap_grad_reduce is False'

        # Print buckets for all PP stages.
        if (
            parallel_state.get_data_parallel_rank() == 0
            and parallel_state.get_tensor_model_parallel_rank() == 0
        ):
            logger.info(
                f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
            )
            for index, bucket in enumerate(self.buckets):
                numel = 0
                for param in bucket.params:
                    numel += param.data.nelement()
                logger.info(f'Params for bucket {index+1} ({numel} elements):')
                for param in bucket.params:
                    logger.info(f'    {param_to_name[param]}'+'    '+str(param.size()))

    def _get(self, shape: torch.Size, start_index: int) -> torch.Tensor:
        """
        Return a tensor with the input `shape` as a view into the 1-D data starting at
        `start_index`.
        """
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, 'Requested tensor is out of buffer range'
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor

    def _set_bucket(
        self,
        bucket_params: List[torch.nn.Parameter],
        start_index: int,
        end_index: int,
        bucket_id: int,
    ):
        """
        Helper function to create new bucket, add it to list of buckets, and
        also update param->bucket mapping.
        """

        # Assert that indices are correctly padded (if needed), and that bucket
        # position is same as originally computed.
        if self.use_distributed_optimizer:
            assert start_index % self.data_parallel_world_size == 0
            assert end_index % self.data_parallel_world_size == 0
        true_numel = self.bucket_indices[bucket_id][-1]
        assert (start_index, end_index, true_numel) == self.bucket_indices[bucket_id]

        # Get appropriate view into global GradBuffer.
        bucket_data = self._get(torch.Size([end_index - start_index]), start_index)
        bucket = Bucket(
            params=bucket_params,
            data=bucket_data,
            offset=start_index,
            true_numel=true_numel,
            data_parallel_group=self.data_parallel_group,
            data_parallel_world_size=self.data_parallel_world_size,
            overlap_grad_reduce=self.overlap_grad_reduce,
            use_distributed_optimizer=self.use_distributed_optimizer,
        )
        self.buckets.append(bucket)
        for bucket_param in bucket_params:
            assert bucket_param not in self.param_to_bucket
            self.param_to_bucket[bucket_param] = bucket

    def reset(self):
        """
        Zero out the underlying buffer and reset all buckets in preparation for the next
        iteration of training.
        """
        self.data.zero_()
        for bucket in self.buckets:
            bucket.reset()
        self.is_last_microbatch = True

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.
        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the grad buffer.
        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket in self.buckets:
            bucket.finish_grad_sync()

    def register_grad_ready(self, param: torch.nn.Parameter):
        """
        Registers grads for the passed-in param to be "ready" for grad sync.
        When the number of microbatches is greater than 1, we only want to register
        grads as ready when processing the last microbatch and overlap_grad_reduce is True.
        """
        assert (
            self.overlap_grad_reduce
        ), 'register_grad_ready() should only be called when overlap_grad_reduce is True'
        if self.is_last_microbatch:
            bucket = self.param_to_bucket[param]
            bucket.register_grad_ready(param)


class DistributedDataParallel(MegatronModule):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).
    Arguments:
        config: Transformer config object.
        module: Underlying model.
        data_parallel_group: Data-parallel process group.
        accumulate_allreduce_grads_in_fp32: If true, do the gradient accumulation and
            communication in fp32.
        overlap_grad_reduce: If true, overlap communication with backprop computation by
            breaking up grads into buckets. If false, single synchronous communication call
            is used instead.
        use_distributed_optimizer: If true, issue reduce-scatter communication calls as part
            of distributed optimizer. If false, issue all-reduce communication calls.
    """

    def __init__(
        self,
        config: TransformerConfig,
        module: torch.nn.Module,
        data_parallel_group: torch.distributed.ProcessGroup,
        accumulate_allreduce_grads_in_fp32: bool,
        overlap_grad_reduce: bool,
        use_distributed_optimizer: bool,
        bucket_size: int = None,
    ):
        super().__init__(config=config)
        self.module = module

        # Set bucket_size to infinity if overlap_grad_reduce is False.
        self.overlap_grad_reduce = overlap_grad_reduce
        self.use_distributed_optimizer = use_distributed_optimizer

        # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
        # that is not the first (since data-parallel communication on these stages is not on
        # the critical path), or we might not want to break up model parameters into buckets
        # for model chunks after the first in the interleaved schedule).
        if not self.overlap_grad_reduce:
            bucket_size = None
        if parallel_state.get_pipeline_model_parallel_rank() > 0:
            bucket_size = None
        self.bucket_size = bucket_size

        self.module = module
        self.grad_buffers = {}
        self.expert_grads = []
        self.grad_buffer_param_index_map = {}
        self.param_to_grad_buffer = {}

        # Group parameters by their gradient type.
        grad_dtype_to_params = {}
        param_to_name = {}
        for name, param in self.module.named_parameters():
            if param.requires_grad and getattr(param, 'allreduce', True):
                if self.overlap_grad_reduce:
                    param.grad_added_to_main_grad = False
                param_to_name[param] = name
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

                params = grad_dtype_to_params.get(dtype, [])
                params.append(param)
                grad_dtype_to_params[dtype] = params

        # Allocate the grad buffers and map the grads.
        # The grad buffer under the hood creates buckets as appropriate based on bucket_size.
        for dtype, params in grad_dtype_to_params.items():

            self.grad_buffers[dtype] = GradBuffer(
                dtype,
                params,
                data_parallel_group,
                bucket_size,
                param_to_name,
                self.overlap_grad_reduce,
                self.use_distributed_optimizer,
            )

            self.grad_buffer_param_index_map[dtype] = self.grad_buffers[dtype].param_index_map
            for param in params:
                self.param_to_grad_buffer[param] = self.grad_buffers[dtype]

        # Allocate discreate buffer for MoE params' grads
        for param in self.module.parameters():
            if param.requires_grad and not getattr(param, 'allreduce', True):
                dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype
                param.main_grad = torch.zeros(
                    param.data.shape,
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )
                self.expert_grads.append(param.main_grad)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # Expand so we get access to grad_fn.
                param_tmp = param.expand_as(param)
                # Get the gradient accumulator function.
                grad_acc = param_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_param_hook(param, self.param_to_grad_buffer))
                self.grad_accs.append(grad_acc)

    def forward(self, *inputs, **kwargs):
        """
        Calls the wrapped module's forward() method.
        """
        return self.module(*inputs, **kwargs)

    def _make_param_hook(
        self, param: torch.nn.Parameter, param_to_grad_buffer: Dict[torch.nn.Parameter, GradBuffer]
    ):
        """
        Creates the all-reduce / reduce-scatter hook for backprop.
        """

        def param_hook(*unused):
            if param.requires_grad:
                if self.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and not getattr(param, 'grad_added_to_main_grad', False):
                    param.main_grad.add_(param.grad.data)
                param.grad = None
                if self.overlap_grad_reduce:
                    param_to_grad_buffer[param].register_grad_ready(param)

        return param_hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.is_last_microbatch = False
        try:
            yield
        finally:
            for grad_buffer in self.grad_buffers.values():
                grad_buffer.is_last_microbatch = True

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.
        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.
        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.finish_grad_sync()

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        for param in self.module.parameters():
            if param.requires_grad and hasattr(param, 'grad_added_to_main_grad'):
                param.grad_added_to_main_grad = False
        for grad_buffer in self.grad_buffers.values():
            grad_buffer.reset()
        for expert_grad in self.expert_grads:
            expert_grad.zero_()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            torch.distributed.broadcast(
                param.data,
                src=parallel_state.get_data_parallel_src_rank(),
                group=parallel_state.get_data_parallel_group(),
            )

    def state_dict(self, prefix='', keep_vars=False):
        """
        Returns a dictionary containing references to the whole state of the
        wrapped module.
        Both parameters and persistent buffers (e.g. running averages) are included.
        Keys are corresponding parameter and buffer names. Parameters and buffers
        set to None are not included.
        """
        return self.module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """
        Returns wrapped module's state_dict for checkpoint saving.
        """
        return self.module.state_dict_for_save_checkpoint(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """
        Copies parameters and buffers from state_dict into the wrapped module and its
        descendants. If strict is True, then the keys of state_dict must exactly match
        the keys returned by this module’s state_dict() function.
        """
        self.module.load_state_dict(state_dict, strict=strict)


def _allreduce_word_embedding_grads(model, config):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings
    parameters stay in sync. This should only run for models that support
    pipelined model parallelism (BERT and GPT-2).
    """

    if (
        parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support the interleaved schedule for T5 yet.
            model_module = model[0]

        # Look for module with 'pre_process' attribute to get around the fact that DDP and
        # other wrapper classes inherit from non-core MegatronModule that has
        # 'share_embeddings_and_output_weights' and 'shared_embedding_or_output_weight'
        # attributes already, causing get_attr_wrapped_model() to not unwrap anything here.
        # TODO: Clean this up once the wrapper classes inherit from core MegatronModule.
        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        if model_module.share_embeddings_and_output_weights:
            weight = model_module.shared_embedding_or_output_weight()
            grad = weight.main_grad
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _allreduce_position_embedding_grads(model, config):
    """
    All-reduce position_embeddings grad across first (encoder) and
    split (decoder) stages to ensure that position embeddings parameters
    stay in sync. This should only run for T5 models with pipeline
    parallelism.
    """
    if (
        parallel_state.is_rank_in_position_embedding_group()
        and parallel_state.get_pipeline_model_parallel_world_size() > 1
        and config.pipeline_model_parallel_split_rank is not None
    ):
        model_module = model[0]
        grad = get_attr_wrapped_model(
            model_module, 'language_model.embedding.position_embeddings.weight.main_grad'
        )
        torch.distributed.all_reduce(grad, group=parallel_state.get_position_embedding_group())


def _allreduce_embedding_grads(model, config):
    """All-reduce both word and position embeddings."""
    _allreduce_word_embedding_grads(model, config)
    _allreduce_position_embedding_grads(model, config)


def _allreduce_layernorm_grads(model, config):
    """All-reduce layernorm grads (for sequence parallelism)."""

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and config.sequence_parallel:
        grads = []
        for model_chunk in model:
            for param in get_attr_wrapped_model(model_chunk, 'parameters')():
                if getattr(param, 'sequence_parallel', False):
                    grad = param.main_grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_tensor_model_parallel_group())
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def _allreduce_expert_grads(model, config):
    """All-reduce expert grads (for expert parallelism)."""

    # All-reduce switchmlp parameters across data modulo expert parallel nodes
    if (
        config.expert_model_parallel_size > 1
        and config.expert_model_parallel_size < parallel_state.get_data_parallel_world_size()
    ):
        grads = []
        for model_chunk in model:
            for param in get_attr_wrapped_model(model_chunk, 'parameters')():
                if not getattr(param, 'allreduce', True):
                    grad = param.main_grad
                    grads.append(grad.data)
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, group=parallel_state.get_data_modulo_expert_parallel_group())
        for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
            buf.copy_(synced)


def finalize_model_grads(model):
    """All-reduce all grads across DP replicas, layernorm grads
    for sequence parallelism, and embedding grads across first and
    last pipeline stages (if not tied)."""

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()

    # All-reduce expert grads (for expert parallelism).
    if config.timers is not None:
        config.timers('expert-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_expert_grads(model, config)
    if config.timers is not None:
        config.timers('expert-grads-all-reduce').stop()
