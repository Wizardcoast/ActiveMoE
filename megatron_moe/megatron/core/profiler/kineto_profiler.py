# encoding:utf-8

import functools

import os
import torch


def kineto_handler(prof, save_dir, profile_ranks=None):
    # TODO: use Holistic Trace Analysis (HTA) for more detailed/high-level performance analysis.
    # Now only provide the saving of profiling files.
    if save_dir is not None:
        file_path = os.path.join(save_dir, "trace_{}.json".format(torch.distributed.get_rank()))
        if profile_ranks is not None and torch.distributed.get_rank() in profile_ranks:
            prof.export_chrome_trace(file_path)
        else:
            prof.export_chrome_trace(file_path)


def get_kineto_profiler(save_dir,
                        profile_ranks=False,
                        schedule=[2, 1, 1, 1],
                        record_shapes: bool = False,
                        profile_memory: bool = False,
                        with_stack: bool = False,
                        with_flops: bool = False,):
    assert len(schedule) == 4, "Need four values for [wait, warmup, active, repeat] steps."
    p = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        with_stack=with_stack,
        with_flops=with_flops,
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        schedule=torch.profiler.schedule(
            wait=schedule[0],
            warmup=schedule[1],
            active=schedule[2],
            repeat=schedule[3]),
        on_trace_ready=functools.partial(kineto_handler, save_dir=save_dir, profile_ranks=profile_ranks)
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
        )
    return p
