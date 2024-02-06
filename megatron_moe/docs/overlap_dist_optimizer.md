# OverlappedDistributedOptimizer

In the vanilla Megatron-LM, users can leverage [`DistributedOptimizer`](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/distrib_optimizer.md) to partition gradients and optimizer states to reduce GPU memory occupation. After accumulated all gradients in GA, `DistributedOptimizer` employs a `ReduceScatter` operation to scatter the gradients to the corresponding ranks. Each rank then updates the local parameters, and then collect the remaining parameters through an `AllGather` operation from all the other ranks. However, we observe a significant overhead on communication under small GA settings (over 50% time consumption without GA). 

To mitigate the overhead, we try to overlap the collective communication with computation, according to the partition strategy in DeepSpeed ZeRO Stage-2. This strategy fails to scale. It takes too many small `Reduce` operations at large scale, which makes it under-utilize the inter-connection bandwidth.

We abstract the above problem into two aspects:
1. Finding the room for overlapping communication with computation logically.
2. Implementing a partition strategy that fully utilizes the room for overlapping and inter-connection bandwidth, without introducing overhead in term of communication volume.

In this case, we propose `OverlappedDistributedOptimizer `, with a novel partition strategy of gradients and optimizer states. The design principles are summarized as follows:

* Common optimizers such as Adam and SGD update each value in the parameters independently. Therefore, it is not necessary to keep each parameter as a whole.
* Any single collective communication operation should commit a sufficient amount of data, making full use of the communication bandwidth.
* No extra communication volume or GPU memory-copy should be involved.


**Brief introduction to OverlappedDistributedOptimizer**

![](docs/images/overlap_dist_optimizer/shuffle_grad_bucket.png "The partition strategy")
<center>Figure 1. The partition strategy</center>

As shown in Figure 1, all parameters are assigned to their respective `Buckets` during the initialization of `OverlappedDistributedOptimizer`. All the model parameters within a `Bucket` are complete, with each parameter belonging to only one `Bucket`. Conceptually, each `Bucket` is divided equally into *P* (the number of ranks of the data parallel group) shards. Each rank would be responsible for one shard. The `Buckets` would be placed in a local queue (Local grad bucket queue) to ensure the communication order. During the training process, the data parallel groups exchange the required gradients at the `Bucket` level through collective communication.

![](docs/images/overlap_dist_optimizer/communicate_params.png "The communication mechanism")
<center>Figure 2. The communication mechanism</center>

`OverlappedDistributedOptimizer` incorporates an efficient communication mechanism over the `Buckets`. `OverlappedDistributedOptimizer` initializes a local buffer called `PartitionedParameter` with a size equal to the sum of sizes of all parameters that the current rank is responsible for. The respective parameters are taken from the pre-sharded model parameters and assigned to the `PartitionedParameter`. Besides, a buffer called `PartitionedGradient`, with the same size as `PartitionedParameter`, is created to store the gradients corresponding to the `PartitionedParameter`. Then, The communication mechanism mainly consists of the following three procedures:

a) As shown in Figure 2-(i), once a parameter's gradient is obtained, the gradient would be copied to the corresponding position in the Bucket. Once all gradients for the parameters in a Bucket are collected, a single `ReduceScatter` operation is performed to exchange the gradients, with the corresponding position in the `PartitionedGradient` as destination.

b) As shown in Figure 2-(ii), each rank updates `PartitionedParameter` by the `PartitionedGradient ` once all `ReduceScatter` operations are finished.

c) As shown in Figure 2-(iii), each rank re-constructs the full parameters from all the other ranks through `AllGather` with the logical `Bucket`.


Specifically, we reduce the memory copy and GPU memory occupation through the following approaches:

a. During the initialization of `OverlappedDistributedOptimizer`, a buffer called `ParameterBuffer` is allocated with the same size as the sum of all parameter sizes, and all model parameters are actually placed in `ParameterBuffer`. The destination addresses for re-constructing the full parameters via `AllGather` can directly reference to the corresponding positions in `ParameterBuffer`. It avoids the temporary memory allocation and reduces GPU memory copy. (This optimization is inspired by DeepSpeed).

b. Once copying gradients to the `Bucket` has been complete, the original space for gradients can be released, reducing GPU memory usage. Additionally, the memory for `Bucket` can also be released after the `ReduceScatter` operation. On top of this, we introduce a *Buffer Alternation* mechanism to avoid the issue of memory fragmentation caused by frequent memory allocation and deallocation.


**Performance**

As ZERO2 brings extra communication operators when sync gradients during Gradient Accumulation steps, Overlapped Distributed Optimizer is only suitable for large cluster training (>256) without or less Gradient Accumulation steps.