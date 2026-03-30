"""Colocation weight synchronization via direct tensor passing.

In colocation mode, training and inference processes share the same GPU.
NCCL cannot be used for inter-process communication on the same device.
Instead, this module gathers full tensors from the training model and
passes them directly to the inference engine via its
``update_weights_from_tensor`` API.

The approach:
1. Training process gathers full tensors (handling DTensor/FSDP sharding).
2. Tensors are chunked by ``weight_chunked_mem_mb`` to control peak memory.
3. Each chunk is sent to the inference engine (backend decides transport).

This module is engine-agnostic: callers supply a ``param_iterator`` that
yields ``(name, param)`` pairs already gathered to full tensors.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api.engine_api import InferenceEngine
    from areal.api.io_struct import WeightUpdateMeta

logger = logging.getLogger("ColocationSync")


@trace_perf("colocation.update_weights_from_tensor", category="comm")
def update_weights_from_tensor(
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    cpu_group: dist.ProcessGroup,
    param_iterator: Iterator[tuple[str, nn.Parameter | torch.Tensor]],
    get_full_tensor_fn: callable,
    use_lora: bool = False,
) -> None:
    """Update inference engine weights by direct tensor passing (colocation mode).

    Parameters
    ----------
    meta : WeightUpdateMeta
        Weight update metadata (must be type="tensor").
    rollout_engine : InferenceEngine
        The inference engine to update.
    cpu_group : dist.ProcessGroup
        CPU process group for barriers.
    param_iterator : Iterator[tuple[str, Parameter | Tensor]]
        Iterator over (name, param) pairs from the training model.
        For LoRA, caller should pre-filter to trainable params only.
    get_full_tensor_fn : callable
        Function that takes a Parameter/Tensor and returns the full
        (un-sharded) tensor on the current device.  Engine-specific:
        FSDP uses DTensor.full_tensor(), Megatron uses all_gather_param, etc.
    use_lora : bool
        If True, only export trainable (LoRA) parameters.
        Note: caller is expected to already filter param_iterator for LoRA.
    """
    if dist.get_rank() == 0:
        rollout_engine.pause_generation()

    dist.barrier(group=cpu_group)

    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
    main_rank = dist.get_rank() == 0

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in param_iterator:
        tensor = get_full_tensor_fn(param)

        # Non-main ranks only help with collective gather
        if not main_rank:
            continue

        tensor_size = tensor.numel() * tensor.element_size()

        if tensor_size + buffer_size > weight_chunked_mem_size and named_tensors:
            _flush_tensor_bucket(rollout_engine, named_tensors)
            buffer_size = 0

        named_tensors.append((name, tensor))
        buffer_size += tensor_size

    # Flush remaining tensors
    if named_tensors and main_rank:
        _flush_tensor_bucket(rollout_engine, named_tensors)

    dist.barrier(group=cpu_group)

    if dist.get_rank() == 0:
        rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=cpu_group)


def stage_weights_from_tensor(
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    cpu_group: dist.ProcessGroup,
    param_iterator: Iterator[tuple[str, nn.Parameter | torch.Tensor]],
    get_full_tensor_fn: callable,
    use_lora: bool = False,
) -> None:
    """Stage tensor weight update without pause/resume (for colocated orchestrator).

    In colocated mode, the orchestrator manages pause/resume lifecycle.
    This function only does the gather + send, without touching generation state.

    Parameters are the same as ``update_weights_from_tensor``.
    """
    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
    main_rank = dist.get_rank() == 0

    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, param in param_iterator:
        tensor = get_full_tensor_fn(param)

        if not main_rank:
            continue

        tensor_size = tensor.numel() * tensor.element_size()

        if tensor_size + buffer_size > weight_chunked_mem_size and named_tensors:
            _flush_tensor_bucket(rollout_engine, named_tensors)
            buffer_size = 0

        named_tensors.append((name, tensor))
        buffer_size += tensor_size

    if named_tensors and main_rank:
        _flush_tensor_bucket(rollout_engine, named_tensors)

    current_platform.synchronize()
    dist.barrier(group=cpu_group)


def _flush_tensor_bucket(
    rollout_engine: InferenceEngine,
    named_tensors: list[tuple[str, torch.Tensor]],
) -> None:
    """Send a bucket of tensors to the inference engine and clear the list."""
    if not named_tensors:
        return
    fut = rollout_engine.update_weights_from_tensor(list(named_tensors))
    fut.result()
    named_tensors.clear()
