"""XCCL (NCCL) transport for weight synchronization.

This module provides the core NCCL-based weight broadcast transport used
to synchronize model parameters from the training engine (rank 0 in the
weight update group) to the inference engine workers.

The three public entry points mirror the lifecycle:

1. :func:`init_xccl_group` – create the NCCL process group.
2. :func:`update_weights_xccl` – broadcast all model parameters.
3. :func:`teardown_xccl_group` – destroy the group (elastic topology).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from areal.api import InferenceEngine, ParamSpec, WeightUpdateMeta
from areal.engine.core.distributed import init_custom_process_group
from areal.engine.weight_sync.state import WeightSyncState
from areal.infra.platforms import current_platform
from areal.utils import logging
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.network import find_free_ports, gethostip
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from collections.abc import Iterator

    from areal.utils.lock import DistributedLock

logger = logging.getLogger("[WeightSync]")


# ---------------------------------------------------------------------------
# Group lifecycle
# ---------------------------------------------------------------------------


def init_xccl_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    *,
    rollout_engine: InferenceEngine,
    is_sync_rank: bool,
    engine_lock: DistributedLock | None = None,
    logger_override: logging.Logger | None = None,
) -> None:
    """Initialize the XCCL weight update process group.

    Parameters
    ----------
    state : WeightSyncState
        Mutable state container.
    meta : WeightUpdateMeta
        Metadata for the weight update (will be mutated with address info).
    rollout_engine : InferenceEngine
        The inference engine that will join the group on the remote side.
    is_sync_rank : bool
        ``True`` on the single training rank that participates in the
        weight broadcast (rank 0 for FSDP, PP-head for Megatron/Archon).
    engine_lock : DistributedLock | None
        Optional lock to hold during group creation (Megatron/Archon).
    logger_override : logging.Logger | None
        Use a specific logger instead of the module-level one.
    """
    assert meta.type == "xccl"
    _log = logger_override or logger

    state.master_addr = gethostip()
    state.master_port = find_free_ports(1)[0]

    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    # Processes launched with torchrun set TORCHELASTIC_USE_AGENT_STORE=True,
    # which blocks creating another TCP store for weight update.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)

    if is_sync_rank:
        assert meta.gen_allocation is not None

        def _do_init() -> None:
            fut = rollout_engine.init_weights_update_group(meta)

            gen_world_size = meta.gen_allocation.parallel.world_size
            _log.info(
                f"Initializing weight update group: type={meta.type}, "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port}, "
                f"group={meta.nccl_group_name}"
            )
            state.group = init_custom_process_group(
                backend=current_platform.communication_backend,
                world_size=gen_world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            fut.result()

        if engine_lock is not None:
            with engine_lock:
                _do_init()
        else:
            _do_init()

    state.group_initialized = True


def teardown_xccl_group(state: WeightSyncState) -> None:
    """Tear down an existing XCCL weight update group.

    After calling this function the *state* is reset and
    :func:`init_xccl_group` can be called again with a new topology.
    """
    if state.group is not None:
        try:
            dist.destroy_process_group(state.group)
        except Exception:
            # Best-effort; the group may already be invalid.
            logger.warning(
                f"Failed to destroy process group '{state.group_name}', ignoring."
            )
    state.reset()


# ---------------------------------------------------------------------------
# Bucket broadcast
# ---------------------------------------------------------------------------


def broadcast_bucket(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    rollout_engine: InferenceEngine,
    named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    *,
    engine_lock: DistributedLock | None = None,
    peft_config: dict | None = None,
) -> None:
    """Broadcast a bucket of named tensors to the inference engine.

    This is the low-level primitive shared by all engines.  Higher-level
    ``update_weights_xccl`` iterates over model parameters and calls this
    function for each chunk.

    Parameters
    ----------
    state : WeightSyncState
        Must have ``group_initialized == True``.
    meta : WeightUpdateMeta
        Metadata forwarded to the inference engine.
    rollout_engine : InferenceEngine
        The inference engine to receive the weights.
    named_tensors : list[tuple[str, Tensor]]
        The chunk of (name, tensor) pairs to broadcast.
    engine_lock : DistributedLock | None
        Optional lock to hold during the broadcast (Megatron/Archon).
    peft_config : dict | None
        Optional LoRA PEFT config to inject into *meta* before sending.
    """
    if not named_tensors:
        return

    def _do_broadcast() -> None:
        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        if peft_config is not None:
            meta.peft_config = peft_config

        fut = rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        assert state.group is not None
        for _, tensor in named_tensors:
            t = tensor.data if isinstance(tensor, nn.Parameter) else tensor
            handles.append(dist.broadcast(t, src=0, group=state.group, async_op=True))
        for handle in handles:
            handle.wait()

        fut.result()
        named_tensors.clear()

    if engine_lock is not None:
        with engine_lock:
            _do_broadcast()
    else:
        _do_broadcast()


# ---------------------------------------------------------------------------
# Full weight update
# ---------------------------------------------------------------------------


@trace_perf("weight_sync.update_weights_xccl", category="comm")
def update_weights_xccl(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    *,
    rollout_engine: InferenceEngine,
    cpu_group: dist.ProcessGroup,
    param_iter: Iterator[tuple[str, torch.Tensor]],
    is_sync_rank: bool,
    engine_lock: DistributedLock | None = None,
    peft_config: dict | None = None,
) -> None:
    """Broadcast all model parameters from train engine to inference engine.

    This function implements the full weight-update lifecycle:

    1. Pause inference generation.
    2. Iterate over ``param_iter``, chunk by ``weight_chunked_mem_mb``.
    3. Broadcast each chunk via :func:`broadcast_bucket`.
    4. Resume inference generation.

    Parameters
    ----------
    state : WeightSyncState
        Initialised state with ``group_initialized == True``.
    meta : WeightUpdateMeta
        Metadata (XCCL addresses, chunk size, etc.).
    rollout_engine : InferenceEngine
        Target inference engine.
    cpu_group : dist.ProcessGroup
        CPU (gloo) process group used for barriers across train ranks.
    param_iter : Iterator[tuple[str, torch.Tensor]]
        Iterator over ``(name, full_tensor)`` pairs.  For non-sync ranks
        this may yield nothing (they only help with the all-gather).
    is_sync_rank : bool
        True on the single rank that does the broadcast (rank 0 / PP-head).
    engine_lock : DistributedLock | None
        Lock held during each bucket broadcast (Megatron/Archon).
    peft_config : dict | None
        LoRA PEFT config to embed in the meta for inference engine.
    """
    # Populate meta with stored address info
    meta.nccl_master_address = state.master_addr
    meta.nccl_master_port = state.master_port
    meta.nccl_group_name = state.group_name

    if dist.get_rank() == 0:
        rollout_engine.pause_generation()

    dist.barrier(group=cpu_group)

    weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
    buffer_size = 0
    named_tensors: list[tuple[str, torch.Tensor]] = []

    for name, tensor in param_iter:
        if not is_sync_rank:
            continue

        tensor_size = tensor.numel() * tensor.element_size()

        if tensor_size + buffer_size > weight_chunked_mem_size:
            broadcast_bucket(
                state,
                meta,
                rollout_engine,
                named_tensors,
                engine_lock=engine_lock,
                peft_config=peft_config,
            )
            buffer_size = 0

        named_tensors.append((name, tensor))
        buffer_size += tensor_size

    # Flush remaining
    if named_tensors:
        broadcast_bucket(
            state,
            meta,
            rollout_engine,
            named_tensors,
            engine_lock=engine_lock,
            peft_config=peft_config,
        )

    dist.barrier(group=cpu_group)

    if dist.get_rank() == 0:
        rollout_engine.continue_generation()

    current_platform.synchronize()
    dist.barrier(group=cpu_group)
