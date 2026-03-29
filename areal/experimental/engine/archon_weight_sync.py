"""Archon engine weight synchronization.

This module adapts the shared :mod:`areal.engine.weight_sync` primitives to
the Archon engine's specific needs (``state_dict_adapter``, ``engine_lock``,
pipeline parallel head check, DTensor/CPU-offload handling).

The public API consumed by :class:`ArchonEngine` is unchanged:

- :func:`init_weight_update_group`
- :func:`update_weights_from_distributed`
- :func:`update_weights_from_disk`

The ``WeightSyncState`` class is re-exported from the shared module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from areal.engine.weight_sync import (
    WeightSyncState,
    init_xccl_group,
    update_weights_disk,
    update_weights_xccl,
)
from areal.infra.platforms import current_platform
from areal.utils.perf_tracer import trace_perf

if TYPE_CHECKING:
    from areal.api import WeightUpdateMeta
    from areal.experimental.engine.archon_engine import ArchonEngine

# Re-export so ArchonEngine's imports stay unchanged
__all__ = [
    "WeightSyncState",
    "init_weight_update_group",
    "update_weights_from_distributed",
    "update_weights_from_disk",
]


def init_weight_update_group(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Initialize the weight update process group for XCCL synchronization."""
    init_xccl_group(
        state,
        meta,
        rollout_engine=engine.rollout_engine,
        is_sync_rank=engine.is_pipeline_parallel_head(),
        engine_lock=engine.engine_lock,
        logger_override=engine.logger,
    )


def _get_full_tensor(param: nn.Parameter) -> torch.Tensor:
    """Get full tensor from a parameter, handling DTensor and CPU offload."""
    tensor = param.data
    if isinstance(tensor, DTensor):
        if tensor.device.type != "cpu":
            return tensor.full_tensor()

        return DTensor.from_local(
            tensor.to_local(),
            device_mesh=tensor.device_mesh,
            placements=tensor.placements,
        ).full_tensor()
    else:
        if tensor.device.type == "cpu":
            tensor = tensor.to(current_platform.device_type)
        return tensor


def _iter_archon_sync_params(engine: ArchonEngine):
    """Iterate over Archon params, yielding (hf_name, tensor) for sync rank."""
    for name, param in engine._get_model_name_parameters():
        tensor = _get_full_tensor(param)

        if not engine.is_pipeline_parallel_head():
            continue

        if engine.state_dict_adapter is not None:
            hf_pairs = engine.state_dict_adapter.convert_single_to_hf(name, tensor)
        else:
            hf_pairs = [(name, tensor)]

        yield from hf_pairs


@trace_perf("archon_engine.update_weights_from_distributed", category="comm")
def update_weights_from_distributed(
    state: WeightSyncState,
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Update weights by broadcasting from training engine to inference engine."""
    assert engine.rollout_engine is not None

    update_weights_xccl(
        state,
        meta,
        rollout_engine=engine.rollout_engine,
        cpu_group=engine.cpu_group,
        param_iter=_iter_archon_sync_params(engine),
        is_sync_rank=engine.is_pipeline_parallel_head(),
        engine_lock=engine.engine_lock,
    )


@trace_perf("archon_engine.update_weights_from_disk", category="io")
def update_weights_from_disk(
    meta: WeightUpdateMeta,
    engine: ArchonEngine,
) -> None:
    """Update weights by saving to disk and loading in inference engine."""
    from areal.experimental.engine.archon_checkpoint import save_model_to_hf

    update_weights_disk(
        meta,
        rollout_engine=engine.rollout_engine,
        cpu_group=engine.cpu_group,
        save_fn=lambda: save_model_to_hf(engine, meta.path, engine.tokenizer, None),
        experiment_name=engine.config.experiment_name,
        trial_name=engine.config.trial_name,
        version=engine.get_version(),
    )
