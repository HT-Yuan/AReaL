"""Disk-based transport for weight synchronization.

Provides the ``update_weights_disk`` function that saves the model to disk
in HuggingFace format and notifies the inference engine to reload.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from datetime import datetime
from typing import Any

import torch.distributed as dist

from areal.api import InferenceEngine, WeightUpdateMeta
from areal.infra.platforms import current_platform
from areal.utils import name_resolve, names
from areal.utils.perf_tracer import trace_perf


@trace_perf("weight_sync.update_weights_disk", category="io")
def update_weights_disk(
    meta: WeightUpdateMeta,
    *,
    rollout_engine: InferenceEngine,
    cpu_group: dist.ProcessGroup,
    save_fn: Callable[..., Any],
    experiment_name: str,
    trial_name: str,
    version: int,
) -> None:
    """Save model to disk and signal the inference engine to reload.

    Parameters
    ----------
    meta : WeightUpdateMeta
        Must have ``type == "disk"`` and a valid ``path``.
    rollout_engine : InferenceEngine
        Target inference engine.
    cpu_group : dist.ProcessGroup
        CPU (gloo) process group for cross-rank barriers.
    save_fn : Callable
        A callable that persists the model to ``meta.path``.  Typically
        ``engine._save_model_to_hf`` (FSDP/Megatron) or
        ``save_model_to_hf`` (Archon).
    experiment_name : str
        Experiment name for name-resolve registration.
    trial_name : str
        Trial name for name-resolve registration.
    version : int
        Current model version for name-resolve registration.
    """
    fut = Future()

    if dist.get_rank() == 0:
        fut = rollout_engine.update_weights_from_disk(meta)

    assert meta.path is not None
    save_fn()

    if dist.get_rank() == 0:
        update_name = names.update_weights_from_disk(
            experiment_name,
            trial_name,
            version,
        )
        name_resolve.add(
            update_name, str(datetime.now().timestamp()), keepalive_ttl=120
        )
        fut.result()

    current_platform.synchronize()
    dist.barrier(group=cpu_group)
