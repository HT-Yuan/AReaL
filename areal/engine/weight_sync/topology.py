"""Topology manager for elastic weight synchronization.

The :class:`TopologyManager` wraps a :class:`WeightSyncState` and adds the
ability to **tear down** an existing NCCL weight-update group and **rebuild**
it with a different topology (e.g. after the inference engine has scaled up
or changed its parallelism configuration).

Usage
-----
::

    topo_mgr = TopologyManager(state)

    # First connection (same as before)
    topo_mgr.connect(meta, rollout_engine=..., is_sync_rank=..., ...)

    # --- later: inference engine scaled from 4 GPUs to 8 GPUs ---

    new_meta = WeightUpdateMeta.from_fsdp_xccl(new_allocation, ...)
    topo_mgr.reconnect(new_meta, rollout_engine=..., is_sync_rank=..., ...)

    # update_weights works transparently with the new group
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from areal.api import InferenceEngine, WeightUpdateMeta
from areal.engine.weight_sync.state import WeightSyncState
from areal.engine.weight_sync.transport_xccl import (
    init_xccl_group,
    teardown_xccl_group,
)
from areal.utils import logging

if TYPE_CHECKING:
    from areal.utils.lock import DistributedLock

logger = logging.getLogger("[WeightSync]")


class TopologyManager:
    """Manage elastic topology changes for weight synchronization.

    The manager is **thread-safe**: concurrent calls to :meth:`reconnect`
    and :meth:`update_weights_xccl` are serialised with a lock so that a
    weight update never races with a topology change.

    Parameters
    ----------
    state : WeightSyncState
        The underlying state container.  The manager takes ownership.
    """

    def __init__(self, state: WeightSyncState) -> None:
        self._state = state
        self._lock = threading.Lock()

    # -- Properties ----------------------------------------------------------

    @property
    def state(self) -> WeightSyncState:
        """Access the underlying :class:`WeightSyncState`."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Whether an XCCL group is currently active."""
        return self._state.group_initialized

    @property
    def topology_version(self) -> int:
        """Monotonically increasing version bumped on each reconnect."""
        return self._state.topology_version

    # -- Lifecycle -----------------------------------------------------------

    def connect(
        self,
        meta: WeightUpdateMeta,
        *,
        rollout_engine: InferenceEngine,
        is_sync_rank: bool,
        engine_lock: DistributedLock | None = None,
        logger_override: logging.Logger | None = None,
    ) -> None:
        """Create the initial XCCL weight-update group.

        This is a thin wrapper around :func:`init_xccl_group` that
        additionally guards against double-initialization.
        """
        with self._lock:
            if self._state.group_initialized:
                logger.warning(
                    "connect() called but group is already initialized; skipping."
                )
                return
            init_xccl_group(
                self._state,
                meta,
                rollout_engine=rollout_engine,
                is_sync_rank=is_sync_rank,
                engine_lock=engine_lock,
                logger_override=logger_override,
            )

    def reconnect(
        self,
        meta: WeightUpdateMeta,
        *,
        rollout_engine: InferenceEngine,
        is_sync_rank: bool,
        engine_lock: DistributedLock | None = None,
        logger_override: logging.Logger | None = None,
    ) -> None:
        """Tear down the existing group and create a new one.

        This is the core **P1 elastic** operation.  It allows the training
        engine to adapt to inference-engine topology changes (scale-up,
        scale-down, parallelism strategy change) without restarting.

        The method is **idempotent**: if no group exists yet it behaves
        exactly like :meth:`connect`.
        """
        with self._lock:
            if self._state.group_initialized:
                logger.info(
                    f"Tearing down existing weight-update group "
                    f"'{self._state.group_name}' (v{self._state.topology_version}) "
                    f"for reconnect."
                )
                teardown_xccl_group(self._state)

            init_xccl_group(
                self._state,
                meta,
                rollout_engine=rollout_engine,
                is_sync_rank=is_sync_rank,
                engine_lock=engine_lock,
                logger_override=logger_override,
            )
            logger.info(
                f"Reconnected weight-update group '{self._state.group_name}' "
                f"(v{self._state.topology_version})."
            )

    def disconnect(self) -> None:
        """Tear down the XCCL group without creating a new one."""
        with self._lock:
            if self._state.group_initialized:
                teardown_xccl_group(self._state)
