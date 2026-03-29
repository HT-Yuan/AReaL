"""Weight synchronization module for training-to-inference weight updates.

This module provides a unified, pluggable weight synchronization framework
that abstracts away the transport mechanism (XCCL, disk, shared memory)
and supports elastic topology changes (inference engine scaling).

Components
----------
WeightSyncState
    State container for weight synchronization (process group, addresses, etc.).
XcclTransport
    NCCL/XCCL-based weight broadcast transport.
DiskTransport
    Disk-based weight update transport.
TopologyManager
    Manages elastic topology changes for weight sync process groups.

Usage
-----
Engines create a ``WeightSyncState`` during ``__init__`` and delegate all
weight synchronization logic to functions in this module::

    from areal.engine.weight_sync import (
        WeightSyncState,
        init_xccl_group,
        update_weights_xccl,
        update_weights_disk,
        teardown_xccl_group,
    )
"""

from areal.engine.weight_sync.state import WeightSyncState
from areal.engine.weight_sync.topology import TopologyManager
from areal.engine.weight_sync.transport_disk import update_weights_disk
from areal.engine.weight_sync.transport_xccl import (
    broadcast_bucket,
    init_xccl_group,
    teardown_xccl_group,
    update_weights_xccl,
)

__all__ = [
    "WeightSyncState",
    "TopologyManager",
    "broadcast_bucket",
    "init_xccl_group",
    "teardown_xccl_group",
    "update_weights_xccl",
    "update_weights_disk",
]
