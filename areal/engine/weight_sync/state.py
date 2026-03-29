"""State container for weight synchronization."""

from __future__ import annotations

import torch.distributed as dist

from areal.utils import logging

logger = logging.getLogger("[WeightSync]")


class WeightSyncState:
    """State container for weight synchronization between train and inference engines.

    This class holds all mutable state related to the XCCL weight update
    process group.  It is created once per engine and passed into the
    transport-level helper functions.

    Parameters
    ----------
    group_name : str
        Unique name for the NCCL/XCCL process group used for weight updates.
        Typically derived from the pipeline-parallel rank so that each PP
        stage has its own group.

    Attributes
    ----------
    group_initialized : bool
        Whether the XCCL weight update group has been created.
    group_name : str
        Name of the NCCL group for weight updates.
    master_addr : str
        Master address for TCP store initialization.
    master_port : int
        Master port for TCP store initialization.
    group : dist.ProcessGroup | None
        The distributed process group for weight updates.
    topology_version : int
        Monotonically increasing version counter that is bumped whenever
        the process group is torn down and rebuilt (elastic topology change).
    """

    def __init__(self, group_name: str) -> None:
        self.group_initialized: bool = False
        self.group_name: str = group_name
        self.master_addr: str = ""
        self.master_port: int = 0
        self.group: dist.ProcessGroup | None = None
        self.topology_version: int = 0

    def reset(self) -> None:
        """Reset the state so that a new group can be initialized.

        This is the first step of an elastic topology change: the old
        process group is destroyed (via :func:`teardown_xccl_group`) and
        the state is reset so that :func:`init_xccl_group` can be called
        again with a possibly different topology.
        """
        self.group_initialized = False
        self.group = None
        self.master_addr = ""
        self.master_port = 0
        self.topology_version += 1
        logger.info(
            f"WeightSyncState reset for group '{self.group_name}', "
            f"topology_version={self.topology_version}"
        )
