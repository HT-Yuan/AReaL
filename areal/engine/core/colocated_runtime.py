"""Colocated (GPU time-sharing) orchestration for on-policy training.

In colocated mode, the training engine and inference engine share the same
GPUs and alternate between offloaded/onloaded states.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from areal.utils import logging, perf_tracer, stats_tracker
from areal.utils.perf_tracer import Category

if TYPE_CHECKING:
    from areal.api import InferenceEngine, TrainEngine

logger = logging.getLogger("Colocated")


class ColocatedOrchestrator:
    """Orchestrate GPU ownership between colocated training and inference."""

    def __init__(
        self,
        train_engine: TrainEngine,
        inf_engine: InferenceEngine,
        *,
        train_pre_offloaded: bool = False,
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self._train_on_gpu: bool = not train_pre_offloaded
        self._inf_on_gpu: bool = True

    def _is_rollout_coordinator(self) -> bool:
        return not dist.is_initialized() or dist.get_rank() == 0

    def _barrier(self) -> None:
        if not dist.is_initialized():
            return
        cpu_group = self._train_engine.cpu_group
        if cpu_group is None:
            dist.barrier()
            return
        dist.barrier(group=cpu_group)

    @contextmanager
    def _training_switch_scope(self, global_step: int | None):
        if global_step is None:
            yield
            return

        with (
            stats_tracker.record_timing("colocated_switch_to_train"),
            perf_tracer.trace_scope(
                "train.colocated_switch_to_train",
                category=Category.COMM,
                args={"global_step": global_step},
            ),
        ):
            yield

    @contextmanager
    def prepare_batch_context(
        self,
        inner_context: Any,
        *,
        global_step: int | None = None,
    ):
        with inner_context:
            try:
                yield
            except Exception:
                raise
            else:
                with self._training_switch_scope(global_step):
                    self.prepare_for_training()

    def prepare_for_training(self) -> None:
        """Switch GPU ownership from inference to training."""
        if self._train_on_gpu:
            logger.debug("Training engine already on GPU, skipping switch")
            return

        if self._is_rollout_coordinator():
            logger.info("Switching to training mode")

        # Pause local submission on every rank first so no new requests are queued.
        self._inf_engine.pause()

        # Only one coordinator should touch the shared rollout servers.
        if self._is_rollout_coordinator():
            self._inf_engine.pause_generation()
            if self._inf_on_gpu:
                self._inf_engine.offload()

        self._barrier()
        self._inf_on_gpu = False

        # All training ranks must participate in the training-engine collective.
        self._train_engine.onload()
        self._train_on_gpu = True

    def prepare_for_inference(self) -> None:
        """Switch GPU ownership from training to inference.

        Callers remain responsible for any colocated weight-finalization work
        (for example, syncing a staged disk checkpoint) before marking the
        switch complete.
        """
        if self._inf_on_gpu:
            logger.debug("Inference engine already on GPU, skipping switch")
            return

        if self._is_rollout_coordinator():
            logger.info("Switching to inference mode")

        if self._train_on_gpu:
            self._train_engine.offload()
        self._train_on_gpu = False

        self._barrier()

        if self._is_rollout_coordinator() and not self._inf_on_gpu:
            self._inf_engine.onload()

    def complete_inference_switch(self) -> None:
        """Finalize a training→inference ownership switch after engine work completes."""
        self._barrier()
        self._inf_on_gpu = True

    def finalize(self) -> None:
        """Ensure the training engine is onloaded before teardown."""
        if not self._train_on_gpu:
            self._train_engine.onload()
            self._train_on_gpu = True
