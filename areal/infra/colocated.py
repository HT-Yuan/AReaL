"""Colocated (GPU time-sharing) orchestration for on-policy training.

In colocated mode, the training engine and inference engine share the same
GPUs and alternate between offloaded/onloaded states.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from areal.api.io_struct import WeightUpdateMeta
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
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self._train_on_gpu: bool = True
        self._inf_on_gpu: bool = True
        self._pending_weight_update: WeightUpdateMeta | None = None

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

    def _stage_weight_update(self, meta: WeightUpdateMeta) -> None:
        stage_weight_update = getattr(self._train_engine, "_stage_weight_update", None)
        if stage_weight_update is None:
            raise AttributeError(
                "Train engine does not support colocated staged weight updates."
            )
        stage_weight_update(meta)
        self._pending_weight_update = meta

    def _sync_pending_weight_update(
        self, *, continue_generation: bool = True
    ) -> None:
        if self._pending_weight_update is None:
            return

        if self._is_rollout_coordinator():
            self._inf_engine.sync_weights_from_disk(self._pending_weight_update)
            if continue_generation:
                self._inf_engine.continue_generation()

        self._barrier()
        self._pending_weight_update = None

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

    def update_weights(self, meta: WeightUpdateMeta) -> None:
        """Stage a colocated disk weight update for the next inference phase."""
        if meta.type != "disk":
            raise ValueError(
                "Colocated orchestration only supports disk-based weight updates. "
                f"Got '{meta.type}'."
            )
        self._stage_weight_update(meta)

    def initial_offload_training(self) -> None:
        """Offload training once so inference owns the GPU before first rollout."""
        if not self._train_on_gpu:
            logger.warning(
                "initial_offload_training called but training engine is already off GPU."
            )
            return

        if self._is_rollout_coordinator():
            logger.info("Initial offload: moving training engine off GPU")
        self._train_engine.offload()
        self._train_on_gpu = False
        self._sync_pending_weight_update(continue_generation=False)

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
        """Switch GPU ownership from training to inference and flush staged weights.

        Rollout submission remains paused here; the trainer resumes it after
        evaluation/logging so the main loop keeps a single resume owner.
        """
        if self._inf_on_gpu and self._pending_weight_update is None:
            logger.debug("Inference engine already on GPU, skipping switch")
            return

        if self._is_rollout_coordinator():
            logger.info("Switching to inference mode")

        if self._train_on_gpu:
            self._train_engine.offload()
        self._train_on_gpu = False

        self._barrier()

        if self._is_rollout_coordinator():
            if not self._inf_on_gpu:
                self._inf_engine.onload()
            if self._pending_weight_update is not None:
                self._inf_engine.sync_weights_from_disk(self._pending_weight_update)
            self._inf_engine.continue_generation()

        self._barrier()
        self._pending_weight_update = None
        self._inf_on_gpu = True
