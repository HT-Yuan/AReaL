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
        *,
        train_pre_offloaded: bool = False,
    ) -> None:
        self._train_engine: TrainEngine = train_engine
        self._inf_engine: InferenceEngine = inf_engine
        self._train_on_gpu: bool = not train_pre_offloaded
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

    def _sync_pending_weight_update(self, *, continue_generation: bool = True) -> None:
        if self._pending_weight_update is None:
            return

        if self._is_rollout_coordinator():
            meta = self._pending_weight_update
            if meta.type == "disk":
                self._inf_engine.sync_weights_from_disk(meta)
            # tensor mode: weights already transferred during stage phase
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
        """Stage a colocated weight update for the next inference phase."""
        if meta.type not in ("disk", "tensor"):
            raise ValueError(
                "Colocated orchestration only supports disk or tensor weight updates. "
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

            meta = self._pending_weight_update
            if meta is not None:
                if meta.type == "disk":
                    self._inf_engine.sync_weights_from_disk(meta)
                # tensor mode: weights already transferred via IPC during stage

            self._inf_engine.continue_generation()

        self._barrier()
        self._pending_weight_update = None
        self._inf_on_gpu = True

    def publish_weights(
        self,
        meta: WeightUpdateMeta,
        *,
        set_version_fn: Any | None = None,
    ) -> None:
        """Stage weight update without pause/resume (colocated mode).

        In colocated mode the inference engine is already paused (GPU belongs
        to training), so we only stage the update.  The staged weights will be
        flushed during ``prepare_for_inference``.

        Parameters
        ----------
        meta : WeightUpdateMeta
            Must carry the target version.
        set_version_fn : callable, optional
            ``fn(version)`` called after staging to propagate the version to
            all engines (actor, critic, rollout, eval-rollout).
        """
        self.update_weights(meta)
        if set_version_fn is not None and meta.version is not None:
            set_version_fn(meta.version)

    def switch_to_inference(
        self,
        *,
        global_step: int,
        capture_stats_fn: Any | None = None,
    ) -> None:
        """Capture train stats, switch GPU to inference, and set rollout version.

        Encapsulates the colocated-specific work that used to live in
        ``PPOTrainer._prepare_inference_phase``.
        """
        if capture_stats_fn is not None:
            capture_stats_fn()

        with (
            stats_tracker.record_timing("colocated_switch_to_inference"),
            perf_tracer.trace_scope(
                "train.colocated_switch_to_inference",
                category=Category.COMM,
                args={"global_step": global_step},
            ),
        ):
            self.prepare_for_inference()

    def finalize(self) -> None:
        """Ensure the training engine is onloaded before teardown."""
        if not self._train_on_gpu:
            self._train_engine.onload()
            self._train_on_gpu = True
