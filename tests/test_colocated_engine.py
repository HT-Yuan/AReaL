"""Unit tests for colocated orchestration and scheduler-driven trainer behavior."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from areal.api.cli_args import SchedulingStrategy, SchedulingStrategyType
from areal.api.io_struct import WeightUpdateMeta
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.colocated import ColocatedOrchestrator
from areal.infra.controller.rollout_controller import RolloutController
from areal.infra.controller.train_controller import TrainController
from areal.infra.remote_inf_engine import RemoteInfEngine
from areal.infra.scheduler.local import LocalScheduler, _apply_env_patch
from areal.trainer.rl_trainer import PPOTrainer
from areal.utils import names


@pytest.fixture
def mock_train_engine():
    engine = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    return engine


@pytest.fixture
def mock_inf_engine():
    engine = MagicMock()
    engine.pause = MagicMock()
    engine.resume = MagicMock()
    engine.pause_generation = MagicMock()
    engine.continue_generation = MagicMock()
    engine.offload = MagicMock()
    engine.onload = MagicMock()
    engine.sync_weights_from_disk = MagicMock()
    engine.set_version = MagicMock()
    return engine


@pytest.fixture
def orchestrator(mock_train_engine, mock_inf_engine):
    return ColocatedOrchestrator(
        train_engine=mock_train_engine,
        inf_engine=mock_inf_engine,
    )


class TestColocatedOrchestrator:
    def test_initial_state(self, orchestrator):
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is True

    def test_initial_state_with_pre_offloaded_training(
        self, mock_train_engine, mock_inf_engine
    ):
        orchestrator = ColocatedOrchestrator(
            train_engine=mock_train_engine,
            inf_engine=mock_inf_engine,
            train_pre_offloaded=True,
        )

        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_initial_offload_training(self, orchestrator, mock_train_engine):
        orchestrator.initial_offload_training()

        mock_train_engine.offload.assert_called_once()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_initial_offload_training_flushes_pending_weight_update(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v0", version=0)
        orchestrator.update_weights(meta)

        orchestrator.initial_offload_training()

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.sync_weights_from_disk.assert_called_once_with(meta)
        mock_inf_engine.continue_generation.assert_not_called()
        assert orchestrator._pending_weight_update is None

    def test_update_weights_rejects_non_disk_or_tensor_meta(self, orchestrator):
        meta = WeightUpdateMeta(type="xccl")

        with pytest.raises(ValueError, match="disk or tensor"):
            orchestrator.update_weights(meta)

    def test_prepare_for_training_switches_gpu_owner(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_called_once()
        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is False

    def test_prepare_for_training_orders_local_and_remote_rollout_shutdown(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        events: list[str] = []
        mock_inf_engine.pause.side_effect = lambda: events.append("pause")
        mock_inf_engine.pause_generation.side_effect = lambda: events.append(
            "pause_generation"
        )
        mock_inf_engine.offload.side_effect = lambda: events.append("inf_offload")
        mock_train_engine.onload.side_effect = lambda: events.append("train_onload")

        orchestrator.initial_offload_training()
        events.clear()

        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()

        assert events == [
            "pause",
            "pause_generation",
            "inf_offload",
            "train_onload",
        ]

    def test_prepare_for_training_only_coordinator_controls_shared_rollout_server(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        mock_train_engine.offload.reset_mock()

        with (
            patch("areal.infra.colocated.dist.is_initialized", return_value=True),
            patch("areal.infra.colocated.dist.get_rank", return_value=3),
            patch("areal.infra.colocated.dist.barrier") as mock_barrier,
        ):
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_not_called()
        mock_inf_engine.offload.assert_not_called()
        mock_train_engine.onload.assert_called_once()
        mock_barrier.assert_called()
        assert orchestrator._train_on_gpu is True
        assert orchestrator._inf_on_gpu is False

    def test_prepare_batch_context_switches_to_training_after_success(
        self, orchestrator
    ):
        events: list[str] = []

        @contextmanager
        def inner_context():
            events.append("enter")
            yield
            events.append("exit")

        orchestrator.prepare_for_training = MagicMock(
            side_effect=lambda: events.append("switch_to_train")
        )

        with orchestrator.prepare_batch_context(inner_context()):
            events.append("body")

        orchestrator.prepare_for_training.assert_called_once_with()
        assert events == ["enter", "body", "exit", "switch_to_train"]

    def test_prepare_batch_context_skips_training_switch_on_exception(
        self, orchestrator
    ):
        @contextmanager
        def inner_context():
            yield

        orchestrator.prepare_for_training = MagicMock()

        with pytest.raises(RuntimeError, match="boom"):
            with orchestrator.prepare_batch_context(inner_context()):
                raise RuntimeError("boom")

        orchestrator.prepare_for_training.assert_not_called()

    def test_prepare_for_inference_switches_gpu_owner_and_syncs_weights(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
        mock_inf_engine.pause.reset_mock()
        mock_inf_engine.pause_generation.reset_mock()
        mock_inf_engine.offload.reset_mock()
        mock_inf_engine.onload.reset_mock()
        mock_inf_engine.sync_weights_from_disk.reset_mock()
        mock_inf_engine.continue_generation.reset_mock()
        mock_inf_engine.resume.reset_mock()
        mock_train_engine.onload.reset_mock()
        mock_train_engine.offload.reset_mock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v1", version=1)
        orchestrator.update_weights(meta)
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_inference()

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_called_once()
        mock_inf_engine.set_version.assert_not_called()
        mock_inf_engine.sync_weights_from_disk.assert_called_once_with(meta)
        mock_inf_engine.continue_generation.assert_called_once()
        mock_inf_engine.resume.assert_not_called()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_for_inference_only_coordinator_controls_shared_rollout_server(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
        mock_train_engine.offload.reset_mock()
        mock_inf_engine.onload.reset_mock()
        mock_inf_engine.sync_weights_from_disk.reset_mock()
        mock_inf_engine.continue_generation.reset_mock()
        mock_inf_engine.resume.reset_mock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v3", version=3)
        orchestrator.update_weights(meta)
        with patch("areal.infra.colocated.dist.is_initialized", return_value=True):
            with patch("areal.infra.colocated.dist.get_rank", return_value=5):
                with patch("areal.infra.colocated.dist.barrier") as mock_barrier:
                    orchestrator.prepare_for_inference()

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_not_called()
        mock_inf_engine.sync_weights_from_disk.assert_not_called()
        mock_inf_engine.continue_generation.assert_not_called()
        mock_inf_engine.resume.assert_not_called()
        mock_barrier.assert_called()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_for_inference_allows_unversioned_meta(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
        mock_train_engine.offload.reset_mock()
        mock_inf_engine.onload.reset_mock()
        mock_inf_engine.set_version.reset_mock()
        mock_inf_engine.sync_weights_from_disk.reset_mock()
        mock_inf_engine.continue_generation.reset_mock()
        mock_inf_engine.resume.reset_mock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v_missing")
        orchestrator.update_weights(meta)
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_inference()

        mock_train_engine.offload.assert_called_once()
        mock_inf_engine.onload.assert_called_once()
        mock_inf_engine.set_version.assert_not_called()
        mock_inf_engine.sync_weights_from_disk.assert_called_once_with(meta)
        mock_inf_engine.continue_generation.assert_called_once()
        mock_inf_engine.resume.assert_not_called()
        assert orchestrator._train_on_gpu is False
        assert orchestrator._inf_on_gpu is True

    def test_prepare_calls_are_idempotent(
        self, orchestrator, mock_train_engine, mock_inf_engine
    ):
        orchestrator.initial_offload_training()
        with patch("areal.infra.colocated.dist.is_initialized", return_value=False):
            orchestrator.prepare_for_training()
            orchestrator.prepare_for_training()

        mock_inf_engine.pause.assert_called_once()
        mock_inf_engine.pause_generation.assert_called_once()
        mock_inf_engine.offload.assert_called_once()
        mock_train_engine.onload.assert_called_once()

    def test_publish_weights_only_stages_update(
        self, orchestrator, mock_train_engine
    ):
        meta = WeightUpdateMeta(type="disk", path="/tmp/w", version=5)

        orchestrator.publish_weights(meta)

        mock_train_engine._stage_weight_update.assert_called_once_with(meta)

    def test_switch_to_inference_calls_capture_and_switches(self, orchestrator):
        capture_fn = MagicMock()
        orchestrator.prepare_for_inference = MagicMock()

        orchestrator.switch_to_inference(
            global_step=10,
            capture_stats_fn=capture_fn,
        )

        capture_fn.assert_called_once_with()
        orchestrator.prepare_for_inference.assert_called_once_with()

    def test_switch_to_inference_without_capture_fn(self, orchestrator):
        orchestrator.prepare_for_inference = MagicMock()

        orchestrator.switch_to_inference(global_step=10)

        orchestrator.prepare_for_inference.assert_called_once_with()

    def test_finalize_onloads_training_engine_if_offloaded(
        self, orchestrator, mock_train_engine
    ):
        orchestrator._train_on_gpu = False

        orchestrator.finalize()

        mock_train_engine.onload.assert_called_once()
        assert orchestrator._train_on_gpu is True

    def test_finalize_noop_if_already_on_gpu(self, orchestrator, mock_train_engine):
        orchestrator._train_on_gpu = True

        orchestrator.finalize()

        mock_train_engine.onload.assert_not_called()


class TestTrainControllerColocatedInterfaces:
    def test_offload_updates_state_and_dispatches(self):
        controller = TrainController.__new__(TrainController)
        controller._custom_function_call = MagicMock()
        controller._colocated_orch = None
        controller.is_offload = False

        controller.offload()

        controller._custom_function_call.assert_called_once_with("offload")
        assert controller.is_offload is True

    def test_onload_updates_state_and_dispatches(self):
        controller = TrainController.__new__(TrainController)
        controller._custom_function_call = MagicMock()
        controller._colocated_orch = None
        controller.is_offload = True

        controller.onload()

        controller._custom_function_call.assert_called_once_with("onload")
        assert controller.is_offload is False

    def test_prepare_batch_context_is_noop(self):
        controller = TrainController.__new__(TrainController)
        controller._colocated_orch = None

        with controller.prepare_batch_context(global_step=3):
            pass

    def test_register_colocated_peer_forwards_pre_offloaded_state(self):
        controller = TrainController.__new__(TrainController)
        inf_engine = MagicMock()

        with patch("areal.infra.colocated.ColocatedOrchestrator") as mock_orch_cls:
            controller.register_colocated_peer(
                inf_engine,
                train_pre_offloaded=True,
            )

        mock_orch_cls.assert_called_once_with(
            train_engine=controller,
            inf_engine=inf_engine,
            train_pre_offloaded=True,
        )
        assert controller._colocated_orch is mock_orch_cls.return_value


class TestRolloutControllerColocatedInterfaces:
    def test_sync_weights_from_disk_uses_run_async_task(self):
        controller = RolloutController.__new__(RolloutController)
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v2")

        with patch(
            "areal.infra.controller.rollout_controller.run_async_task"
        ) as mock_run_async_task:
            controller.sync_weights_from_disk(meta)

        mock_run_async_task.assert_called_once_with(
            controller.update_weights_from_disk, meta
        )

    def test_pause_generation_and_continue_generation_use_run_async_task(self):
        controller = RolloutController.__new__(RolloutController)

        with patch(
            "areal.infra.controller.rollout_controller.run_async_task"
        ) as mock_run_async_task:
            controller.pause_generation()
            controller.continue_generation()

        assert mock_run_async_task.call_count == 2
        assert (
            mock_run_async_task.call_args_list[0].args[0].__name__
            == "_pause_generation_async"
        )
        assert (
            mock_run_async_task.call_args_list[1].args[0].__name__
            == "_continue_generation_async"
        )

    def test_offload_and_onload_delegate_to_collective_rpc(self):
        controller = RolloutController.__new__(RolloutController)
        controller._collective_rpc = MagicMock()

        controller.offload()
        controller.onload(tags=["lora"])

        assert controller._collective_rpc.call_args_list == [
            (("offload",), {"http_timeout": 60.0}),
            (("onload",), {"tags": ["lora"], "http_timeout": 60.0}),
        ]

    def test_update_weights_from_disk_does_not_mutate_original_meta(self):
        controller = RolloutController.__new__(RolloutController)
        controller._collective_rpc_async = AsyncMock()

        temp_dir = Path(tempfile.mkdtemp(prefix="areal-colocated-test-"))
        try:
            meta = WeightUpdateMeta(
                type="disk",
                path=str(temp_dir),
                clear_checkpoint_after_load=True,
            )

            asyncio.run(controller.update_weights_from_disk(meta))

            assert meta.clear_checkpoint_after_load is True
            assert not temp_dir.exists()
            await_args = controller._collective_rpc_async.await_args
            assert await_args is not None
            sent_meta = await_args.kwargs["meta"]
            assert sent_meta.clear_checkpoint_after_load is False
            assert sent_meta.path == meta.path
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_build_create_worker_kwargs_clears_tms_env_for_actor_colocated_fork(self):
        controller = RolloutController.__new__(RolloutController)
        job = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=True,
            )
        )

        assert controller._build_create_worker_kwargs(job) == {
            "fork_unset_env_keys": [
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ]
        }

    @pytest.mark.parametrize(
        "strategy",
        [
            SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=False,
            ),
            SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="critic",
                fork=True,
            ),
            SchedulingStrategy(type=SchedulingStrategyType.separation),
        ],
    )
    def test_build_create_worker_kwargs_skips_non_actor_fork_colocation(self, strategy):
        controller = RolloutController.__new__(RolloutController)
        job = SimpleNamespace(scheduling_strategy=strategy)

        assert controller._build_create_worker_kwargs(job) == {}


class TestLocalSchedulerForkEnvPatch:
    def test_apply_env_patch_unsets_and_overrides(self):
        patched = _apply_env_patch(
            {
                "LD_PRELOAD": "/tmp/libtms.so",
                "TMS_INIT_ENABLE": "1",
                "KEEP": "yes",
            },
            env_overrides={"NEW_KEY": "value"},
            unset_env_keys=["LD_PRELOAD", "TMS_INIT_ENABLE", "MISSING_KEY"],
        )

        assert patched == {
            "KEEP": "yes",
            "NEW_KEY": "value",
        }

    def test_create_workers_forwards_fork_env_cleanup(self):
        scheduler = LocalScheduler.__new__(LocalScheduler)
        scheduler._workers = cast(Any, {"actor": [object()]})
        scheduler._colocated_roles = {}
        scheduler._prepare_worker_specs = MagicMock(return_value=[SimpleNamespace()])
        scheduler.fork_workers = MagicMock(return_value=["rollout/0"])

        job = SimpleNamespace(
            role="rollout",
            replicas=1,
            tasks=[SimpleNamespace()],
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
                fork=True,
            ),
        )

        worker_ids = scheduler.create_workers(
            job=job,
            fork_unset_env_keys=[
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ],
        )

        assert worker_ids == ["rollout/0"]
        scheduler.fork_workers.assert_called_once_with(
            "rollout",
            "actor",
            env_overrides=None,
            unset_env_keys=[
                "LD_PRELOAD",
                "TMS_INIT_ENABLE",
                "TMS_INIT_ENABLE_CPU_BACKUP",
            ],
        )
        assert scheduler._colocated_roles["rollout"] == "actor"


def _make_validation_trainer(
    *,
    colocated: bool = True,
    weight_update_mode: str = "disk",
) -> Any:
    trainer = cast(Any, PPOTrainer.__new__(PPOTrainer))
    trainer.rollout_alloc = SimpleNamespace(backend="sglang")
    scheduling_strategy = SchedulingStrategy(
        type=(
            SchedulingStrategyType.colocation
            if colocated
            else SchedulingStrategyType.separation
        ),
        target="actor" if colocated else None,
    )
    trainer._colocated = colocated
    trainer.config = SimpleNamespace(
        enable_offload=False,
        actor=SimpleNamespace(
            kl_ctl=0,
            weight_update_mode=weight_update_mode,
            scheduling_spec=[SimpleNamespace(env_vars={})],
        ),
        rollout=SimpleNamespace(
            return_routed_experts=False,
            openai=None,
            scheduling_strategy=scheduling_strategy,
            scheduling_spec=[SimpleNamespace(env_vars={})],
        ),
        critic=None,
        ref=None,
        teacher=None,
        cluster=SimpleNamespace(n_nodes=1),
        experiment_name="gsm8k-grpo-colocated",
        trial_name="trial0",
    )
    return trainer


class TestPPOTrainerColocatedScheduling:
    def test_is_colocated_rollout_detects_actor_colocation(self):
        rollout_cfg = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="actor",
            )
        )

        assert cast(Any, PPOTrainer)._is_colocated_rollout(rollout_cfg) is True

    def test_is_colocated_rollout_rejects_other_topologies(self):
        rollout_cfg = SimpleNamespace(
            scheduling_strategy=SchedulingStrategy(
                type=SchedulingStrategyType.colocation,
                target="critic",
            )
        )

        assert cast(Any, PPOTrainer)._is_colocated_rollout(rollout_cfg) is False

    def test_validate_cfg_allows_single_controller(self):
        trainer = _make_validation_trainer()

        with patch("areal.trainer.rl_trainer.is_single_controller", return_value=True):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_multi_node(self):
        trainer = _make_validation_trainer()
        trainer.config.cluster.n_nodes = 2

        with pytest.raises(ValueError, match="single-node runs"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_non_disk_weight_update(self):
        trainer = _make_validation_trainer(weight_update_mode="xccl")

        with pytest.raises(ValueError, match="weight_update_mode='disk'"):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_rejects_missing_train_dataset(self):
        trainer = _make_validation_trainer()

        with pytest.raises(ValueError, match="requires a train_dataset"):
            trainer._validate_cfg(train_dataset=None)

    def test_validate_cfg_rejects_online_mode(self):
        trainer = _make_validation_trainer()
        trainer.config.rollout.openai = SimpleNamespace(mode="online")

        with pytest.raises(ValueError, match="rollout.openai.mode='online'"):
            trainer._validate_cfg(train_dataset=object())

    @pytest.mark.parametrize(
        ("mutate", "expected_error"),
        [
            (
                lambda trainer: setattr(trainer.config, "critic", object()),
                "critic is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "ref", object()),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config.actor, "kl_ctl", 0.1),
                "ref/kl_ctl is not supported",
            ),
            (
                lambda trainer: setattr(trainer.config, "teacher", object()),
                "teacher is not supported",
            ),
        ],
    )
    def test_validate_cfg_rejects_non_actor_only_components(
        self, mutate, expected_error
    ):
        trainer = _make_validation_trainer()
        mutate(trainer)

        with pytest.raises(ValueError, match=expected_error):
            trainer._validate_cfg(train_dataset=object())

    def test_validate_cfg_skips_colocated_restrictions_for_standard_mode(self):
        trainer = _make_validation_trainer(colocated=False)
        trainer.config.cluster.n_nodes = 4
        trainer.config.rollout.openai = SimpleNamespace(mode="online")
        trainer.config.critic = object()
        trainer.config.ref = object()
        trainer.config.teacher = object()

        trainer._validate_cfg(train_dataset=None)

    def test_amend_xccl_weight_update_envvar_injects_tms_for_colocated_controller(self):
        trainer = _make_validation_trainer()
        trainer.rollout_alloc = SimpleNamespace(backend="vllm")

        with (
            patch("areal.trainer.rl_trainer.is_single_controller", return_value=True),
            patch(
                "areal.trainer.rl_trainer.get_tms_env_vars",
                return_value={"LD_PRELOAD": "/tmp/libtms.so", "TMS_INIT_ENABLE": "1"},
            ),
        ):
            trainer._amend_xccl_weight_update_envvar()

        assert (
            trainer.config.actor.scheduling_spec[0].env_vars["LD_PRELOAD"]
            == "/tmp/libtms.so"
        )
        assert "LD_PRELOAD" not in trainer.config.rollout.scheduling_spec[0].env_vars

    def test_prepare_inference_phase_switches_and_sets_version(self):
        trainer = _make_validation_trainer()
        trainer.actor = MagicMock()
        trainer.actor.is_colocated = True

        trainer._prepare_inference_phase(global_step=11, version=12)

        trainer.actor.switch_to_inference.assert_called_once_with(
            global_step=11,
            capture_stats_fn=trainer._capture_train_stats_snapshot,
        )

    def test_publish_rollout_weights_pauses_and_sets_version_in_standard_mode(self):
        trainer = _make_validation_trainer(colocated=False)
        trainer.weight_update_meta = WeightUpdateMeta(
            type="disk", path="/tmp/weight_update_v0"
        )
        trainer.rollout = MagicMock()
        trainer.actor = MagicMock()
        trainer.actor.is_colocated = False
        trainer._set_rollout_version = MagicMock()

        new_version = trainer._publish_rollout_weights(global_step=4)

        assert new_version == 5
        trainer.rollout.pause.assert_called_once_with()
        trainer.actor.update_weights.assert_called_once_with(
            trainer.weight_update_meta.with_version(5)
        )
        trainer._set_rollout_version.assert_called_once_with(5)

    def test_publish_rollout_weights_keeps_colocated_switch_deferred(self):
        trainer = _make_validation_trainer()
        trainer.weight_update_meta = WeightUpdateMeta(
            type="disk", path="/tmp/weight_update_v0"
        )
        trainer.rollout = MagicMock()
        trainer.actor = MagicMock()
        trainer.actor.is_colocated = True
        trainer._set_rollout_version = MagicMock()

        new_version = trainer._publish_rollout_weights(global_step=6)

        assert new_version == 7
        trainer.rollout.pause.assert_not_called()
        trainer.actor.publish_colocated_weights.assert_called_once_with(
            trainer.weight_update_meta.with_version(7)
        )
        trainer._set_rollout_version.assert_called_once_with(7)

    def test_internal_stage_weight_update_dispatches_to_workers(self):
        controller = TrainController.__new__(TrainController)
        controller._colocated_orch = None
        controller._check_rollout_engine_connected = MagicMock()
        controller._custom_function_call = MagicMock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v7", version=7)

        controller._stage_weight_update(meta)

        controller._check_rollout_engine_connected.assert_called_once()
        controller._custom_function_call.assert_called_once_with(
            "_stage_weight_update", meta=meta
        )


class TestFSDPEngineStagedWeightUpdate:
    def test_register_colocated_peer_forwards_pre_offloaded_state(self):
        engine = cast(Any, FSDPEngine.__new__(FSDPEngine))
        inf_engine = MagicMock()

        with patch("areal.infra.colocated.ColocatedOrchestrator") as mock_orch_cls:
            engine.register_colocated_peer(
                inf_engine,
                train_pre_offloaded=True,
            )

        mock_orch_cls.assert_called_once_with(
            train_engine=engine,
            inf_engine=inf_engine,
            train_pre_offloaded=True,
        )
        assert engine._colocated_orch is mock_orch_cls.return_value

    def test_publish_disk_weight_update_ready_uses_engine_version(self):
        engine = cast(Any, FSDPEngine.__new__(FSDPEngine))
        engine.config = SimpleNamespace(
            experiment_name="gsm8k-grpo-colocated", trial_name="trial0"
        )
        engine.get_version = MagicMock(return_value=3)

        with patch("areal.engine.fsdp_engine.name_resolve.add") as mock_add:
            engine._publish_disk_weight_update_ready()

        mock_add.assert_called_once_with(
            names.update_weights_from_disk(
                "gsm8k-grpo-colocated",
                "trial0",
                3,
            ),
            mock_add.call_args.args[1],
            keepalive_ttl=120,
        )

    def test_stage_weight_update_from_tensor_only_streams_weights(self):
        engine = cast(Any, FSDPEngine.__new__(FSDPEngine))
        engine._initialized = True
        engine._cpu_group = object()
        engine.rollout_engine = MagicMock()
        engine._send_weights_from_tensor = MagicMock()
        meta = WeightUpdateMeta(type="tensor", weight_chunked_mem_mb=16)

        with (
            patch("areal.engine.fsdp_engine.current_platform.synchronize") as mock_sync,
            patch("areal.engine.fsdp_engine.dist.barrier") as mock_barrier,
        ):
            engine._stage_weight_update_from_tensor(meta)

        engine._send_weights_from_tensor.assert_called_once_with(meta)
        engine.rollout_engine.pause_generation.assert_not_called()
        engine.rollout_engine.continue_generation.assert_not_called()
        mock_sync.assert_called_once_with()
        mock_barrier.assert_called_once_with(group=engine.cpu_group)

    def test_update_weights_from_tensor_manages_rollout_pause_resume(self):
        engine = cast(Any, FSDPEngine.__new__(FSDPEngine))
        engine._initialized = True
        engine._cpu_group = object()
        engine.rollout_engine = MagicMock()
        engine._send_weights_from_tensor = MagicMock()
        meta = WeightUpdateMeta(type="tensor", weight_chunked_mem_mb=16)

        with (
            patch("areal.engine.fsdp_engine.dist.get_rank", return_value=0),
            patch("areal.engine.fsdp_engine.current_platform.synchronize") as mock_sync,
            patch("areal.engine.fsdp_engine.dist.barrier") as mock_barrier,
        ):
            engine._update_weights_from_tensor(meta)

        engine.rollout_engine.pause_generation.assert_called_once_with()
        engine._send_weights_from_tensor.assert_called_once_with(meta)
        engine.rollout_engine.continue_generation.assert_called_once_with()
        mock_sync.assert_called_once_with()
        assert mock_barrier.call_count == 3

    def test_send_weights_from_tensor_chunks_only_on_main_rank(self):
        engine = cast(Any, FSDPEngine.__new__(FSDPEngine))
        engine.rollout_engine = MagicMock()
        engine.rollout_engine.update_weights_from_tensor.return_value = MagicMock()
        engine._get_full_tensor = MagicMock(side_effect=lambda tensor: tensor)

        t0 = torch.ones(200_000, dtype=torch.float32)
        t1 = torch.ones(200_000, dtype=torch.float32)
        t2 = torch.ones(16, dtype=torch.float32)
        engine._get_lora_or_full_param_iterator = MagicMock(
            return_value=iter([
                ("w0", t0),
                ("w1", t1),
                ("w2", t2),
            ])
        )
        meta = WeightUpdateMeta(type="tensor", weight_chunked_mem_mb=1)

        with patch("areal.engine.fsdp_engine.dist.get_rank", return_value=0):
            engine._send_weights_from_tensor(meta)

        update_calls = engine.rollout_engine.update_weights_from_tensor.call_args_list
        assert len(update_calls) == 2
        assert [name for name, _ in update_calls[0].args[0]] == ["w0"]
        assert [name for name, _ in update_calls[1].args[0]] == ["w1", "w2"]


class TestRemoteInfEngineDiskWeightSync:
    def test_update_weights_from_disk_uses_engine_version_for_rendezvous(self):
        engine = cast(Any, RemoteInfEngine.__new__(RemoteInfEngine))
        engine.backend = MagicMock()
        engine.config = SimpleNamespace(
            experiment_name="exp",
            trial_name="trial",
            request_retries=2,
            request_timeout=30.0,
        )
        engine.addresses = ["127.0.0.1:8000"]
        engine.get_version = MagicMock(return_value=0)
        engine.logger = MagicMock()

        meta = WeightUpdateMeta(type="disk", path="/tmp/weight_update_v5", version=5)
        fake_future = MagicMock()

        with patch("areal.infra.remote_inf_engine.get_executor") as mock_get_executor:
            mock_get_executor.return_value.submit.return_value = fake_future
            engine.update_weights_from_disk(meta)

        submit_args = mock_get_executor.return_value.submit.call_args.args
        assert submit_args[4] == 0
        fake_future.add_done_callback.assert_called_once()
