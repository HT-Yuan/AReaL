from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from areal.api import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.cli_args import RecoverConfig
from areal.api.engine_api import TrainEngine
from areal.infra.colocated import ColocatedOrchestrator
from areal.infra.controller.train_controller import TrainController
from areal.utils.recover import RecoverHandler, RecoverInfo


class DummyTrainEngine(TrainEngine):
    def connect_engine(self, engine, meta):
        self.connected = (engine, meta)

    def update_weights(self, meta):
        self.updated_meta = meta

    def set_version(self, version: int):
        self.version = version

    def get_version(self) -> int:
        return getattr(self, "version", -1)


DummyTrainEngine.__abstractmethods__ = frozenset()


def test_train_engine_recover_inference_engine_non_colocated_uses_standard_sync():
    engine = DummyTrainEngine()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=11)

    engine.recover_inference_engine(
        inference_engine,
        meta,
        set_version_fn=set_version_fn,
    )

    assert engine.connected == (inference_engine, meta)
    assert engine.updated_meta == meta
    inference_engine.pause.assert_called_once_with()
    inference_engine.resume.assert_called_once_with()
    set_version_fn.assert_called_once_with(11)
    inference_engine.set_version.assert_not_called()


def test_train_engine_recover_inference_engine_colocated_delegates_to_orchestrator():
    engine = DummyTrainEngine()
    engine._colocated_orch = MagicMock()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=13)

    engine.recover_inference_engine(
        inference_engine,
        meta,
        set_version_fn=set_version_fn,
    )

    assert engine.connected == (inference_engine, meta)
    engine._colocated_orch.recover_inference_engine.assert_called_once_with(
        meta,
        set_version_fn=set_version_fn,
    )
    inference_engine.pause.assert_not_called()
    inference_engine.resume.assert_not_called()


def test_colocated_orchestrator_recover_inference_engine_switches_and_flushes():
    train_engine = MagicMock()
    inf_engine = MagicMock()
    orchestrator = ColocatedOrchestrator(
        train_engine=train_engine,
        inf_engine=inf_engine,
        train_pre_offloaded=True,
    )
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=17)

    orchestrator.recover_inference_engine(meta, set_version_fn=set_version_fn)

    train_engine.onload.assert_called_once_with()
    train_engine.offload.assert_called_once_with()
    train_engine._stage_weight_update.assert_called_once_with(meta)
    inf_engine.pause.assert_called_once_with()
    inf_engine.pause_generation.assert_called_once_with()
    inf_engine.offload.assert_not_called()
    inf_engine.onload.assert_not_called()
    inf_engine.sync_weights_from_disk.assert_called_once_with(meta)
    inf_engine.continue_generation.assert_called_once_with()
    set_version_fn.assert_called_once_with(17)


def test_train_controller_recover_inference_engine_delegates_to_orchestrator():
    controller = TrainController.__new__(TrainController)
    controller._colocated_orch = MagicMock()
    controller.connect_engine = MagicMock()
    controller.update_weights = MagicMock()
    controller.set_version = MagicMock()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=19)

    TrainController.recover_inference_engine(
        controller,
        inference_engine,
        meta,
        set_version_fn=set_version_fn,
    )

    controller.connect_engine.assert_called_once_with(inference_engine, meta)
    controller._colocated_orch.recover_inference_engine.assert_called_once_with(
        meta,
        set_version_fn=set_version_fn,
    )
    controller.update_weights.assert_not_called()
    inference_engine.pause.assert_not_called()
    inference_engine.resume.assert_not_called()


def test_recover_handler_load_delegates_inference_recovery_to_engine():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RecoverConfig(
            experiment_name="test_exp",
            trial_name="test_trial",
            fileroot=tmpdir,
            mode="on",
        )
        handler = RecoverHandler(
            config,
            FinetuneSpec(total_train_epochs=1, dataset_size=1, train_batch_size=1),
        )
        recover_info = RecoverInfo(
            last_step_info=StepInfo(
                global_step=4,
                epoch=0,
                epoch_step=0,
                steps_per_epoch=1,
            ),
            saver_info={"saver": 1},
            evaluator_info={"evaluator": 1},
            stats_logger_info={"stats": 1},
            dataloader_info={"loader": 1},
            checkpoint_info={"freq": 1},
        )
        saver = MagicMock()
        evaluator = MagicMock()
        stats_logger = MagicMock()
        dataloader = MagicMock()
        engine = MagicMock()
        inference_engine = MagicMock()
        set_version_fn = MagicMock()
        meta = WeightUpdateMeta(type="disk", path="/tmp/recover")

        with (
            patch(
                "areal.utils.recover.RecoverInfo.load",
                return_value=recover_info,
            ),
            patch.object(handler, "_load_checkpoint") as mock_load_checkpoint,
        ):
            result = handler.load(
                {"default": engine},
                saver,
                evaluator,
                stats_logger,
                dataloader,
                inference_engine=inference_engine,
                weight_update_meta=meta,
                set_version_fn=set_version_fn,
            )

        assert result is recover_info
        saver.load_state_dict.assert_called_once_with({"saver": 1})
        evaluator.load_state_dict.assert_called_once_with({"evaluator": 1})
        stats_logger.load_state_dict.assert_called_once_with({"stats": 1})
        dataloader.load_state_dict.assert_called_once_with({"loader": 1})
        mock_load_checkpoint.assert_called_once_with(engine, name="default")
        engine.recover_inference_engine.assert_called_once_with(
            inference_engine,
            meta.with_version(5),
            set_version_fn=set_version_fn,
        )
