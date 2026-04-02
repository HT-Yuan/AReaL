from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

from areal.api import FinetuneSpec, StepInfo, WeightUpdateMeta
from areal.api.cli_args import RecoverConfig
from areal.api.engine_api import TrainEngine
from areal.infra.controller.train_controller import TrainController
from areal.utils.recover import (
    RecoverHandler,
    RecoverInfo,
    _sync_recovered_inference_engine,
)


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


def test_sync_recovered_inference_engine_non_colocated_uses_standard_sync():
    engine = DummyTrainEngine()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=11)

    _sync_recovered_inference_engine(
        engine,
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


def test_sync_recovered_inference_engine_colocated_uses_orchestrator_runtime():
    engine = DummyTrainEngine()
    engine._colocated_orch = MagicMock()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=13)

    _sync_recovered_inference_engine(
        engine,
        inference_engine,
        meta,
        set_version_fn=set_version_fn,
    )

    assert engine.connected == (inference_engine, meta)
    engine._colocated_orch.prepare_for_training.assert_called_once_with()
    engine._colocated_orch.publish_weights.assert_called_once_with(meta)
    engine._colocated_orch.prepare_for_inference.assert_called_once_with()
    inference_engine.pause.assert_not_called()
    inference_engine.resume.assert_not_called()
    set_version_fn.assert_called_once_with(13)


def test_sync_recovered_inference_engine_supports_train_controller():
    controller = TrainController.__new__(TrainController)
    controller._colocated_orch = MagicMock()
    controller.connect_engine = MagicMock()
    controller.update_weights = MagicMock()
    controller.set_version = MagicMock()
    inference_engine = MagicMock()
    set_version_fn = MagicMock()
    meta = WeightUpdateMeta(type="disk", path="/tmp/recover", version=19)

    _sync_recovered_inference_engine(
        controller,
        inference_engine,
        meta,
        set_version_fn=set_version_fn,
    )

    controller.connect_engine.assert_called_once_with(inference_engine, meta)
    controller._colocated_orch.prepare_for_training.assert_called_once_with()
    controller._colocated_orch.publish_weights.assert_called_once_with(meta)
    controller._colocated_orch.prepare_for_inference.assert_called_once_with()
    controller.update_weights.assert_not_called()
    inference_engine.pause.assert_not_called()
    inference_engine.resume.assert_not_called()
    set_version_fn.assert_called_once_with(19)


def test_recover_handler_load_syncs_inference_via_recover_helper():
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
        engine._colocated_orch = None
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

        versioned_meta = meta.with_version(5)
        assert result is recover_info
        saver.load_state_dict.assert_called_once_with({"saver": 1})
        evaluator.load_state_dict.assert_called_once_with({"evaluator": 1})
        stats_logger.load_state_dict.assert_called_once_with({"stats": 1})
        dataloader.load_state_dict.assert_called_once_with({"loader": 1})
        mock_load_checkpoint.assert_called_once_with(engine, name="default")
        engine.connect_engine.assert_called_once_with(inference_engine, versioned_meta)
        engine.update_weights.assert_called_once_with(versioned_meta)
        inference_engine.pause.assert_called_once_with()
        inference_engine.resume.assert_called_once_with()
        set_version_fn.assert_called_once_with(5)
