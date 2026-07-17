from pathlib import Path

import torch

from nanoquant.config.schema import ActivationGpuCacheMode, CalibrationMethod, ExecutorKind
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    resident_request_from_config,
)
from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(8)


def test_experiment008_is_guarded_12b_compression_quality_proof(tmp_path: Path) -> None:
    config = _DEFINITION.config
    experiment = _DEFINITION.workflow

    assert config.model.source == "unsloth/gemma-3-12b-it"
    assert config.model.revision == "9478e665381f42974aa06177b019352fb6291876"
    assert config.intent.experiment_number == 8
    assert config.intent.name == "008-compress-and-benchmark-gemma-3-12b-it"
    assert "unsloth/gemma-3-12b-it-GGUF" in str(config.intent.baseline_run)
    assert config.runtime.executor is ExecutorKind.CPU_OFFLOAD
    assert config.calibration.method is CalibrationMethod.FORWARD_ONLY
    assert config.runtime.activations.gpu_cache is ActivationGpuCacheMode.AUTO
    assert config.runtime.activations.gpu_reserve_gib == 4.0
    assert config.runtime.block_forward_batch_size == 1
    assert config.block_tuning.microbatch_size == 1
    assert config.block_tuning.non_factorized.loop.batch_size == 8
    assert config.block_tuning.factorized.loop.batch_size == 8
    assert config.block_tuning.post_block_refit.batch_size == 8
    assert not config.evaluation.inline_quality
    assert not config.distillation.enabled
    assert experiment.expected_blocks == 48
    assert experiment.large_model_guards
    assert not experiment.restore_completed_blocks
    assert experiment.maximum_wddm_shared_gib == 0.75
    assert experiment.export.gguf_output == Path("outputs/008/gemma-3-12b-it-nanoquant.gguf")

    tokens = torch.zeros((config.calibration.sample_count, 8), dtype=torch.long)
    inputs = ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "run",
        registry_root=tmp_path / "registry",
        token_ids=tokens,
        quality_token_ids=None,
        launcher_path=tmp_path / "experiments/008-compress-and-benchmark-gemma-3-12b-it.py",
        pad_token_id=None,
    )
    request = resident_request_from_config(
        config,
        inputs,
        ResidentExecutionOptions(restore_completed_blocks=False),
    )
    assert request.executor is ExecutorKind.CPU_OFFLOAD
    assert not request.restore_completed_blocks
    assert not request.evaluate_inline_quality
