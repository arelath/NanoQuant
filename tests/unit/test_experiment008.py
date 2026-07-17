import runpy
from pathlib import Path

import torch
from recipes import EXPERIMENT_008, EXPERIMENT_008_CONFIG
from recipes.experiment008 import (
    MODEL_REVISION,
    MODEL_SOURCE,
    REQUESTED_GGUF_FILENAME,
    REQUESTED_GGUF_REPOSITORY,
    REQUESTED_GGUF_REVISION,
)

from nanoquant.config.schema import ActivationGpuCacheMode, CalibrationMethod, ExecutorKind
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    resident_request_from_config,
)


def test_experiment008_is_guarded_12b_compression_quality_proof(tmp_path: Path) -> None:
    config = EXPERIMENT_008_CONFIG

    assert MODEL_SOURCE == "unsloth/gemma-3-12b-it"
    assert MODEL_REVISION == "9478e665381f42974aa06177b019352fb6291876"
    assert REQUESTED_GGUF_REPOSITORY == "unsloth/gemma-3-12b-it-GGUF"
    assert REQUESTED_GGUF_REVISION == "d15e4c7dc21dc55d56bf8549db57a71ad8a2a35d"
    assert REQUESTED_GGUF_FILENAME == "gemma-3-12b-it-BF16.gguf"
    assert config.model.source == MODEL_SOURCE
    assert config.model.revision == MODEL_REVISION
    assert config.intent.experiment_number == 8
    assert config.intent.name == "008-compress-and-benchmark-gemma-3-12b-it-forward-only-v4"
    assert REQUESTED_GGUF_REPOSITORY in str(config.intent.baseline_run)
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
    assert EXPERIMENT_008.expected_blocks == 48
    assert EXPERIMENT_008.large_model_guards
    assert not EXPERIMENT_008.restore_completed_blocks
    assert EXPERIMENT_008.maximum_wddm_shared_gib == 0.75
    assert EXPERIMENT_008.export.gguf_output == Path(
        "outputs/008-gemma-3-12b-it/gemma-3-12b-it-nanoquant.gguf"
    )

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


def test_experiment008_runfile_imports_canonical_recipe() -> None:
    namespace = runpy.run_path("experiments/008-compress-and-benchmark-gemma-3-12b-it.py")

    assert namespace["CONFIG"] is EXPERIMENT_008_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_008
