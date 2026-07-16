import runpy
from pathlib import Path

import torch

from nanoquant.config.schema import ProfilingLevel
from nanoquant.recipes import EXPERIMENT_003, EXPERIMENT_003_CONFIG
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    distillation_request_from_config,
    resident_request_from_config,
)


def _inputs(tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.zeros((256, 2048), dtype=torch.long)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "run",
        registry_root=tmp_path / "registry",
        token_ids=tokens,
        quality_token_ids=tokens[:1, :8],
        launcher_path=tmp_path / "experiments/003.py",
        pad_token_id=0,
    )


def test_experiment003_is_full_gemma4b_compression_quality_proof(tmp_path: Path) -> None:
    config = EXPERIMENT_003_CONFIG
    assert config.model.source == "google/gemma-3-4b-it"
    assert config.model.revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert config.intent.name == "003-compress-and-benchmark-gemma-3-4b-it-v5"
    assert config.runtime.block_forward_batch_size == 4
    assert config.block_tuning.non_factorized.loop.batch_size == 4
    assert config.block_tuning.factorized.loop.batch_size == 1
    assert config.block_tuning.microbatch_size == 1
    assert EXPERIMENT_003.maximum_wddm_shared_gib == 0.75
    assert not config.evaluation.inline_quality
    assert not EXPERIMENT_003.restore_completed_blocks
    assert EXPERIMENT_003.quality_backend == "dense"
    assert config.profiling.level is ProfilingLevel.MACRO
    assert config.profiling.cuda_timing
    assert config.profiling.memory_counters
    assert config.profiling.emit_span_events
    assert EXPERIMENT_003.expected_blocks == 34
    assert len(EXPERIMENT_003.task_names) == 6
    assert EXPERIMENT_003.task_limit == 200

    options = ResidentExecutionOptions(maximum_wddm_shared_bytes=int(0.75 * 2**30))
    resident = resident_request_from_config(config, _inputs(tmp_path), options)
    distillation = distillation_request_from_config(config, _inputs(tmp_path), options)
    assert resident.maximum_wddm_shared_bytes == int(0.75 * 2**30)
    assert distillation.maximum_wddm_shared_bytes == int(0.75 * 2**30)


def test_003_runfile_imports_canonical_recipe_objects() -> None:
    namespace = runpy.run_path("experiments/003-compress-and-benchmark-gemma-3-4b-it.py")

    assert namespace["CONFIG"] is EXPERIMENT_003_CONFIG
    assert namespace["EXPERIMENT"] is EXPERIMENT_003
