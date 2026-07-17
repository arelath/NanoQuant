from pathlib import Path

import torch

from nanoquant.config.schema import ProfilingLevel
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    distillation_request_from_config,
    resident_request_from_config,
)
from tests.support.experiments import load_experiment

_DEFINITION = load_experiment(3)


def _inputs(tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.zeros((256, 2048), dtype=torch.long)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "run",
        registry_root=tmp_path / "registry",
        token_ids=tokens,
        quality_token_ids=tokens[:1, :8],
        launcher_path=tmp_path / "experiments/003-compress-and-benchmark-gemma-3-4b-it.py",
        pad_token_id=0,
    )


def test_experiment003_is_full_gemma4b_compression_quality_proof(tmp_path: Path) -> None:
    config = _DEFINITION.config
    experiment = _DEFINITION.workflow
    assert config.model.source == "google/gemma-3-4b-it"
    assert config.model.revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert config.intent.name == "003-compress-and-benchmark-gemma-3-4b-it"
    assert config.output.run_root == "evidence/003"
    assert config.runtime.block_forward_batch_size == 4
    assert config.block_tuning.non_factorized.loop.batch_size == 4
    assert config.block_tuning.factorized.loop.batch_size == 1
    assert config.block_tuning.microbatch_size == 1
    assert experiment.maximum_wddm_shared_gib == 0.75
    assert not config.evaluation.inline_quality
    assert not experiment.restore_completed_blocks
    assert experiment.quality_backend == "dense"
    assert experiment.export.gguf_output == Path("outputs/003/gemma-3-4b-it-nanoquant.gguf")
    assert experiment.quality_markdown_output == Path(
        "Results/003/003-compress-and-benchmark-gemma-3-4b-it-quality.md"
    )
    assert config.profiling.level is ProfilingLevel.MACRO
    assert config.profiling.cuda_timing
    assert config.profiling.memory_counters
    assert config.profiling.emit_span_events
    assert experiment.expected_blocks == 34
    assert len(experiment.task_names) == 6
    assert experiment.task_limit == 200

    options = ResidentExecutionOptions(maximum_wddm_shared_bytes=int(0.75 * 2**30))
    resident = resident_request_from_config(config, _inputs(tmp_path), options)
    distillation = distillation_request_from_config(config, _inputs(tmp_path), options)
    assert resident.maximum_wddm_shared_bytes == int(0.75 * 2**30)
    assert distillation.maximum_wddm_shared_bytes == int(0.75 * 2**30)
