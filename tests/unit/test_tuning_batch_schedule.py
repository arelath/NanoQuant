from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import nanoquant.resident_quantization as resident
from nanoquant.config.codec import config_hash, semantic_hash, to_dict
from nanoquant.config.schema import FactorizedOptimizerConfig, ModelConfig, RunConfig, TuningBatchScheduleConfig
from nanoquant.config.validation import ValidationPhase, validate
from nanoquant.resident_quantization import (
    ResidentQuantizationRequest,
    _block_tuning_batches,
    _resolve_tuning_batch_schedule,
    _tuning_batch_candidates,
)


def _request(tmp_path: Path, **changes: Any) -> ResidentQuantizationRequest:
    request = ResidentQuantizationRequest(
        tmp_path / "snapshot",
        tmp_path / "run",
        "fixture/model",
        "revision",
        ((1, 2), (3, 4)),
        device="cpu",
        factorized_tuning_batch_size=1,
        nonfactorized_tuning_batch_size=4,
        post_block_refit_batch_size=1,
        tuning_microbatch_size=1,
        post_block_refit_microbatch_size=1,
    )
    return replace(request, **changes)


class _Events:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, _component: str, _level: str, name: str, **fields: object) -> None:
        self.events.append((name, fields))


def test_blocks_before_the_schedule_keep_their_committed_sizes(tmp_path: Path) -> None:
    schedule = TuningBatchScheduleConfig(enabled=True, start_block=2)
    request = _request(tmp_path, tuning_batch_schedule=schedule, selected_tuning_batch_size=8)

    before = _block_tuning_batches(request, 1)
    after = _block_tuning_batches(request, 2)

    assert (before.factorized, before.nonfactorized, before.post_block_refit) == (1, 4, 1)
    assert (before.tuning_microbatch, before.refit_microbatch, before.scheduled) == (1, 1, False)
    assert (after.factorized, after.nonfactorized, after.post_block_refit) == (8, 8, 8)
    assert (after.tuning_microbatch, after.refit_microbatch, after.scheduled) == (8, 8, True)
    assert _block_tuning_batches(_request(tmp_path), 5).factorized == 1


def test_scheduled_block_requires_a_resolved_batch(tmp_path: Path) -> None:
    request = _request(tmp_path, tuning_batch_schedule=TuningBatchScheduleConfig(enabled=True))
    with pytest.raises(RuntimeError, match="not resolved"):
        _block_tuning_batches(request, 0)


def test_batch_candidates_descend_from_the_sample_capped_maximum() -> None:
    assert _tuning_batch_candidates(8, 256) == (8, 4, 2, 1)
    assert _tuning_batch_candidates(8, 3) == (3, 2, 1)
    assert _tuning_batch_candidates(6, 100) == (6, 4, 2, 1)
    assert _tuning_batch_candidates(1, 100) == (1,)


def test_default_schedule_preserves_existing_config_and_identity_hashes(tmp_path: Path) -> None:
    config = RunConfig(ModelConfig("x"))
    scheduled = replace(config, block_tuning=replace(
        config.block_tuning, batch_schedule=TuningBatchScheduleConfig(enabled=True, start_block=3),
    ))

    assert config_hash(scheduled) != config_hash(config)
    base = _request(tmp_path)
    assert resident._resident_config_hash(replace(base, selected_tuning_batch_size=4)) == (
        resident._resident_config_hash(base)
    )
    assert resident._resident_config_hash(
        replace(base, tuning_batch_schedule=TuningBatchScheduleConfig(enabled=True, start_block=3))
    ) != resident._resident_config_hash(base)
    manifest = resident._resident_manifest_config(replace(base, selected_tuning_batch_size=4), "test")
    assert "tuning_batch_schedule" not in manifest and "selected_tuning_batch_size" not in manifest
    assert semantic_hash(manifest) == semantic_hash(resident._resident_manifest_config(base, "test"))


@pytest.mark.parametrize(
    ("schedule", "code"),
    [
        (TuningBatchScheduleConfig(enabled=True, maximum_batch_size=0), "CFG256"),
        (TuningBatchScheduleConfig(enabled=True, memory_safety_factor=0.5), "CFG256"),
        (TuningBatchScheduleConfig(enabled=True, start_block=0, continuation_journal_sha256="a" * 64), "CFG257"),
        (TuningBatchScheduleConfig(enabled=False, start_block=4, continuation_journal_sha256="a" * 64), "CFG257"),
    ],
)
def test_invalid_schedules_are_rejected(schedule: TuningBatchScheduleConfig, code: str) -> None:
    config = RunConfig(ModelConfig("x"))
    config = replace(config, block_tuning=replace(config.block_tuning, batch_schedule=schedule))
    assert code in {issue.code for issue in validate(config, ValidationPhase.RESOLVED)}


def test_schedule_selection_is_persisted_and_reused(tmp_path: Path) -> None:
    schedule = TuningBatchScheduleConfig(enabled=True, start_block=1, maximum_batch_size=8)
    request = _request(tmp_path, tuning_batch_schedule=schedule)
    inputs = torch.zeros(5, 2, 3)
    events = _Events()
    arguments = (None, None, None, None, inputs, {}, events)

    assert _resolve_tuning_batch_schedule(request, *arguments) == 5  # type: ignore[arg-type]
    stored = json.loads((request.output / "state/tuning-batch-selection.json").read_text(encoding="utf-8"))
    assert stored["selected_batch_size"] == 5 and stored["method"] == "unmeasured_non_cuda_device"
    assert _resolve_tuning_batch_schedule(request, *arguments) == 5  # type: ignore[arg-type]
    assert [name for name, _fields in events.events] == ["tuning_batch.selected", "tuning_batch.reused"]

    changed = replace(request, tuning_batch_schedule=replace(schedule, maximum_batch_size=4))
    with pytest.raises(ValueError, match="does not match this schedule"):
        _resolve_tuning_batch_schedule(changed, *arguments)  # type: ignore[arg-type]
    assert _resolve_tuning_batch_schedule(_request(tmp_path), *arguments) is None  # type: ignore[arg-type]


def test_batch_schedule_continuation_removes_only_its_own_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = FactorizedOptimizerConfig("fused_adamw_scales_fp32", 19, ("mlp.gate_proj",))
    schedule = TuningBatchScheduleConfig(enabled=True, start_block=11, continuation_journal_sha256="b" * 64)
    config = RunConfig(ModelConfig("x"))
    config = replace(config, block_tuning=replace(
        config.block_tuning,
        factorized=replace(config.block_tuning.factorized, optimizer=optimizer),
        batch_schedule=schedule,
    ))
    request = _request(
        tmp_path, run_config=config, factorized_tuning_optimizer=optimizer, tuning_batch_schedule=schedule,
    )
    request.output.mkdir(parents=True)
    stored = resident._resident_manifest(replace(request, run_config=None), "resident-quantization")
    (request.output / "manifest.json").write_text(json.dumps(to_dict(stored)), encoding="utf-8")
    captured: dict[str, Any] = {}
    sources: list[ResidentQuantizationRequest] = []
    original_manifest = resident._resident_manifest

    def manifest(candidate: ResidentQuantizationRequest, component: str) -> Any:
        sources.append(candidate)
        return original_manifest(replace(candidate, run_config=None), component)

    monkeypatch.setattr(resident, "_resident_manifest", manifest)
    monkeypatch.setattr(resident, "import_optimizer_prefix", lambda output, **fields: captured.update(fields))

    resident._prepare_optimizer_continuation(request, SimpleNamespace(
        config_hash="target", resolved_config={}, launcher=stored.launcher, environment=stored.environment,
    ))  # type: ignore[arg-type]

    assert captured["start_block"] == 11
    assert captured["source_journal_sha256"] == "b" * 64
    assert captured["numerical_payload"] == "unchanged; computed with the source tuning batch sizes"
    source = sources[-1]
    assert source.tuning_batch_schedule == TuningBatchScheduleConfig()
    assert source.factorized_tuning_optimizer == optimizer
    assert source.run_config is not None
    assert source.run_config.block_tuning.factorized.optimizer == optimizer
    assert source.run_config.block_tuning.batch_schedule == TuningBatchScheduleConfig()
    assert captured["source_config_hash"] != captured["target_config_hash"]

    both = replace(request, factorized_tuning_optimizer=replace(optimizer, continuation_journal_sha256="c" * 64))
    with pytest.raises(ValueError, match="cannot be imported together"):
        resident._prepare_optimizer_continuation(both, stored)
