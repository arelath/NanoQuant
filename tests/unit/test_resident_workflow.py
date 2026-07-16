from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from recipes.legacy.experiment018 import EXPERIMENT_018_CONFIG

import nanoquant.resident_quantization as resident
import nanoquant.resident_workflow as workflow
from nanoquant.config.codec import from_dict
from nanoquant.config.schema import DistillationLoss, RunConfig
from nanoquant.domain.runs import RunManifest, RunStatus
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest_from_resolved,
    launcher_provenance,
    transition,
)
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    distillation_request_from_config,
    execute_resident_workflow,
    resident_request_from_config,
    resolve_resident_experiment_inputs,
)


def _experiment018_config() -> RunConfig:
    return EXPERIMENT_018_CONFIG


def _inputs(tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.arange(256 * 8, dtype=torch.long).reshape(256, 8)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "runs" / "018",
        registry_root=tmp_path / "runs",
        token_ids=tokens,
        quality_token_ids=tokens[:1, :8],
        launcher_path=Path("experiments/recipes/legacy/experiment018.py"),
        pad_token_id=0,
    )


def test_experiment018_maps_every_hidden_resident_parity_semantic(tmp_path: Path) -> None:
    config = _experiment018_config()
    request = resident_request_from_config(config, _inputs(tmp_path))

    assert request.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert request.calibration_method == "online_fisher"
    assert request.calibration_batch_size == 1
    assert request.calibration_shrinkage == 0.6
    assert request.rank_retry.maximum_attempts == 3
    assert request.rank_retry.thresholds.raw_normalized_error == 0.5
    assert request.maximum_rank_layer_patterns == config.allocation.maximum_rank_layer_patterns
    assert request.nonfactorized_tuning_epochs == 0
    assert request.nonfactorized_tuning_epochs_by_layer == (8, 4, 3, 2, 2, 2, 2)
    assert request.factorized_tuning_epochs == 8
    assert request.post_block_refit_epochs == 2
    assert request.tuning_microbatch_size == 8
    assert request.legacy_tuning_seed_reset
    assert not request.restore_best_tuning_state
    assert request.tuning_epoch_loss_mode == "legacy_training"
    assert request.activation_retention == "rolling"
    assert request.evaluate_inline_quality
    assert request.run_config is config
    assert request.defer_run_completion

    manifest = resident._resident_manifest(request, "resident-quantization")
    assert manifest.launcher.kind == "numbered_runfile"
    assert manifest.launcher.experiment_number == 18
    assert manifest.launcher.repository_relative_path == (
        "experiments/recipes/legacy/experiment018.py"
    )
    assert cast(dict[str, object], manifest.resolved_config)["canonical_run_config"]


def test_experiment018_maps_implicit_legacy_model_kd_defaults(tmp_path: Path) -> None:
    request = distillation_request_from_config(_experiment018_config(), _inputs(tmp_path))

    assert request.config.epochs == 8
    assert request.config.batch_size == 1
    assert request.config.learning_rate == 1e-5
    assert request.config.top_k == 64
    assert request.config.maximum_tokens_per_batch == 512
    assert request.config.gradient_checkpointing
    assert request.config.weight_decay == 0.0
    assert request.config.optimizer_version == "legacy-optimi-adamw-v1"
    assert request.config.sampling_version == "legacy-python-device-rng-v1"
    assert request.block_snapshot_samples == 4
    assert request.block_snapshot_tokens == 512


def test_resident_mapping_rejects_unimplemented_semantics(tmp_path: Path) -> None:
    config = _experiment018_config()
    unsupported = replace(config, distillation=replace(config.distillation, loss=DistillationLoss.FULL_KL))

    with pytest.raises(ValueError, match="only top_k is implemented"):
        resident_request_from_config(unsupported, _inputs(tmp_path))


def test_combined_workflow_runs_quantization_before_distillation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _experiment018_config()
    inputs = _inputs(tmp_path)
    calls: list[str] = []
    quantization_result = object()
    distillation_result = type(
        "DistillationResult",
        (),
        {"reference": type("Reference", (), {"artifact_id": "sha256-distillation"})()},
    )()

    def quantize(_request: Any) -> Any:
        calls.append("quantize")
        return quantization_result

    def distill(_request: Any) -> Any:
        calls.append("distill")
        return distillation_result

    monkeypatch.setattr("nanoquant.resident_workflow.run_resident_quantization", quantize)
    monkeypatch.setattr("nanoquant.resident_workflow.run_global_topk_distillation", distill)
    transitions = []
    monkeypatch.setattr(
        "nanoquant.resident_workflow._transition_workflow_manifest",
        lambda *_args, **kwargs: transitions.append(kwargs),
    )

    result = execute_resident_workflow(config, inputs, ResidentExecutionOptions())

    assert calls == ["quantize", "distill"]
    assert cast(Any, result.quantization) is quantization_result
    assert cast(Any, result.distillation) is distillation_result
    assert transitions == [{"artifact_id": "sha256-distillation"}]


def test_zero_argument_resolution_uses_pinned_snapshot_and_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    launcher = repository / "experiments" / "018-example.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = repository / "snapshot"
    snapshot.mkdir()
    config = _experiment018_config()
    config = replace(config, model=replace(config.model, source=str(snapshot)))
    tokens = torch.arange(256 * 16, dtype=torch.long).reshape(256, 16)
    observed: dict[str, object] = {}

    def load_calibration(path: Path, reference: object) -> Any:
        observed["path"] = path
        observed["reference"] = reference
        return type("Calibration", (), {"input_ids": tokens})()

    monkeypatch.setattr(workflow, "load_pinned_calibration", load_calibration)
    monkeypatch.setattr(
        workflow.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: type("Tokenizer", (), {"pad_token_id": 7})(),
    )

    resolved = resolve_resident_experiment_inputs(config, launcher_path=launcher)

    assert resolved.snapshot == snapshot
    assert resolved.output == repository / "runs" / config.intent.name
    assert resolved.registry_root == repository / "runs"
    assert torch.equal(cast(torch.Tensor, resolved.token_ids), tokens)
    assert torch.equal(cast(torch.Tensor, resolved.quality_token_ids), tokens[:1, :8])
    assert resolved.pad_token_id == 7
    assert observed["path"] == repository / "evidence/m3/experiment018-calibration"


def test_workflow_manifest_completes_only_with_global_tuning_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    directory = RunDirectory(inputs.output.parent, inputs.output.name)
    manifest = initial_manifest_from_resolved(
        "sha256:config",
        {},
        launcher_provenance(
            "experiments/recipes/legacy/experiment018.py",
            18,
        ),
        {},
    )
    directory.write_manifest(
        replace(transition(manifest, RunStatus.RUNNING), artifacts=("sha256-resident",))
    )

    workflow._transition_workflow_manifest(
        inputs,
        RunStatus.COMPLETED,
        artifact_id="sha256-distillation",
    )

    completed = from_dict(RunManifest, directory.read_manifest(), path="manifest")
    assert completed.status is RunStatus.COMPLETED
    assert completed.artifacts == ("sha256-resident", "sha256-distillation")
