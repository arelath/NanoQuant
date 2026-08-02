from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import nanoquant.resident_quantization as resident
import nanoquant.resident_workflow as workflow
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    ActivationStorageConfig,
    CalibrationMethod,
    DistillationLoss,
    ExecutorKind,
    FactorizedLearningRates,
    ObjectiveKind,
    PostRefitCovarianceRefinementConfig,
    RunConfig,
)
from nanoquant.domain.models import ArtifactRef
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
    load_completed_resident_workflow,
    mass_floor_correction_request_from_config,
    resident_request_from_config,
    resolve_resident_experiment_inputs,
)
from tests.support.experiments import load_experiment


def _resident_config() -> RunConfig:
    return load_experiment(1).config


def _inputs(tmp_path: Path) -> ResolvedResidentInputs:
    tokens = torch.arange(256 * 8, dtype=torch.long).reshape(256, 8)
    return ResolvedResidentInputs(
        snapshot=tmp_path / "snapshot",
        output=tmp_path / "runs" / "001",
        registry_root=tmp_path / "runs",
        token_ids=tokens,
        quality_token_ids=tokens[:1, :8],
        launcher_path=Path("experiments/001-compress-gemma-3-1b-it.py"),
        pad_token_id=0,
    )


def test_resident_recipe_maps_every_hidden_parity_semantic(tmp_path: Path) -> None:
    config = _resident_config()
    request = resident_request_from_config(config, _inputs(tmp_path))

    assert request.revision == "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
    assert request.calibration_method == "online_fisher"
    assert request.calibration_batch_size == 1
    assert request.calibration_shrinkage == 0.6
    assert request.rank_retry.maximum_attempts == 3
    assert request.rank_retry.thresholds.raw_normalized_error == 0.5
    assert request.maximum_rank_layer_patterns == config.allocation.maximum_rank_layer_patterns
    assert request.layer_budget_multipliers == config.allocation.layer_budget_multipliers
    assert request.nonfactorized_tuning_epochs == 0
    assert request.nonfactorized_tuning_epochs_by_layer == (8, 4, 3, 2, 2, 2, 2)
    assert request.factorized_tuning_epochs == 8
    assert request.factorized_tuning_binary_learning_rate == 1e-5
    assert request.factorized_tuning_learning_rate == 1e-5
    assert request.factorized_tuning_outlier_learning_rate == 1e-5
    assert request.factorized_tuning_bias_learning_rate == 1e-5
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
    assert manifest.launcher.experiment_number == 1
    assert manifest.launcher.repository_relative_path == (
        "experiments/001-compress-gemma-3-1b-it.py"
    )
    assert cast(dict[str, object], manifest.resolved_config)["canonical_run_config"]


def test_resident_recipe_maps_implicit_model_kd_defaults(tmp_path: Path) -> None:
    request = distillation_request_from_config(_resident_config(), _inputs(tmp_path))

    assert request.config.epochs == 8
    assert request.config.objective == "top_k"
    assert request.config.batch_size == 1
    assert request.config.learning_rate == 1e-5
    assert request.config.top_k == 64
    assert request.config.maximum_tokens_per_batch == 512
    assert request.config.maximum_batches_per_epoch is None
    assert request.config.tail_mass_weight == 1.0
    assert request.config.gradient_checkpointing
    assert request.config.weight_decay == 0.0
    assert request.config.optimizer_version == "legacy-optimi-adamw-v1"
    assert request.config.sampling_version == "legacy-python-device-rng-v1"
    assert request.block_snapshot_samples == 4
    assert request.block_snapshot_tokens == 512


def test_resident_mapping_rejects_unimplemented_semantics(tmp_path: Path) -> None:
    config = _resident_config()
    unsupported = replace(config, distillation=replace(config.distillation, loss=DistillationLoss.FULL_KL))

    with pytest.raises(ValueError, match="only top_k and top_k_tail are implemented"):
        resident_request_from_config(unsupported, _inputs(tmp_path))


def test_resident_recipe_maps_distinct_factorized_learning_rates(tmp_path: Path) -> None:
    base = _resident_config()
    factorized = replace(
        base.block_tuning.factorized,
        learning_rates=FactorizedLearningRates(
            binary=3e-5,
            scale=1e-5,
            outlier=2e-5,
            bias=4e-5,
        ),
    )
    config = replace(
        base,
        block_tuning=replace(base.block_tuning, factorized=factorized),
    )

    request = resident_request_from_config(config, _inputs(tmp_path))

    assert request.factorized_tuning_binary_learning_rate == 3e-5
    assert request.factorized_tuning_learning_rate == 1e-5
    assert request.factorized_tuning_outlier_learning_rate == 2e-5
    assert request.factorized_tuning_bias_learning_rate == 4e-5


def test_resident_recipe_maps_tail_kd_protocol(tmp_path: Path) -> None:
    base = _resident_config()
    configured = replace(
        base,
        distillation=replace(
            base.distillation,
            loss=DistillationLoss.TOP_K_TAIL,
            maximum_batches_per_epoch=32,
            tail_mass_weight=0.5,
        ),
    )

    request = distillation_request_from_config(configured, _inputs(tmp_path))

    assert request.config.objective == "top_k_tail"
    assert request.config.maximum_batches_per_epoch == 32
    assert request.config.tail_mass_weight == 0.5


def test_resident_recipe_maps_warm_started_mass_floor_correction(tmp_path: Path) -> None:
    base = _resident_config()
    expected_protocol_hash = workflow.distillation_protocol_hash(
        distillation_request_from_config(base, _inputs(tmp_path)).config
    )
    correction = replace(
        base.distillation.mass_floor_correction,
        enabled=True,
        expected_initializer_protocol_hash=expected_protocol_hash,
        expected_initializer_steps=256,
        epochs=1,
        learning_rate=1e-5,
        maximum_batches_per_epoch=32,
        scheduler_total_steps=128,
        minimum_teacher_mass_ratio=0.8,
        mass_loss_weight=2.0,
    )
    configured = replace(
        base,
        distillation=replace(base.distillation, mass_floor_correction=correction),
    )
    initializer = ArtifactRef("global-tuning-result", "sha256-" + "a" * 64, 1)
    options = ResidentExecutionOptions(
        interrupt_after_mass_floor_correction_epoch_commits=1
    )

    request = mass_floor_correction_request_from_config(
        configured,
        _inputs(tmp_path),
        initializer,
        options,
    )

    assert request.config.objective == "top_k_mass_floor"
    assert request.config.epochs == 1
    assert request.config.learning_rate == 1e-5
    assert request.config.maximum_batches_per_epoch == 32
    assert request.config.scheduler_total_steps == 128
    assert request.config.minimum_teacher_mass_ratio == 0.8
    assert request.config.mass_floor_weight == 2.0
    assert request.initializer_global_tuning == initializer
    assert request.expected_initializer_protocol_hash == expected_protocol_hash
    assert request.expected_initializer_steps == 256
    assert request.state_namespace == "global-distillation-mass-floor"
    assert request.interrupt_after_epoch_commits == 1

    mismatched = replace(
        configured,
        distillation=replace(
            configured.distillation,
            mass_floor_correction=replace(
                correction,
                expected_initializer_protocol_hash="sha256:" + "0" * 64,
            ),
        ),
    )
    with pytest.raises(ValueError, match="frozen initializer regime"):
        mass_floor_correction_request_from_config(
            mismatched,
            _inputs(tmp_path),
            initializer,
            options,
        )


def test_resident_recipe_maps_dense_objective_to_covariance_refinement(tmp_path: Path) -> None:
    base = _resident_config()
    objective = replace(
        base.calibration.objective,
        kind=ObjectiveKind.DENSE_HESSIAN,
        sampling=replace(
            base.calibration.objective.sampling,
            max_tokens_per_layer=8192,
        ),
    )
    config = replace(base, calibration=replace(base.calibration, objective=objective))

    request = resident_request_from_config(config, _inputs(tmp_path))

    assert request.covariance_refinement == objective


def test_resident_recipe_maps_post_refit_covariance_selection(tmp_path: Path) -> None:
    base = _resident_config()
    selected = PostRefitCovarianceRefinementConfig(
        enabled=True,
        block_indices=(5, 11, 24, 25),
        shared_input_groups=("self_attn.attn_qkv",),
    )
    config = replace(
        base,
        block_tuning=replace(
            base.block_tuning,
            post_refit_covariance_refinement=selected,
        ),
    )

    request = resident_request_from_config(config, _inputs(tmp_path))

    assert request.covariance_refinement is None
    assert request.post_refit_covariance_refinement == selected


def test_cpu_offload_mapping_requires_large_model_guards(tmp_path: Path) -> None:
    base = _resident_config()
    guarded = replace(
        base,
        calibration=replace(base.calibration, method=CalibrationMethod.FORWARD_ONLY),
        runtime=replace(base.runtime, executor=ExecutorKind.CPU_OFFLOAD),
        evaluation=replace(base.evaluation, inline_quality=False),
        distillation=replace(base.distillation, enabled=False),
    )

    request = resident_request_from_config(
        guarded,
        _inputs(tmp_path),
        ResidentExecutionOptions(restore_completed_blocks=False),
    )

    assert request.executor is ExecutorKind.CPU_OFFLOAD
    assert not request.restore_completed_blocks
    assert not request.evaluate_inline_quality
    with pytest.raises(ValueError, match="requires complete precomputed calibration"):
        resident_request_from_config(
            replace(guarded, calibration=base.calibration),
            _inputs(tmp_path),
            ResidentExecutionOptions(restore_completed_blocks=False),
        )
    precomputed = replace(
        _inputs(tmp_path),
        precomputed_calibration=ArtifactRef("calibration", "calibration-id", 1),
        precomputed_objectives=ArtifactRef("objectives", "objectives-id", 1),
        precomputed_plan=ArtifactRef("plan", "plan-id", 1),
    )
    fisher_request = resident_request_from_config(
        replace(guarded, calibration=base.calibration),
        precomputed,
        ResidentExecutionOptions(restore_completed_blocks=False),
    )
    assert fisher_request.precomputed_calibration == precomputed.precomputed_calibration
    with pytest.raises(ValueError, match="inline quality"):
        resident_request_from_config(
            replace(guarded, evaluation=base.evaluation),
            _inputs(tmp_path),
            ResidentExecutionOptions(restore_completed_blocks=False),
        )
    with pytest.raises(ValueError, match="distillation"):
        resident_request_from_config(
            replace(guarded, distillation=base.distillation),
            _inputs(tmp_path),
            ResidentExecutionOptions(restore_completed_blocks=False),
        )


def test_activation_gpu_cache_policy_maps_as_nonsemantic_execution_control(tmp_path: Path) -> None:
    base = _resident_config()
    cached = replace(
        base,
        runtime=replace(
            base.runtime,
            activations=ActivationStorageConfig(
                gpu_cache=ActivationGpuCacheMode.AUTO,
                gpu_reserve_gib=1.25,
            ),
        ),
    )

    request = resident_request_from_config(cached, _inputs(tmp_path))

    assert request.activation_gpu_cache is ActivationGpuCacheMode.AUTO
    assert request.activation_gpu_reserve_bytes == 1_342_177_280


def test_sweep_reuse_paths_map_as_nonsemantic_execution_controls(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    baseline = resident_request_from_config(_resident_config(), inputs)
    options = ResidentExecutionOptions(
        interrupt_after_preprocessing=True,
        preprocessing_reuse_run=tmp_path / "calibration-donor",
        rank_probe_reuse_run=tmp_path / "probe-donor",
    )

    request = resident_request_from_config(_resident_config(), inputs, options)

    assert request.interrupt_after_preprocessing
    assert request.preprocessing_reuse_run == options.preprocessing_reuse_run
    assert request.rank_probe_reuse_run == options.rank_probe_reuse_run
    assert resident._resident_manifest_config(request, "resident-quantization") == resident._resident_manifest_config(
        baseline,
        "resident-quantization",
    )


def test_combined_workflow_runs_quantization_before_distillation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _resident_config()
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


def test_combined_workflow_runs_mass_floor_correction_after_primary_distillation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _resident_config()
    inputs = _inputs(tmp_path)
    expected_protocol_hash = workflow.distillation_protocol_hash(
        distillation_request_from_config(base, inputs).config
    )
    config = replace(
        base,
        distillation=replace(
            base.distillation,
            mass_floor_correction=replace(
                base.distillation.mass_floor_correction,
                enabled=True,
                expected_initializer_protocol_hash=expected_protocol_hash,
                expected_initializer_steps=256,
            ),
            final_norm_calibration=replace(
                base.distillation.final_norm_calibration,
                enabled=True,
                scale=1.015,
            ),
        ),
    )
    calls: list[tuple[str, ArtifactRef | None]] = []
    primary_reference = ArtifactRef("global-tuning-result", "sha256-" + "a" * 64, 1)
    correction_reference = ArtifactRef("global-tuning-result", "sha256-" + "b" * 64, 1)
    calibrated_reference = ArtifactRef("global-tuning-result", "sha256-" + "c" * 64, 1)
    primary = type("Result", (), {"reference": primary_reference})()
    correction = type("Result", (), {"reference": correction_reference})()

    monkeypatch.setattr(workflow, "run_resident_quantization", lambda _request: object())

    def distill(request: Any) -> Any:
        calls.append((request.state_namespace, request.initializer_global_tuning))
        return primary if len(calls) == 1 else correction

    monkeypatch.setattr(workflow, "run_global_topk_distillation", distill)
    calibrations: list[tuple[Path, ArtifactRef, float]] = []
    calibrated_result = type(
        "CalibratedResult",
        (),
        {
            "epoch_losses": (1.0,),
            "steps_completed": 1,
            "selected_parameter_count": 1,
            "teacher_cache_bytes": 1,
        },
    )()

    def calibrate(run_output: Path, source: ArtifactRef, scale: float) -> Any:
        calibrations.append((run_output, source, scale))
        return type(
            "CommittedCalibration",
            (),
            {"reference": calibrated_reference, "result": calibrated_result},
        )()

    monkeypatch.setattr(workflow, "calibrate_global_tuning_final_norm", calibrate)
    transitions = []
    monkeypatch.setattr(
        workflow,
        "_transition_workflow_manifest",
        lambda *_args, **kwargs: transitions.append(kwargs),
    )

    result = execute_resident_workflow(config, inputs)

    assert calls == [
        ("global-distillation", None),
        ("global-distillation-mass-floor", primary_reference),
    ]
    assert calibrations == [(inputs.output, correction_reference, 1.015)]
    assert result.distillation is not None
    assert result.distillation.reference == calibrated_reference
    assert transitions == [{"artifact_id": calibrated_reference.artifact_id}]


def test_completed_workflow_explicitly_loads_historical_algorithm_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _resident_config()
    config = replace(base, distillation=replace(base.distillation, enabled=False))
    inputs = _inputs(tmp_path)
    directory = RunDirectory(inputs.output.parent, inputs.output.name)
    manifest = initial_manifest_from_resolved(
        "sha256:historical",
        {},
        launcher_provenance("experiments/001-compress-gemma-3-1b-it.py", 1),
        {},
    )
    directory.write_manifest(transition(transition(manifest, RunStatus.RUNNING), RunStatus.COMPLETED))
    request = object()
    quantization = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(workflow, "_resolve_workflow_memory_plan", lambda *_args: (None, None))
    monkeypatch.setattr(workflow, "resident_request_from_config", lambda *_args, **_kwargs: request)

    def load_completed(
        candidate: object,
        *,
        allow_historical_algorithm: bool = False,
        allow_relocated_run: bool = False,
    ) -> object:
        observed["request"] = candidate
        observed["allow_historical_algorithm"] = allow_historical_algorithm
        observed["allow_relocated_run"] = allow_relocated_run
        return quantization

    monkeypatch.setattr(workflow, "load_completed_resident_quantization", load_completed)

    result = load_completed_resident_workflow(config, inputs)

    assert result is not None
    assert cast(Any, result.quantization) is quantization
    assert result.distillation is None
    assert observed == {
        "request": request,
        "allow_historical_algorithm": True,
        "allow_relocated_run": False,
    }


def test_zero_argument_resolution_generates_run_local_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    launcher = repository / "experiments" / "001-compress-gemma-3-1b-it.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = repository / "snapshot"
    snapshot.mkdir()
    config = _resident_config()
    config = replace(config, model=replace(config.model, source=str(snapshot)))
    tokens = torch.arange(256 * 16, dtype=torch.long).reshape(256, 16)
    observed: dict[str, object] = {}

    def load_calibration(snapshot_path: Path, path: Path, **kwargs: object) -> Any:
        observed["snapshot"] = snapshot_path
        observed["path"] = path
        observed["kwargs"] = kwargs
        return type("Calibration", (), {"input_ids": tokens})()

    monkeypatch.setattr(workflow, "load_or_prepare_calibration", load_calibration)
    monkeypatch.setattr(
        workflow,
        "load_repository_dotenv",
        lambda path: observed.setdefault("dotenv_root", path) or True,
    )
    monkeypatch.setattr(
        workflow.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: type("Tokenizer", (), {"pad_token_id": 7})(),
    )

    resolved = resolve_resident_experiment_inputs(config, launcher_path=launcher)

    assert resolved.snapshot == snapshot
    assert resolved.output == repository / "evidence/001" / config.intent.name
    assert resolved.registry_root == repository / "evidence/001"
    assert torch.equal(cast(torch.Tensor, resolved.token_ids), tokens)
    assert torch.equal(cast(torch.Tensor, resolved.quality_token_ids), tokens[:1, :8])
    assert resolved.pad_token_id == 7
    assert observed["dotenv_root"] == repository
    assert observed["snapshot"] == snapshot
    assert observed["path"] == resolved.output
    assert observed["kwargs"] == {
        "sample_count": 256,
        "sequence_length": 2048,
        "seed": 0,
        "preparation_id": workflow.config_hash(config),
    }

    override = repository / "evidence" / "reproducibility" / "run-a"
    overridden = resolve_resident_experiment_inputs(
        config,
        launcher_path=launcher,
        output_override=override,
    )
    assert overridden.output == override.resolve()
    assert observed["path"] == override.resolve()


def test_zero_argument_resolution_rejects_incompatible_run_before_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    launcher = repository / "experiments" / "001-compress-gemma-3-1b-it.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = repository / "snapshot"
    snapshot.mkdir()
    config = replace(_resident_config(), model=replace(_resident_config().model, source=str(snapshot)))
    output = repository / "evidence" / "001" / config.intent.name
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        '{"config_hash":"sha256:incompatible"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        workflow,
        "load_or_prepare_calibration",
        lambda *_args, **_kwargs: pytest.fail("calibration must not start"),
    )
    monkeypatch.setattr(workflow, "load_repository_dotenv", lambda _path: True)

    with pytest.raises(ValueError, match="before calibration preparation.*rollover"):
        resolve_resident_experiment_inputs(config, launcher_path=launcher)


def test_zero_argument_resolution_accepts_matching_canonical_config_in_resident_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    launcher = repository / "experiments" / "001-compress-gemma-3-1b-it.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = repository / "snapshot"
    snapshot.mkdir()
    config = replace(_resident_config(), model=replace(_resident_config().model, source=str(snapshot)))
    output = repository / "evidence" / "001" / config.intent.name
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "config_hash": "sha256:resident-execution-request",
                "resolved_config": {"canonical_run_config": to_dict(config)},
            }
        ),
        encoding="utf-8",
    )
    tokens = torch.arange(256 * 16, dtype=torch.long).reshape(256, 16)
    observed: dict[str, object] = {}

    def load_calibration(*_args: object, **kwargs: object) -> Any:
        observed["preparation_id"] = kwargs["preparation_id"]
        return type("Calibration", (), {"input_ids": tokens})()

    monkeypatch.setattr(workflow, "load_or_prepare_calibration", load_calibration)
    monkeypatch.setattr(workflow, "load_repository_dotenv", lambda _path: True)
    monkeypatch.setattr(
        workflow.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: type("Tokenizer", (), {"pad_token_id": 7})(),
    )

    resolve_resident_experiment_inputs(config, launcher_path=launcher)

    assert observed == {"preparation_id": workflow.config_hash(config)}


def test_workflow_manifest_completes_only_with_global_tuning_artifact(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    directory = RunDirectory(inputs.output.parent, inputs.output.name)
    manifest = initial_manifest_from_resolved(
        "sha256:config",
        {},
        launcher_provenance(
            "experiments/001-compress-gemma-3-1b-it.py",
            1,
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


def test_resident_manifest_reopens_failed_run_with_current_inputs(tmp_path: Path) -> None:
    config = _resident_config()
    request = resident_request_from_config(config, _inputs(tmp_path))
    current = resident._resident_manifest(request, "resident-quantization")
    failed = transition(
        transition(current, RunStatus.RUNNING),
        RunStatus.FAILED,
        artifacts=("sha256-durable",),
        failure={"type": "RuntimeError", "message": "injected"},
    )

    resumed = resident._start_resident_manifest(failed, current)

    assert resumed.status is RunStatus.RUNNING
    assert resumed.run_id == failed.run_id
    assert resumed.artifacts == ("sha256-durable",)
    assert resumed.failure is None
