from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nanoquant.kl_budget_workflow as budget
import nanoquant.kl_teacher_dataset_workflow as builder
import nanoquant.self_measured_d2_workflow as self_measured
from nanoquant.application.kl_budget import load_kl_budget_profile
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.artifacts import ArtifactCorruptionError, LocalArtifactStore
from nanoquant.infrastructure.commits import CommitIdentity
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.kl_teacher_dataset import load_kl_teacher_dataset
from tests.support.experiments import load_experiment
from tests.unit.test_kl_budget import _ToyCausalModel


@pytest.fixture(params=["llama", "gemma3"])
def teacher_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest,
) -> dict[str, object]:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    text_config = {"vocab_size": 11, "num_hidden_layers": 1}
    config = ({"model_type": "gemma3", "text_config": text_config}
              if request.param == "gemma3" else {"model_type": "llama", **text_config})
    (snapshot / "config.json").write_text(json.dumps(config))
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1], [2, 3, 4, 5]])
    torch.manual_seed(17)
    model = _ToyCausalModel(11, 5)
    for module in (builder, budget):
        monkeypatch.setattr(module, "load_causal_language_model", lambda *_args, **_kwargs: model)
        monkeypatch.setattr(module, "hash_hf_tokenizer_snapshot", lambda *_args: "tokenizer")
        monkeypatch.setattr(
            module, "adapter_for_config", lambda *_args: SimpleNamespace(attention_implementation="eager")
        )
        monkeypatch.setattr(module, "_wikitext_tokens", lambda *_args, **_kwargs: (tokens, "wiki-fixture", None))
    monkeypatch.setenv("NANOQUANT_DEVICE_LEASE_ROOT", str(tmp_path / "device-leases"))
    return dict(
        snapshot=snapshot, source="model", revision="a" * 40, device="cpu", samples=3, sequence_length=4, batch_size=2
    )


def test_build_extract_and_resume_without_teacher_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_inputs: dict[str, object],
) -> None:
    original = builder.build_kl_teacher_dataset(**teacher_inputs, output=tmp_path / "first")

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("reused targets must not execute teacher inference")

    monkeypatch.setattr(builder, "load_causal_language_model", unexpected)
    extracted = builder.build_kl_teacher_dataset(
        **teacher_inputs,
        output=tmp_path / "extracted",
        reuse_cache_root=tmp_path / "first",
    )
    assert extracted.reference == original.reference
    assert extracted.cache.reference == original.cache.reference
    assert torch.equal(extracted.tokens, original.tokens)
    for before, after in zip(original.cache.batches, extracted.cache.batches, strict=True):
        assert torch.equal(before, after)
    monkeypatch.setattr(builder, "_wikitext_tokens", unexpected)
    resumed = builder.build_kl_teacher_dataset(**teacher_inputs, output=tmp_path / "extracted")
    assert resumed.reference == original.reference


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "other"),
        ("revision", "b" * 40),
        ("samples", 2),
        ("sequence_length", 5),
        ("batch_size", 1),
        ("device", "cuda"),
    ],
)
def test_reuse_rejects_incompatible_protocol(
    tmp_path: Path,
    teacher_inputs: dict[str, object],
    field: str,
    value: object,
) -> None:
    builder.build_kl_teacher_dataset(**teacher_inputs, output=tmp_path / "dataset")
    with pytest.raises(ValueError, match="differs from the requested"):
        builder.build_kl_teacher_dataset(**{**teacher_inputs, field: value}, output=tmp_path / "dataset")


def test_corrupt_dataset_tokens_fail_validation(tmp_path: Path, teacher_inputs: dict[str, object]) -> None:
    root = tmp_path / "dataset"
    dataset = builder.build_kl_teacher_dataset(**teacher_inputs, output=root)
    path = LocalArtifactStore(root / "artifacts").path_for(dataset.reference.artifact_id) / "tokens.safetensors"
    del dataset
    path.write_bytes(b"invalid tokens")
    with pytest.raises(ArtifactCorruptionError):
        load_kl_teacher_dataset(root)


def test_kl_workflow_uses_saved_tokens_and_targets_and_matches_live_scoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_inputs: dict[str, object],
) -> None:
    root = tmp_path / "dataset"
    dataset = builder.build_kl_teacher_dataset(**teacher_inputs, output=root)
    model = budget.load_causal_language_model(teacher_inputs["snapshot"])
    layer = LayerId(BlockId(0), "proj")
    weight = model.model.layers[0].proj.weight.detach().clone()
    reconstructions = SpliceReconstructionSet(
        (SpliceReconstruction(layer, weight + torch.eye(5) * 0.1, None, 1.0),),
        (("0:proj", (layer,)),),
        (("0:proj", 1.0),),
    )
    evaluator = DenseKlSpliceEvaluator(model, reconstructions, dataset.tokens, device="cpu", batch_size=2)
    expected = evaluator("full")
    monkeypatch.setattr(
        budget,
        "load_splice_reconstructions_from_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            reconstructions=reconstructions,
            identity=CommitIdentity("config", dataset.protocol.model_hash, "plan"),
            global_tuning=None,
        ),
    )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        pytest.fail("dataset consumer must use persisted tokens and teacher targets")

    monkeypatch.setattr(budget, "_wikitext_tokens", unexpected)
    monkeypatch.setattr(DenseKlSpliceEvaluator, "cache_teacher", unexpected)
    args = argparse.Namespace(
        **{key: value for key, value in teacher_inputs.items() if key != "samples"},
        wikitext_samples=3,
        teacher_dataset=root,
        teacher_cache_mode="cpu",
        teacher_cache_root=None,
        profile_output=tmp_path / "profile",
        run_output=tmp_path / "control",
        token_chunk_size=128,
        use_global_tuning=False,
        local_files_only=True,
        arm=["full"],
    )
    assert budget.execute_kl_budget(args) == 0
    profile = load_kl_budget_profile(args.profile_output / "kl-budget-profile.json")
    assert profile.arms == (expected,)
    assert profile.baseline_negative_log_likelihood == dataset.cache.baseline_negative_log_likelihood
    receipt = json.loads((args.profile_output / "artifact.json").read_text())
    assert receipt["teacher_cache_reused"] is True
    assert receipt["teacher_dataset"]["artifact"]["artifact_id"] == dataset.reference.artifact_id


def test_numbered_workflow_accepts_dataset_and_validates_completed_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    teacher_inputs: dict[str, object],
) -> None:
    root = tmp_path / "dataset"
    dataset = builder.build_kl_teacher_dataset(**{**teacher_inputs, "batch_size": 1}, output=root)
    experiment = load_experiment(70)
    experiment = replace(
        experiment,
        config=replace(
            experiment.config,
            model=replace(experiment.config.model, source="model", revision="a" * 40),
            runtime=replace(experiment.config.runtime, compute_device="cpu"),
        ),
    )
    observed = []
    monkeypatch.setattr(
        self_measured,
        "_prepare_automatic_kl_inputs",
        lambda *_args, **kwargs: (
            observed.append(kwargs["teacher_dataset"]) or tmp_path / "profile",
            tmp_path / "control",
        ),
    )
    provenance = SimpleNamespace(
        dataset_slice_hash=dataset.protocol.token_hash, dataset_fingerprint=dataset.protocol.dataset_fingerprint
    )
    monkeypatch.setattr(
        self_measured,
        "_validated_kl_profile",
        lambda *_args, **_kwargs: SimpleNamespace(
            complete=True,
            profile_key="key",
            provenance=provenance,
        ),
    )
    monkeypatch.setattr(self_measured, "run_compression_quality_experiment", lambda *_args, **_kwargs: 0)
    options = dict(
        launcher_path=tmp_path / "experiments/070.py",
        arguments=["--kl-teacher-dataset", str(root)],
        profile_options=self_measured.SelfMeasuredD2ProfileOptions(wikitext_samples=3, sequence_length=4),
    )
    assert self_measured.run_self_measured_d2_experiment(experiment, **options) == 0
    assert observed == [root]
    provenance.dataset_slice_hash = "other-tokens"
    with pytest.raises(ValueError, match="different inputs"):
        self_measured.run_self_measured_d2_experiment(experiment, **options)
