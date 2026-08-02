import json
import os
from types import SimpleNamespace

import pytest
import torch

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.domain.runs import LauncherProvenance, RunManifest, RunStatus
from tools.materialize_topk_tail_checkpoint import (
    _checkpoint_result_metadata,
    _derived_manifest,
    _exact_reload_audit,
    _hardlink_tree,
    _parser,
)


def test_materializer_uses_resident_endpoint_metadata_without_probe_report(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = (ArtifactRef("activation-generation", "sha256-" + "a" * 64, 1),)
    pointer = ArtifactRef("global-tuning-result", "sha256-" + "b" * 64, 1)
    (tmp_path / "global-distillation-result.json").write_text(
        json.dumps(to_dict(pointer)),
        encoding="utf-8",
    )
    checkpoint = SimpleNamespace(
        identity=SimpleNamespace(protocol_hash="sha256:protocol", source_blocks=source),
        state=SimpleNamespace(steps_completed=224),
    )
    endpoint = SimpleNamespace(
        protocol_hash="sha256:protocol",
        source_blocks=source,
        steps_completed=256,
        teacher_cache_bytes=123,
        wall_seconds=4.5,
    )
    monkeypatch.setattr(
        "tools.materialize_topk_tail_checkpoint.load_global_tuning",
        lambda *_args, **_kwargs: SimpleNamespace(result=endpoint),
    )

    assert _checkpoint_result_metadata(
        tmp_path,
        checkpoint,
        state_namespace="global-distillation",
    ) == (123, 4.5, "resident-global-tuning")


def test_parser_accepts_an_explicit_checkpoint_epoch() -> None:
    args = _parser().parse_args(
        [
            "--run-output",
            "run",
            "--snapshot",
            "snapshot",
            "--checkpoint-output",
            "checkpoint",
            "--derived-run-output",
            "derived",
            "--epoch",
            "1",
        ]
    )

    assert args.epoch == 1
    assert args.state_namespace == "global-distillation"


def test_parser_accepts_canonical_correction_namespace() -> None:
    args = _parser().parse_args(
        [
            "--run-output",
            "run",
            "--snapshot",
            "snapshot",
            "--checkpoint-output",
            "checkpoint",
            "--derived-run-output",
            "derived",
            "--epoch",
            "3",
            "--state-namespace",
            "global-distillation-mass-floor",
        ]
    )

    assert args.state_namespace == "global-distillation-mass-floor"


def test_hardlink_tree_reproduces_nested_inventory_without_copying(tmp_path) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "nested").mkdir(parents=True)
    (source / "root.json").write_text("root", encoding="utf-8")
    (source / "nested" / "tensor.bin").write_bytes(b"tensor")

    linked = _hardlink_tree(source, destination)

    assert linked == 2
    assert (destination / "root.json").read_text(encoding="utf-8") == "root"
    assert os.stat(source / "root.json").st_ino == os.stat(destination / "root.json").st_ino
    assert os.stat(source / "nested" / "tensor.bin").st_ino == os.stat(
        destination / "nested" / "tensor.bin"
    ).st_ino


def test_exact_reload_audit_binds_every_parameter() -> None:
    expected = {
        "model.layers.0.scale": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
        "model.norm.weight": torch.tensor([0.5], dtype=torch.float32),
    }
    actual = {name: value.clone() for name, value in expected.items()}

    audit = _exact_reload_audit(expected, actual)

    assert audit["passed"] is True
    assert audit["parameter_count"] == 2
    assert audit["element_count"] == 3
    assert audit["checkpoint_inventory_hash"] == audit["reloaded_inventory_hash"]
    assert [item["name"] for item in audit["parameters"]] == [
        "model.layers.0.scale",
        "model.norm.weight",
    ]


@pytest.mark.parametrize(
    "actual",
    (
        {"model.layers.0.scale": torch.tensor([1.0])},
        {"model.layers.0.scale": torch.tensor([1.0, 3.0])},
        {"model.layers.0.scale": torch.tensor([1.0, 2.0], dtype=torch.float64)},
    ),
)
def test_exact_reload_audit_fails_closed_on_inventory_or_value_change(
    actual: dict[str, torch.Tensor],
) -> None:
    expected = {"model.layers.0.scale": torch.tensor([1.0, 2.0])}

    with pytest.raises(ValueError, match="differs from its checkpoint"):
        _exact_reload_audit(expected, actual)


def test_derived_manifest_has_new_lineage_and_authorizes_selected_tuning() -> None:
    source = RunManifest(
        1,
        "source-run",
        RunStatus.COMPLETED,
        "created",
        "updated",
        "sha256:config",
        {"output": "source"},
        LauncherProvenance("numbered_runfile", 48, "experiments/048.py", "sha256:x", "rev", ()),
        {"machine": "fixture"},
        artifacts=("sha256-source", "sha256-old-tuning"),
    )

    derived = _derived_manifest(
        source,
        global_tuning_artifact_id="sha256-selected-tuning",
        arguments=("--epoch", "3"),
    )

    assert derived.status is RunStatus.COMPLETED
    assert derived.run_id != source.run_id
    assert derived.parent_run_id == source.run_id
    assert derived.forked_from_stage == "distillation-checkpoint-materialization"
    assert derived.config_hash == source.config_hash
    assert derived.resolved_config == source.resolved_config
    assert derived.artifacts == (
        "sha256-source",
        "sha256-old-tuning",
        "sha256-selected-tuning",
    )
