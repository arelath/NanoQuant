from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tools.materialize_product_codebook_mixed_policy import (
    MaterializationJob,
    MaterializationSettings,
    build_probe_command,
    load_policy_jobs,
    validate_job_receipt,
    write_overlay_bundle,
)


def _settings() -> MaterializationSettings:
    return MaterializationSettings(
        model_revision="revision",
        calibration_state="calibration",
        baseline_rank=970,
        candidate_rank=1152,
        outer_iterations=1200,
        inner_iterations=5,
        regularization=0.03,
        penalty_schedule="cubic",
        convergence_check_interval=100,
        codebook_update_interval=10,
        codebook_freeze_fraction=0.5,
        assignment_batch_words=8192,
        linear_assignment_sweeps=2,
        corrected_assignment_candidates=16,
        scale_fit_passes=2,
        calibration_shrinkage=0.6,
        seed=0,
    )


def _command(job: MaterializationJob) -> list[str]:
    return build_probe_command(
        job,
        python=Path("python.exe"),
        probe=Path("probe.py"),
        model=Path("model.safetensors"),
        snapshot=Path("snapshot"),
        calibration_state=Path("calibration"),
        output=Path("output.json"),
        reconstruction_cache=Path("cache"),
        model_revision="revision",
        device="cuda:0",
    )


def test_gate_up_command_preserves_projection_specific_rows_and_search() -> None:
    job = MaterializationJob(
        "block-03-gate-up",
        3,
        ("gate", "up"),
        (("gate", 672), ("up", 704)),
        (768, 890),
        _settings(),
    )

    command = _command(job)

    assert command[command.index("--projections") + 1] == "gate,up"
    assert command[command.index("--right-free-rows-by-projection") + 1] == ("gate:672,up:704")
    assert "--transpose-matrix" in command
    assert "--binary-search" in command
    assert command[command.index("--fixed-outlier-indices") + 1] == "768,890"
    assert command[command.index("--wikitext-samples") + 1] == "1"


def test_down_command_keeps_source_orientation_and_exact_outliers() -> None:
    job = MaterializationJob(
        "block-01-down",
        1,
        ("down",),
        (("down", 704),),
        (1023, 1328, 1704),
        _settings(),
    )

    command = _command(job)

    assert command[command.index("--projection") + 1] == "down"
    assert command[command.index("--right-free-rows") + 1] == "704"
    assert "--transpose-matrix" not in command
    assert command[command.index("--fixed-outlier-indices") + 1] == ("1023,1328,1704")


def test_single_wide_projection_command_transposes_only_that_projection() -> None:
    job = MaterializationJob(
        "block-00-gate",
        0,
        ("gate",),
        (("gate", 672),),
        (367, 768),
        _settings(),
    )

    command = _command(job)

    assert command[command.index("--projection") + 1] == "gate"
    assert command[command.index("--right-free-rows") + 1] == "672"
    assert "--transpose-matrix" in command


def test_job_receipt_requires_exact_cache_and_policy_identity(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    job = MaterializationJob(
        "block-04-gate-up",
        4,
        ("gate", "up"),
        (("gate", 640), ("up", 672)),
        (0, 768),
        _settings(),
    )
    receipt = tmp_path / "job.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "completed",
                "blocks": [4],
                "projections": ["mlp.gate_proj", "mlp.up_proj"],
                "candidate": {"right_free_rows_by_projection": {"gate": 640, "up": 672}},
                "fixed_outlier_columns": {"indices": [0, 768]},
                "reconstruction_cache": {
                    "root": str(cache),
                    "keys_by_unit": {"4:gate": "gate-key", "4:up": "up-key"},
                },
            }
        ),
        encoding="utf-8",
    )

    assert validate_job_receipt(receipt, job, cache) == {
        "gate": "gate-key",
        "up": "up-key",
    }

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["candidate"]["right_free_rows_by_projection"]["gate"] = 672
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    try:
        validate_job_receipt(receipt, job, cache)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("mismatched materialization receipt was accepted")


def test_overlay_bundle_publishes_both_arms_atomically(tmp_path: Path) -> None:
    baseline = {
        "model.layers.0.mlp.gate_proj.weight": torch.zeros((3, 2)),
    }
    candidate = {
        "model.layers.0.mlp.gate_proj.weight": torch.ones((3, 2)),
    }
    destination = tmp_path / "overlays"

    manifests = write_overlay_bundle(destination, baseline, candidate)

    assert manifests["free_words"]["arm"] == "free_words"
    assert manifests["corrected_codebook"]["arm"] == "corrected_codebook"
    for directory in ("free-words", "corrected-codebook"):
        assert (destination / directory / "manifest.json").is_file()
        assert (destination / directory / "weights.safetensors").is_file()


def test_materializer_accepts_kl_calibrated_allocation_schema(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "allocation.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "status": "completed",
                "effective_bpw": 1.0,
                "target_bpw": 1.0,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input or selection inventory"):
        load_policy_jobs(receipt)
