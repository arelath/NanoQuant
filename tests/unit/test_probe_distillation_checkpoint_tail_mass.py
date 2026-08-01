from __future__ import annotations

import json
from pathlib import Path

import torch

from tools.probe_distillation_checkpoint_tail_mass import (
    _apply_gemma_final_norm_scale,
    _scaled_result_name,
    discover_checkpoints,
)


def _write_checkpoint(root: Path, artifact_id: str, *, epoch: int, protocol: str) -> None:
    path = root / "artifacts" / artifact_id.removeprefix("sha256-")[:2] / artifact_id
    path.mkdir(parents=True)
    (path / "checkpoint.json").write_text(
        json.dumps(
            {
                "completed_epochs": epoch,
                "steps_completed": epoch * 32,
                "identity": {
                    "source_blocks": [],
                    "protocol_hash": protocol,
                    "token_hash": "sha256:tokens",
                    "target_hash": None,
                },
            }
        ),
        encoding="utf-8",
    )


def test_discover_checkpoints_filters_to_active_protocol(tmp_path: Path) -> None:
    active = "sha256-" + "a" * 64
    epoch_one = "sha256-" + "b" * 64
    unrelated = "sha256-" + "c" * 64
    _write_checkpoint(tmp_path, active, epoch=2, protocol="sha256:tail")
    _write_checkpoint(tmp_path, epoch_one, epoch=1, protocol="sha256:tail")
    _write_checkpoint(tmp_path, unrelated, epoch=1, protocol="sha256:conditional")
    (tmp_path / "global-distillation-training.json").write_text(
        json.dumps(
            {
                "artifact_id": active,
                "artifact_type": "distillation-checkpoint",
                "schema_version": 1,
            }
        ),
        encoding="utf-8",
    )

    candidates = discover_checkpoints(tmp_path, {1, 2})

    assert [(candidate.epoch, candidate.steps) for candidate in candidates] == [(1, 32), (2, 64)]
    assert [candidate.reference.artifact_id for candidate in candidates] == [epoch_one, active]


def test_scaled_result_name_preserves_legacy_single_scale_name() -> None:
    assert _scaled_result_name("epoch_8", 1.0, multiple=False) == "epoch_8"
    assert _scaled_result_name("epoch_8", 1.075, multiple=True) == "epoch_8@scale=1.075"


def test_apply_gemma_final_norm_scale_scales_effective_weight() -> None:
    source = torch.tensor([-0.25, 0.0, 0.5], dtype=torch.bfloat16)
    parameter = torch.empty_like(source)

    _apply_gemma_final_norm_scale(parameter, source, 1.2)

    assert torch.allclose(1.0 + parameter.float(), (1.0 + source.float()) * 1.2, atol=0.004)
