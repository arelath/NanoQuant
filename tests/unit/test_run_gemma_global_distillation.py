from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

import tools.run_gemma_global_distillation as launcher


def test_global_distillation_launcher_uses_canonical_recipe(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(
        launcher,
        "load_pinned_calibration",
        lambda *_args, **_kwargs: SimpleNamespace(input_ids=torch.zeros((256, 8), dtype=torch.long)),
    )
    monkeypatch.setattr(
        launcher.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(pad_token_id=0),
    )

    def distill(request):
        captured.append(request)
        return SimpleNamespace(
            reference=SimpleNamespace(artifact_id="sha256-fixture"),
            metrics=SimpleNamespace(
                epoch_losses=(1.0,),
                steps_completed=1,
                selected_parameter_count=1,
                teacher_cache_bytes=1,
            ),
            result=SimpleNamespace(
                wall_seconds=1.0,
                peak_gpu_bytes=1,
                peak_host_bytes=1,
                block_snapshot_protocol_hash="sha256:fixture",
                block_metrics=(),
            ),
        )

    monkeypatch.setattr(launcher, "run_global_topk_distillation", distill)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gemma_global_distillation.py",
            "--run-output",
            str(tmp_path / "run"),
            "--snapshot",
            str(tmp_path / "snapshot"),
        ],
    )

    launcher.main()

    assert len(captured) == 1
    request = captured[0]
    assert request.config.epochs == 8
    assert request.config.top_k == 64
    assert request.config.optimizer_version == "legacy-optimi-adamw-v1"
