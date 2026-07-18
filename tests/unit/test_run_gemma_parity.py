from pathlib import Path
from types import SimpleNamespace

import torch

import tools.run_gemma_parity as launcher
from tools.run_gemma_parity import _parser


def test_parity_launcher_inherits_logical_tuning_batch_for_microbatch_default() -> None:
    args = _parser().parse_args(["--output", "run", "--snapshot", "snapshot"])

    assert args.factorized_tuning_batch_size == 8
    assert args.nonfactorized_tuning_batch_size == 8
    assert args.post_block_refit_batch_size == 8
    assert args.tuning_microbatch_size is None
    assert args.output == Path("run")


def test_parity_launcher_keeps_explicit_memory_fallback_microbatch() -> None:
    args = _parser().parse_args(
        ["--output", "run", "--snapshot", "snapshot", "--tuning-microbatch-size", "1"]
    )

    assert args.tuning_microbatch_size == 1


def test_parity_launcher_builds_request_through_canonical_recipe(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr(
        launcher,
        "load_or_prepare_calibration",
        lambda *_args, **_kwargs: SimpleNamespace(input_ids=torch.zeros((256, 8), dtype=torch.long)),
    )

    def factor_slice(request):
        captured.append(request)
        return SimpleNamespace(layer=None, peak_device_bytes=0, elapsed_seconds=0.0, remaining_layers=0)

    monkeypatch.setattr(launcher, "run_resident_factorization_slice", factor_slice)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_gemma_parity.py",
            "--output",
            str(tmp_path / "run"),
            "--snapshot",
            str(tmp_path / "snapshot"),
            "--factor-only",
        ],
    )

    launcher.main()

    assert len(captured) == 1
    request = captured[0]
    assert request.run_config is not None
    assert request.run_config.intent.experiment_number is None
    assert request.factorized_tuning_epochs == 0
    assert request.nonfactorized_tuning_epochs == 0
    assert request.rank_retry.maximum_attempts == 3
