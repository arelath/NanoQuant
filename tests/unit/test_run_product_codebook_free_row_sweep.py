from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.run_product_codebook_free_row_sweep import (
    _read_fixed_outliers,
    build_probe_command,
)


def test_reads_exact_down_projection_outliers_from_logical_shard(
    tmp_path: Path,
) -> None:
    save_file(
        {"blocks.3.mlp.down_proj.outlier_indices": torch.tensor([7, 11, 29])},
        tmp_path / "block-00003.safetensors",
    )

    assert _read_fixed_outliers(tmp_path, 3) == (7, 11, 29)


def test_probe_command_carries_all_free_rows_and_exact_outliers() -> None:
    command = build_probe_command(
        python=Path("python.exe"),
        probe=Path("probe.py"),
        model=Path("model.safetensors"),
        calibration_state=Path("calibration"),
        output=Path("block.json"),
        block=4,
        fixed_outliers=(3, 17),
        free_row_counts=(576, 640, 704),
        baseline_rank=970,
        candidate_rank=1152,
        outer_iterations=1200,
        seed=0,
        device="cuda:0",
    )

    assert command[command.index("--right-free-row-counts") + 1] == "576,640,704"
    assert command[command.index("--fixed-outlier-indices") + 1] == "3,17"
    assert command[command.index("--candidate-outlier-columns") + 1] == "2"
    assert command[command.index("--outer-iterations") + 1] == "1200"
