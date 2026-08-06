from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.run_product_codebook_projection_free_row_screen import (
    PROJECTION_CONFIGS,
    _read_fixed_outliers,
    build_probe_command,
)


def test_reads_projection_specific_and_shared_qkv_outliers(tmp_path: Path) -> None:
    save_file(
        {
            "blocks.12.mlp.up_proj.outlier_indices": torch.tensor([7, 19]),
            "blocks.12.self_attn.attn_qkv.outlier_indices": torch.tensor([5, 23]),
        },
        tmp_path / "block-00012.safetensors",
    )

    assert _read_fixed_outliers(tmp_path, 12, PROJECTION_CONFIGS["up"]) == (7, 19)
    assert _read_fixed_outliers(tmp_path, 12, PROJECTION_CONFIGS["q"]) == (5, 23)
    assert _read_fixed_outliers(tmp_path, 12, PROJECTION_CONFIGS["v"]) == (5, 23)


def test_probe_command_uses_shape_specific_ladder_and_transpose() -> None:
    command = build_probe_command(
        python=Path("python.exe"),
        probe=Path("probe.py"),
        model=Path("model.safetensors"),
        calibration_state=Path("calibration"),
        output=Path("block-up.json"),
        block=24,
        config=PROJECTION_CONFIGS["up"],
        fixed_outliers=(412, 835),
        outer_iterations=1200,
        seed=0,
        device="cuda:0",
    )

    assert command[command.index("--projection") + 1] == "up"
    assert command[command.index("--candidate-rank") + 1] == "1152"
    assert (
        command[command.index("--right-free-row-counts") + 1]
        == "576,608,640,672,704"
    )
    assert command[-1] == "--transpose-matrix"


def test_attention_ladders_stay_below_shape_specific_rank() -> None:
    assert PROJECTION_CONFIGS["q"].free_row_counts == (256, 288, 320, 352)
    assert PROJECTION_CONFIGS["k"].free_row_counts == (32, 64, 96, 128)
    assert PROJECTION_CONFIGS["v"].free_row_counts == (32, 64, 96, 128)
    assert PROJECTION_CONFIGS["o"].transpose_matrix
