import json
from pathlib import Path

import torch

from tools.compare_quantization_plan import compare_quantization_plan


def test_compare_quantization_plan_reports_rank_and_outlier_deltas(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "layers": [
                            {
                                "layer": {"block": {"index": 3}, "path": "mlp.gate_proj"},
                                "rank": 32,
                                "outliers": {"count": 1},
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model.layers.3.mlp.gate_proj.U_shape": torch.tensor([8, 64]),
            "model.layers.3.mlp.gate_proj.salient_idx": torch.tensor([2, 5]),
        },
        checkpoint_path,
    )

    result = compare_quantization_plan(plan_path, checkpoint_path)

    assert result["layer_count"] == 1
    assert result["rank_mismatch_count"] == 1
    assert result["legacy_rank_sum"] == 64
    assert result["planned_rank_sum"] == 32
    assert result["absolute_rank_delta_sum"] == 32
    assert result["outlier_count_mismatch_count"] == 1
    assert result["legacy_outlier_count"] == 2
    assert result["planned_outlier_count"] == 1
    assert result["layers"][0]["rank_delta"] == -32
    assert result["layers"][0]["outlier_count_delta"] == -1
