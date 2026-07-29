from __future__ import annotations

import torch

from tools.probe_attention_partition_functional import (
    _aggregate_topologies,
    _relative_output_rmse,
)


def test_aggregate_topologies_preserves_energy_weighting_and_physical_bits() -> None:
    result = _aggregate_topologies(
        (
            {
                "error_energy": 1.0,
                "target_energy": 4.0,
                "original_error_energy": 4.0,
                "original_target_energy": 16.0,
                "source_elements": 10,
                "actual_bits": 9,
            },
            {
                "error_energy": 3.0,
                "target_energy": 12.0,
                "original_error_energy": 12.0,
                "original_target_energy": 48.0,
                "source_elements": 30,
                "actual_bits": 27,
            },
        )
    )

    assert result["weighted_normalized_rmse"] == 0.5
    assert result["original_normalized_rmse"] == 0.5
    assert result["actual_bpw"] == 0.9


def test_relative_output_rmse_uses_all_sequence_energy() -> None:
    reference = (torch.ones((1, 2, 2)), torch.full((1, 1, 2), 2.0))
    candidate = (torch.zeros((1, 2, 2)), torch.ones((1, 1, 2)))

    assert _relative_output_rmse(reference, candidate) == 0.5**0.5
