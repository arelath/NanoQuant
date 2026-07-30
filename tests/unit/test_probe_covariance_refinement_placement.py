from __future__ import annotations

import torch

from nanoquant.domain.scale_fit import reconstruct
from tools.probe_covariance_refinement_placement import (
    FrozenFactorSnapshot,
    _parser,
    _residual_target_and_addition,
)


def test_residual_target_preserves_existing_outlier_or_patch_addition() -> None:
    left = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])
    right = torch.tensor([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    pre = torch.tensor([0.0, 0.8, 1.2])
    mid = torch.tensor([0.7, -0.4])
    post = torch.tensor([1.1, 0.6])
    factor = reconstruct(left, right, pre, mid, post)
    addition = torch.tensor([[0.5, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    original = torch.tensor([[0.8, -0.3, 0.4], [-0.5, 0.2, 0.1]])
    snapshot = FrozenFactorSnapshot(
        left,
        right,
        pre,
        mid,
        post,
        factor + addition,
        None,
        torch.tensor([0]),
        False,
        2,
    )

    residual, retained = _residual_target_and_addition(original, snapshot)

    assert torch.equal(retained, addition)
    assert torch.equal(residual + retained, original)


def test_placement_parser_defaults_to_selected_post_refit_protocol() -> None:
    args = _parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--run-output",
            "run",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
        ]
    )

    assert args.blocks == (0, 12, 24)
    assert args.fit_tokens == 8192
    assert args.left_steps == 32
    assert args.right_batches == 16
    assert args.refine_groups == ("gate", "o", "qkv", "up")
    assert args.full_only is False
    assert args.unit_arms is False
    assert args.direct_frozen_wikitext is False
    assert args.direct_unit_arms is False
