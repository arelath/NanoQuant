from __future__ import annotations

from pathlib import Path

import torch

from nanoquant.rank_expansion_workflow import _target_rank, _verify_derivative
from nanoquant.runtime import (
    LogicalLayerState,
    QuantizedLinearSpec,
    RuntimeModelMetadata,
    open_packed_artifact,
    pack_logical_layer,
    write_packed_artifact,
)


def _logical(name: str, rank: int, *, seed: int) -> LogicalLayerState:
    generator = torch.Generator().manual_seed(seed)
    left = torch.sign(torch.randn((128, rank), generator=generator))
    right = torch.sign(torch.randn((rank, 128), generator=generator))
    return LogicalLayerState(
        QuantizedLinearSpec(name, "nanoquant-v1", 128, 128, rank, "float32", "bfloat16"),
        left,
        right,
        torch.ones(128, dtype=torch.bfloat16),
        torch.ones(rank, dtype=torch.bfloat16),
        torch.ones(128, dtype=torch.bfloat16),
    )


def test_target_rank_meets_requested_packed_bit_multiplier() -> None:
    state = pack_logical_layer(_logical("blocks.0.self_attn.v_proj", 32, seed=1))

    target, old_bits, target_bits = _target_rank(state, 1.30, 32)

    assert target > state.spec.rank
    assert target % 32 == 0
    assert target_bits >= old_bits * 1.30


def test_derivative_verification_requires_exact_non_target_and_target_prefixes(tmp_path: Path) -> None:
    model = RuntimeModelMetadata("fixture/model", "revision", "gemma3", "config", "tokenizer")
    old_v = _logical("blocks.0.self_attn.v_proj", 32, seed=2)
    non_target = _logical("blocks.0.self_attn.q_proj", 32, seed=3)
    source = write_packed_artifact(
        tmp_path / "source",
        model,
        "a" * 64,
        {0: (pack_logical_layer(old_v), pack_logical_layer(non_target))},
    )
    added = _logical("blocks.0.self_attn.v_proj", 32, seed=4)
    expanded_v = LogicalLayerState(
        QuantizedLinearSpec(
            old_v.spec.name,
            old_v.spec.logical_format,
            old_v.spec.in_features,
            old_v.spec.out_features,
            64,
            old_v.spec.factor_dtype,
            old_v.spec.scale_dtype,
        ),
        torch.cat((old_v.left_binary, added.left_binary), dim=1),
        torch.cat((old_v.right_binary, added.right_binary), dim=0),
        old_v.scale_pre,
        torch.cat((old_v.scale_mid, torch.zeros(32, dtype=torch.bfloat16))),
        old_v.scale_post,
    )
    output = write_packed_artifact(
        tmp_path / "output",
        model,
        "b" * 64,
        {0: (pack_logical_layer(expanded_v), pack_logical_layer(non_target))},
    )

    assert _verify_derivative(
        open_packed_artifact(source.root),
        open_packed_artifact(output.root),
        "self_attn.v_proj",
    ) == (1, 1)
