from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from torch.nn import functional as F

from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.kl_splice import (
    SpliceReconstruction,
    SpliceReconstructionSet,
)


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_corrected_codebook_splice")


def test_splice_probe_selects_blocks_without_copying_other_layers() -> None:
    probe = _probe_module()
    first = LayerId(BlockId(0), "mlp.down_proj")
    second = LayerId(BlockId(2), "mlp.down_proj")
    reconstructions = SpliceReconstructionSet(
        (
            SpliceReconstruction(first, torch.ones((1, 1)), None, 1.0),
            SpliceReconstruction(second, torch.ones((1, 1)), None, 2.0),
        ),
        (
            ("0:mlp.down_proj", (first,)),
            ("2:mlp.down_proj", (second,)),
        ),
        (
            ("0:mlp.down_proj", 1.0),
            ("2:mlp.down_proj", 2.0),
        ),
    )

    selected = probe._select_blocks(reconstructions, (2,))

    assert tuple(item.layer for item in selected.layers) == (second,)
    assert selected.unit_members == (("2:mlp.down_proj", (second,)),)


def test_splice_probe_selects_a_disjoint_token_window() -> None:
    probe = _probe_module()
    tokens = torch.arange(30).reshape(10, 3)

    selected = probe._select_token_window(tokens, offset=4, samples=3)

    assert torch.equal(selected, tokens[4:7])
    with pytest.raises(ValueError, match="shorter"):
        probe._select_token_window(tokens, offset=9, samples=2)


def test_splice_probe_accepts_multiple_projection_layers_per_block() -> None:
    probe = _probe_module()
    gate = LayerId(BlockId(2), "mlp.gate_proj")
    up = LayerId(BlockId(2), "mlp.up_proj")
    reconstructions = SpliceReconstructionSet(
        (
            SpliceReconstruction(gate, torch.ones((1, 1)), None, 1.0),
            SpliceReconstruction(up, torch.ones((1, 1)), None, 2.0),
        ),
        (
            ("2:mlp.gate_proj", (gate,)),
            ("2:mlp.up_proj", (up,)),
        ),
        (
            ("2:mlp.gate_proj", 1.0),
            ("2:mlp.up_proj", 2.0),
        ),
    )

    selected = probe._select_blocks(reconstructions, (2,))

    assert tuple(item.layer for item in selected.layers) == (gate, up)
    assert probe._parse_projections("gate,up") == ("gate", "up")


def test_splice_probe_resolves_projection_specific_free_rows() -> None:
    probe = _probe_module()

    parsed = probe._parse_projection_free_rows("gate:640,up:672")

    assert parsed == (("gate", 640), ("up", 672))
    assert probe._resolve_projection_free_rows(
        ("gate", "up"), 0, parsed
    ) == {"gate": 640, "up": 672}
    with pytest.raises(ValueError, match="every requested projection"):
        probe._resolve_projection_free_rows(("gate", "up"), 0, parsed[:1])


def test_splice_probe_composes_gated_down_outputs() -> None:
    probe = _probe_module()
    inputs = torch.tensor([[1.0, -0.5], [0.25, 2.0]])
    gate_weight = torch.tensor([[0.5, 1.0], [-1.0, 0.25]])
    up_weight = torch.tensor([[1.0, -0.25], [0.5, 0.75]])
    down_weight = torch.tensor([[0.75, -0.5], [0.25, 1.0]])

    observed = probe._gated_down_outputs(
        inputs,
        gate_weight,
        up_weight,
        down_weight,
        device="cpu",
    )
    gated = F.silu(F.linear(inputs, gate_weight)) * F.linear(
        inputs,
        up_weight,
    )
    expected = F.linear(gated, down_weight)

    assert torch.allclose(observed.float(), expected, atol=2e-2, rtol=2e-2)


def test_splice_probe_composes_per_block_downstream_policy() -> None:
    probe = _probe_module()
    first = LayerId(BlockId(0), "mlp.down_proj")
    second = LayerId(BlockId(12), "mlp.down_proj")
    third = LayerId(BlockId(16), "mlp.down_proj")

    def arm(
        first_value: float,
        second_value: float,
        third_value: float,
    ) -> SpliceReconstructionSet:
        return SpliceReconstructionSet(
            (
                SpliceReconstruction(
                    first,
                    torch.tensor([[first_value]]),
                    None,
                    1.0,
                ),
                SpliceReconstruction(
                    second,
                    torch.tensor([[second_value]]),
                    None,
                    1.0,
                ),
                SpliceReconstruction(
                    third,
                    torch.tensor([[third_value]]),
                    None,
                    1.0,
                ),
            ),
            (("0", (first,)), ("12", (second,)), ("16", (third,))),
            (("0", 1.0), ("12", 1.0), ("16", 1.0)),
        )

    sets = {}
    for prefix in ("free_words", "corrected_codebook"):
        offset = 0.0 if prefix == "free_words" else 10.0
        sets[prefix] = arm(0.0 + offset, 0.0 + offset, 0.0 + offset)
        sets[f"{prefix}_operator_refit"] = arm(
            1.0 + offset,
            1.0 + offset,
            1.0 + offset,
        )
        sets[f"{prefix}_operator_downstream_input_refit"] = arm(
            2.0 + offset,
            2.0 + offset,
            2.0 + offset,
        )
        sets[f"{prefix}_operator_downstream_joint_refit"] = arm(
            3.0 + offset,
            3.0 + offset,
            3.0 + offset,
        )

    result = probe._downstream_policy_sets(
        sets,
        probe._parse_block_policy("0:joint,12:input,16:base"),
    )

    policy = result["corrected_codebook_operator_policy_refit"]
    assert [float(item.weight.item()) for item in policy.layers] == [
        13.0,
        12.0,
        10.0,
    ]

    hybrid = probe._hybrid_representation_set(
        result,
        probe._parse_representation_policy("0:mixed,12:free,16:mixed"),
    )
    assert [float(item.weight.item()) for item in hybrid.layers] == [
        13.0,
        2.0,
        10.0,
    ]


def test_splice_probe_exports_hashed_reconstruction_set(tmp_path: Path) -> None:
    probe = _probe_module()
    first = LayerId(BlockId(2), "mlp.gate_proj")
    second = LayerId(BlockId(2), "mlp.up_proj")
    reconstructions = SpliceReconstructionSet(
        (
            SpliceReconstruction(
                first,
                torch.tensor([[1.0, 2.0]]),
                None,
                1.0,
            ),
            SpliceReconstruction(
                second,
                torch.tensor([[3.0, 4.0]]),
                None,
                1.0,
            ),
        ),
        (("2:gate", (first,)), ("2:up", (second,))),
        (("2:gate", 1.0), ("2:up", 1.0)),
    )

    destination = tmp_path / "export"
    receipt = probe._export_reconstruction_set(
        destination,
        "hybrid",
        reconstructions,
    )

    manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )
    assert receipt["arm"] == "hybrid"
    assert manifest["layer_count"] == 2
    assert manifest["blocks"] == [2]
    assert manifest["tensor_sha256"] == hash_file(
        destination / "weights.safetensors"
    )
    with safe_open(
        destination / "weights.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        assert set(handle.keys()) == {
            "model.layers.2.mlp.gate_proj.weight",
            "model.layers.2.mlp.up_proj.weight",
        }
        assert handle.get_tensor(
            "model.layers.2.mlp.gate_proj.weight"
        ).dtype == torch.bfloat16


def test_splice_probe_reconstruction_identity_tracks_numerical_settings() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
        ]
    )

    first = probe._reconstruction_cache_identity(
        args,
        model_sha256="model",
        calibration_manifest_sha256="manifest",
        calibration_state_sha256="state",
        block=2,
        projection="down",
        projection_path="mlp.down_proj",
        transposed=False,
        factorization_shape=(2, 3),
        rank=32,
    )
    args.outer_iterations += 1
    second = probe._reconstruction_cache_identity(
        args,
        model_sha256="model",
        calibration_manifest_sha256="manifest",
        calibration_state_sha256="state",
        block=2,
        projection="down",
        projection_path="mlp.down_proj",
        transposed=False,
        factorization_shape=(2, 3),
        rank=32,
    )

    assert first != second


def test_splice_probe_parses_no_flip_product_codebook_mode() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
            "--codebook-mode",
            "product",
            "--corrections-per-word",
            "0",
        ]
    )

    assert args.codebook_mode == "product"
    assert args.corrections_per_word == 0


def test_splice_probe_parses_fixed_outlier_indices() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
            "--fixed-outlier-indices",
            "50,1791,3043",
        ]
    )

    assert args.fixed_outlier_indices == (50, 1791, 3043)


def test_splice_probe_allows_fixed_outliers_for_joint_projections() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
            "--projections",
            "gate,up",
            "--transpose-matrix",
            "--fixed-outlier-indices",
            "768,890",
        ]
    )

    assert args.projections == ("gate", "up")
    assert args.transpose_matrix
    assert args.fixed_outlier_indices == (768, 890)


def test_splice_probe_parses_no_flip_linear_codebook_mode() -> None:
    probe = _probe_module()
    args = probe._parser().parse_args(
        [
            "--model",
            "model.safetensors",
            "--snapshot",
            "snapshot",
            "--calibration-state",
            "calibration",
            "--output",
            "result.json",
            "--codebook-mode",
            "linear",
            "--corrections-per-word",
            "0",
            "--linear-assignment-sweeps",
            "3",
        ]
    )

    assert args.codebook_mode == "linear"
    assert args.corrections_per_word == 0
    assert args.linear_assignment_sweeps == 3
