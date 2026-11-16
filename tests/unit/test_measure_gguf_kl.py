from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

REPOSITORY = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    tools = str(REPOSITORY / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("measure_gguf_kl")


def _write_gguf(path: Path, tensors: dict[str, tuple[np.ndarray, str]]) -> None:
    from gguf import GGMLQuantizationType, GGUFWriter
    from gguf.quants import quantize

    writer = GGUFWriter(str(path), "gemma3")
    for name, (values, kind) in tensors.items():
        qtype = GGMLQuantizationType[kind]
        data = values if qtype == GGMLQuantizationType.F32 else quantize(values, qtype)
        writer.add_tensor(name, data, raw_dtype=qtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_read_gguf_dequantizes_q8_and_maps_projection_embedding_and_norm_names(tmp_path) -> None:
    tool = _tool()
    generator = np.random.default_rng(0)
    q = generator.standard_normal((64, 32)).astype(np.float32)
    path = tmp_path / "tiny.gguf"
    _write_gguf(path, {
        "blk.0.attn_q.weight": (q, "Q8_0"),
        "blk.0.attn_norm.weight": (np.ones(32, dtype=np.float32), "F32"),
        "token_embd.weight": (generator.standard_normal((10, 32)).astype(np.float32), "F32"),
    })
    tensors = tool.read_gguf(path)

    stored = tensors["blk.0.attn_q.weight"]
    assert stored["type"] == "Q8_0" and stored["elements"] == q.size and stored["bytes"] == 64 * 34
    assert tuple(stored["values"].shape) == (64, 32)
    assert tool.relative_error(torch.from_numpy(q), stored["values"]) < 0.01
    projections, embedding, norms = tool.map_tensors(tensors, ["0:self_attn.q_proj"])
    assert set(projections) == {"0:self_attn.q_proj"} and set(norms) == {"0:input_layernorm"}
    assert embedding["type"] == "F32"
    assert tool._bits(projections.values()) == pytest.approx(8.5)
    with pytest.raises(ValueError, match="do not match the sweep targets"):
        tool.map_tensors(tensors, ["0:self_attn.q_proj", "0:mlp.up_proj"])
    with pytest.raises(ValueError, match="shape"):
        tool.relative_error(torch.zeros(32, 64), stored["values"])


def _pack_signs(signs: np.ndarray) -> np.ndarray:
    rows, columns = signs.shape
    padded = np.zeros((rows, -(-columns // 32) * 32), dtype=np.uint64)
    padded[:, :columns] = signs < 0
    words = (padded.reshape(rows, -1, 32) << np.arange(32, dtype=np.uint64)).sum(axis=-1)
    return words.astype(np.uint32).view(np.int32)


def test_read_gguf_rebuilds_stacked_two_stage_nanoquant_qkv(tmp_path) -> None:
    from gguf import GGUFWriter

    tool = _tool()
    generator = np.random.default_rng(1)
    rank, out_features, in_features = 3, 6, 40
    writer = GGUFWriter(str(tmp_path / "nq.gguf"), "gemma3")
    writer.add_string("quantization.nanoquant.stage_format", "additive_binary_v1")
    writer.add_uint32("quantization.nanoquant.blk.0.attn_qkv.residual_stage_count", 1)
    for key, value in {"head_count": 1, "head_count_kv": 1, "key_length": 2, "value_length": 2}.items():
        writer.add_uint32(f"gemma3.attention.{key}", value)
    expected = np.zeros((out_features, in_features), dtype=np.float32)
    for term in ("nq_stage0", "nq_residual_0"):
        left = generator.choice([-1.0, 1.0], (out_features, rank)).astype(np.float32)
        right = generator.choice([-1.0, 1.0], (rank, in_features)).astype(np.float32)
        scales = {name: generator.uniform(0.5, 2, size).astype(np.float32)
                  for name, size in (("pre", in_features), ("mid", rank), ("post", out_features))}
        expected += (left * scales["post"][:, None]) @ (right * scales["mid"][:, None] * scales["pre"][None, :])
        writer.add_tensor(f"blk.0.attn_qkv.{term}_u", _pack_signs(left))
        writer.add_tensor(f"blk.0.attn_qkv.{term}_v", _pack_signs(right))
        for name, values in scales.items():
            writer.add_tensor(f"blk.0.attn_qkv.{term}_scale_{name}", values)
    writer.add_tensor("token_embd.weight", generator.standard_normal((5, in_features)).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    group_bytes = 2 * (out_features * 4 + rank * 2 * 4 + (in_features + rank + out_features) * 4)

    tensors = tool.read_gguf(tmp_path / "nq.gguf")
    assert set(tensors) == {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight", "token_embd.weight"}
    projections, embedding, _ = tool.map_tensors(
        tensors, ["0:self_attn.q_proj", "0:self_attn.k_proj", "0:self_attn.v_proj"],
    )
    rebuilt = torch.cat([projections[f"0:self_attn.{name}_proj"]["values"] for name in "qkv"])
    torch.testing.assert_close(rebuilt, torch.from_numpy(expected))
    assert {item["type"] for item in projections.values()} == {"NanoQuant 2-stage binary"}
    assert sum(item["bytes"] for item in projections.values()) == pytest.approx(group_bytes)
    assert tool._bits(projections.values()) == pytest.approx(8 * group_bytes / expected.size)
    assert embedding["type"] == "F32"


def test_norm_check_expects_gemmas_stored_plus_one() -> None:
    tool = _tool()
    layer = nn.Module()
    layer.input_layernorm = nn.Module()
    layer.input_layernorm.weight = nn.Parameter(torch.tensor([0.5, -0.25]))
    adapter = SimpleNamespace(get_decoder_layers=lambda _model: [layer])
    norms = {"0:input_layernorm": {"values": torch.tensor([1.5, 0.75])}}
    assert tool.check_norms(None, adapter, norms) == 0
    norms["0:input_layernorm"]["values"] = torch.tensor([0.5, -0.25])
    assert tool.check_norms(None, adapter, norms) == pytest.approx(1)


def test_additive_prediction_counts_extrapolated_projections() -> None:
    tool = _tool()
    curves = {name: {"coefficient": 2.0, "exponent": 2.0, "measured_error_range": [0.01, 0.35]}
              for name in ("a", "b")}
    total, outside = tool.additive_prediction(curves, {"a": 0.1, "b": 0.005})
    assert total == pytest.approx(2 * 0.1**2 + 2 * 0.005**2)
    assert outside == 1


def test_markdown_lists_each_file_and_the_prediction_ratio() -> None:
    tool = _tool()
    variant = {"kl_above_control": 0.2, "nll_above_control": 0.0488, "perplexity_ratio": 1.05}
    report = {
        "protocol": {"teacher": {"source": "test/model", "model_dtype": "torch.float32"}},
        "measured_at": "now", "measurement_sweep": "sweep", "measured_error_range": [0.01, 0.35],
        "clean": {"token_count": 12, "kl_nats_per_token": 0.0, "nll_nats_per_token": 3.0},
        "files": [{
            "file": "m-Q4.gguf", "bytes": 2_000_000, "projection_bits_per_weight": 4.5,
            "embedding_type": "Q8_0", "embedding_bits_per_weight": 8.5,
            "variants": {"all weights": variant, "projections": {**variant, "kl_above_control": 0.1},
                         "embedding": {**variant, "kl_above_control": 0.01}},
            "projection_types": {"Q4_K": 3, "Q6_K": 1}, "median_projection_error": 0.05,
            "projection_error_range": [0.02, 0.09], "additive_prediction": 0.4,
            "projections_outside_measured_range": 0, "sha256": "abc", "embedding_error": 0.007,
            "norm_max_abs_difference": 0.0,
        }],
        "joint_reference": {"source": "joint.json", "runs": 100,
                            "rows": [{"kl_target": 0.001, "median_error": 0.03, "mean": 0.12,
                                      "interval": [0.11, 0.13]}]},
        "json": "gguf-kl.json",
    }
    markdown = tool.render_markdown(report)
    assert "| m-Q4.gguf | 2.0 | 4.50 | Q8_0 (8.50 bits) | 0.2000 | +0.0488 | +5.0% | 0.1000 | 0.0100 |" in markdown
    for header in (
        "All weights KL (nats/token)", "All weights NLL change (nats/token)", "Projections-only KL (nats/token)",
        "Embedding-only KL (nats/token)", "Additive prediction (nats/token)",
        "Measured projections-only KL (nats/token)", "KL target per projection (nats/token)",
        "Mean joint KL (nats/token)", "95% CI (nats/token)",
    ):
        assert f" {header} |" in markdown
    assert "| m-Q4.gguf | Q4_K ×3, Q6_K ×1 | 5.00% | 2.00%–9.00% | 0.4000 | 0.1000 | 0.250 | 0 |" in markdown
    assert "| 0.001 | 3.00% | 0.1200 | 0.1100 – 0.1300 |" in markdown


def test_swapped_replaces_weights_only_inside_the_block() -> None:
    tool = _tool()
    weight = nn.Parameter(torch.zeros(2, 3))
    with tool.swapped({"w": (weight, torch.ones(2, 3))}):
        assert weight.detach().sum() == 6
    assert weight.detach().sum() == 0
    with pytest.raises(RuntimeError, match="boom"), tool.swapped({"w": (weight, torch.ones(2, 3))}):
        raise RuntimeError("boom")
    assert weight.detach().sum() == 0
