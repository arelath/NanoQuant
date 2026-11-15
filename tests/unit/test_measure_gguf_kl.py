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
    variant = {"kl_above_control": 0.2, "perplexity_ratio": 1.05}
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
    assert "| m-Q4.gguf | 2.0 | 4.50 | Q8_0 (8.50 bits) | 0.2000 | +5.0% | 0.1000 | 0.0100 |" in markdown
    assert "| m-Q4.gguf | Q4_K ×3, Q6_K ×1 | 5.00% | 2.00%–9.00% | 0.4000 | 0.1000 | 0.250 | 0 |" in markdown
    assert "| 0.001 | 3.00% | 0.1200 | 0.1100 – 0.1300 |" in markdown
