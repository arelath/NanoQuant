import importlib
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors.torch import save_file


def test_share_threshold_and_compact_export(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "tools"))
    tool = importlib.import_module("benchmark_imatrix_q8_hybrid")
    selected = tool.selected_projections({"projections": [
        {"block": 0, "projection": "mlp.up_proj", "share": 0.01001},
        {"block": 1, "projection": "mlp.up_proj", "share": 0.01},
    ]}, 0.01)
    assert [item["tensor"] for item in selected] == ["blk.0.ffn_up.weight"]

    staged = tmp_path / "staged.gguf"
    writer = gguf.GGUFWriter(staged, "gemma3")
    writer.add_uint32("quantization.nanoquant.tensor_count", 2)
    writer.add_uint32("quantization.nanoquant.blk.0.ffn_up.residual_stage_count", 1)
    writer.add_tensor("blk.0.ffn_up.nq_stage0_v", np.arange(32, dtype=np.int32))
    unselected = np.arange(32, dtype=np.int32)[::-1].copy()
    writer.add_tensor("blk.1.ffn_up.nq_stage0_v", unselected)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()

    bf16 = torch.linspace(-1, 1, 64).reshape(2, 32).bfloat16()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"model.layers.0.mlp.up_proj.weight": bf16}, checkpoint)
    hybrid = tmp_path / "hybrid.gguf"
    tool.export_hybrid(staged, checkpoint, hybrid, selected)

    reader = gguf.GGUFReader(hybrid)
    assert reader.fields["quantization.nanoquant.tensor_count"].contents() == 1
    assert "quantization.nanoquant.blk.0.ffn_up.residual_stage_count" not in reader.fields
    tensors = {t.name: t for t in reader.tensors}
    assert set(tensors) == {"blk.0.ffn_up.weight", "blk.1.ffn_up.nq_stage0_v"}
    assert np.array_equal(tensors["blk.1.ffn_up.nq_stage0_v"].data, unselected)
    q8 = tensors["blk.0.ffn_up.weight"]
    assert q8.tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert np.array_equal(q8.data, gguf.quantize(bf16.float().numpy(), q8.tensor_type))
