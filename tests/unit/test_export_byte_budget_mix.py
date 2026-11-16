import gguf
import numpy as np
import pytest
from export_byte_budget_mix import export_mix


def write(writer, path):
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def sources(tmp_path):
    source = tmp_path / "staged.gguf"
    standard = tmp_path / "standard.gguf"
    writer = gguf.GGUFWriter(None, "gemma3")
    writer.add_uint32("quantization.nanoquant.tensor_count", 2)
    for name in ("head_count", "head_count_kv"):
        writer.add_uint32(f"gemma3.attention.{name}", 1)
    for name in ("key_length", "value_length"):
        writer.add_uint32(f"gemma3.attention.{name}", 32)
    # The project's Gemma exporter uses val_length in its grouped tensor metadata.
    writer.add_uint32("gemma3.attention.val_length", 32)
    embedding = np.arange(256, dtype=np.float32).reshape(8, 32)
    writer.add_tensor("token_embd.weight", embedding)
    for owner, rows in (("blk.0.attn_qkv", 96), ("blk.0.ffn_gate", 64)):
        writer.add_uint32(f"quantization.nanoquant.{owner}.residual_stage_count", 1)
        for stage in ("nq_stage0", "nq_residual_0"):
            for suffix, shape in (("v", (32, 1)), ("u", (rows, 1))):
                writer.add_tensor(f"{owner}.{stage}_{suffix}", np.zeros(shape, dtype=np.int32),
                                  raw_dtype=gguf.GGMLQuantizationType.I32)
            for suffix, size in (("scale_pre", 32), ("scale_mid", 32), ("scale_post", rows)):
                writer.add_tensor(f"{owner}.{stage}_{suffix}", np.ones(size, dtype=np.float32))
    write(writer, source)
    writer = gguf.GGUFWriter(None, "gemma3")
    members = [f"blk.0.attn_{kind}.weight" for kind in ("q", "k", "v")]
    byte_counts = {}
    for index, member in enumerate(members):
        values = np.full((32, 32), index + 1.0, dtype=np.float32)
        data = gguf.quantize(values, gguf.GGMLQuantizationType.Q8_0)
        writer.add_tensor(member, data, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        byte_counts[member] = data.nbytes
    write(writer, standard)
    units = {
        "blk.0.attn_qkv": {"members": members,
                           "standard": {name: {"Q8_0": {"bytes": byte_counts[name]}} for name in members}},
        "blk.0.ffn_gate": {"members": ["blk.0.ffn_gate.weight"], "standard": {}},
    }
    return source, standard, units, embedding


def test_replaces_whole_qkv_group_truncates_stages_and_preserves_embedding(tmp_path):
    source, standard, units, embedding = sources(tmp_path)
    output = tmp_path / "mixed.gguf"
    choices = {"blk.0.attn_qkv": "Q8_0 Q8_0 Q8_0", "blk.0.ffn_gate": "1 stage"}
    receipt = export_mix(source, [standard], units, choices, output, 1_000_000)
    assert receipt["standard_units"] == receipt["binary_units"] == 1
    reader = gguf.GGUFReader(output)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    assert np.array_equal(tensors["token_embd.weight"].data, embedding)
    assert not any("nq_residual" in name or "attn_qkv" in name for name in tensors)
    assert reader.fields["quantization.nanoquant.blk.0.ffn_gate.residual_stage_count"].contents() == 0
    assert all(f"blk.0.attn_{kind}.weight" in tensors for kind in ("q", "k", "v"))


def test_rejects_partial_group_or_over_budget_without_publishing(tmp_path):
    source, standard, units, _ = sources(tmp_path)
    output = tmp_path / "mixed.gguf"
    choices = {"blk.0.attn_qkv": "Q8_0", "blk.0.ffn_gate": "2 stages"}
    with pytest.raises(ValueError, match="every group member"):
        export_mix(source, [standard], units, choices, output, 1_000_000)
    choices["blk.0.attn_qkv"] = "Q8_0 Q8_0 Q8_0"
    with pytest.raises(ValueError, match="exceeds byte budget"):
        export_mix(source, [standard], units, choices, output, 1)
    assert not output.exists()
