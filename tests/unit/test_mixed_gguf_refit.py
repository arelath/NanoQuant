import gguf
import numpy as np
import pytest
import torch
from refit_mixed_gguf import block_owners, export_scales, scale_payloads, scale_state

from nanoquant.application.layers import TrainableSharedInputFactorGroup
from nanoquant.application.tuning import TuningRequest, post_block_refit


def write_fixture(path, kind="ffn_gate"):
    writer = gguf.GGUFWriter(None, "gemma3")
    base = f"blk.0.{kind}"
    writer.add_uint32(f"quantization.nanoquant.{base}.residual_stage_count", 1)
    writer.add_tensor("token_embd.weight", np.arange(256, dtype=np.float32).reshape(8, 32))
    for term in ("nq_stage0", "nq_residual_0"):
        writer.add_tensor(f"{base}.{term}_u", np.zeros((8, 1), dtype=np.int32))
        writer.add_tensor(f"{base}.{term}_v", np.zeros((32, 1), dtype=np.int32))
        for axis, size in (("pre", 32), ("mid", 32), ("post", 8)):
            values = gguf.quantize(np.full(size, .1, dtype=np.float32), gguf.GGMLQuantizationType.BF16)
            writer.add_tensor(f"{base}.{term}_scale_{axis}", values, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.write_header_to_file(path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=False)
    writer.close()


def test_multistage_scale_refit_export_preserves_signs_embedding_and_size(tmp_path):
    source, output = tmp_path / "source.gguf", tmp_path / "refit.gguf"
    write_fixture(source)
    reader = gguf.GGUFReader(source)
    tensors = {t.name: t for t in reader.tensors}
    owners = block_owners(reader, tensors, 0)
    owner = owners["blk.0.ffn_gate"][0]
    assert len(owner.residual_terms) == 1
    values = scale_state(owners)
    assert len(values) == 6
    key = "blk.0.ffn_gate.nq_residual_0_scale_post"
    values[key] = values[key] * 2
    receipt = export_scales(source, output, values)
    assert receipt["changed_scale_tensors"] == 1
    assert receipt["signs_exact"] and receipt["embedding_exact"]
    assert output.stat().st_size == source.stat().st_size
    actual = {t.name: t for t in gguf.GGUFReader(output).tensors}
    for name, original in tensors.items():
        if name != key:
            assert np.array_equal(actual[name].data, original.data)
    np.testing.assert_array_equal(
        gguf.dequantize(actual[key].data, actual[key].tensor_type), values[key].float().numpy(),
    )
    with pytest.raises(ValueError, match="preserving"):
        export_scales(source, output, values)


def test_post_refit_excludes_dense_replacement_and_shared_coupled_scales():
    torch.manual_seed(0)
    group = TrainableSharedInputFactorGroup(torch.ones(3, 2), torch.ones(2, 4),
                                           torch.ones(4), torch.ones(2), torch.ones(3),
                                           immutable_binary_factors=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), group)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    inputs = torch.randn(8, 4)
    request = TuningRequest(inputs, torch.zeros(8, 3), 3, 4, .01)
    post_block_refit(model, request, lambda module, x: module(x))
    for name, parameter in model.named_parameters():
        if name != "1.scale_post":
            assert torch.equal(before[name], parameter), name
    assert not torch.equal(before["1.scale_post"], group.scale_post)


def test_shared_residual_stages_keep_coupled_scale_policy(tmp_path):
    source = tmp_path / "group.gguf"
    write_fixture(source, "attn_qkv")
    reader = gguf.GGUFReader(source)
    owners = block_owners(reader, {t.name: t for t in reader.tensors}, 0)
    owner = owners["blk.0.attn_qkv"][0]
    assert owner.coupled_shared_input_scales
    assert owner.residual_terms[0].coupled_shared_input_scales


def test_scale_overlay_rejects_nonfinite_values(tmp_path):
    source = tmp_path / "source.gguf"
    write_fixture(source)
    tensors = {t.name: t for t in gguf.GGUFReader(source).tensors}
    with pytest.raises(ValueError, match="invalid scale"):
        scale_payloads(tensors, {"blk.0.ffn_gate.nq_stage0_scale_pre": torch.full((32,), float("nan"))})
