import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nanoquant.runtime.artifact import RuntimeModelMetadata, open_logical_artifact, write_logical_artifact
from nanoquant.runtime.backend import QuantizedLinearSpec
from nanoquant.runtime.llamacpp import export_llamacpp_checkpoint, open_llamacpp_checkpoint
from nanoquant.runtime.logical import LogicalLayerState
from nanoquant.runtime.packed_artifact import (
    convert_logical_to_packed,
    open_packed_artifact,
    validate_packed_conversion,
)
from nanoquant.runtime.reference import DenseReferenceBackend, FactorizedReferenceBackend, PackedReferenceBackend
from nanoquant.runtime.residual import stage_spec


def _state() -> LogicalLayerState:
    spec = QuantizedLinearSpec(
        "blocks.0.mlp.down_proj",
        "nanoquant-residual-v2",
        6,
        4,
        4,
        "float32",
        "float32",
        has_bias=True,
        residual_ranks=(3, 2),
    )
    generator = torch.Generator().manual_seed(17)

    def term(term_spec):
        rank = term_spec.rank
        return LogicalLayerState(
            term_spec,
            torch.randint(0, 2, (4, rank), generator=generator).float() * 2 - 1,
            torch.randint(0, 2, (rank, 6), generator=generator).float() * 2 - 1,
            torch.rand(6, generator=generator),
            torch.rand(rank, generator=generator),
            torch.rand(4, generator=generator),
        )

    primary = term(replace(spec, logical_format="nanoquant-v1", has_bias=False, residual_ranks=()))
    return replace(
        primary,
        spec=spec,
        bias=torch.ones(4),
        residual_stages=tuple(term(stage_spec(spec, index)) for index in range(2)),
    )


def test_logical_and_packed_artifacts_roundtrip_all_terms_and_reject_missing_inventory(tmp_path: Path):
    state = _state()
    logical = write_logical_artifact(
        tmp_path / "logical", RuntimeModelMetadata("fixture", "pinned", "gemma3", "c", "t"), {0: (state,)}
    )
    assert logical.manifest.minimum_runtime_version == "0.2.0"
    assert logical.manifest.logical_format == "nanoquant-residual-v2"
    assert len(logical.manifest.blocks[0].layers[0].tensors) == 16
    convert_logical_to_packed(logical.root, tmp_path / "packed")
    packed = open_packed_artifact(tmp_path / "packed")
    assert packed.manifest.minimum_runtime_version == "0.2.0"
    assert validate_packed_conversion(logical.root, packed.root).exact
    inputs = torch.randn(7, 6)
    expected = (
        inputs
        @ sum(
            (term.left_binary * term.scale_post[:, None])
            @ (term.right_binary * term.scale_mid[:, None] * term.scale_pre[None, :])
            for term in (state,) + state.residual_stages
        ).T
        + state.bias
    )
    for backend in (DenseReferenceBackend(), FactorizedReferenceBackend(), PackedReferenceBackend()):
        source = (
            packed.load_layer(state.spec.name)
            if isinstance(backend, PackedReferenceBackend)
            else logical.load_layer(state.spec.name)
        )
        torch.testing.assert_close(backend.linear(inputs, backend.prepare(source, "cpu")), expected)
    with pytest.raises(ValueError, match="inventory"):
        replace(state, residual_stages=())
    descriptor = logical.root / "nanoquant-model.json"
    payload = json.loads(descriptor.read_text())
    payload["blocks"][0]["layers"][0]["tensors"].pop()
    descriptor.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="inventory"):
        open_logical_artifact(logical.root)


def test_checkpoint_explicitly_versions_residual_stages(tmp_path: Path):
    state = replace(_state(), bias=None, spec=replace(_state().spec, has_bias=False))
    logical = write_logical_artifact(
        tmp_path / "logical", RuntimeModelMetadata("fixture", "pinned", "gemma3", "c", "t"), {0: (state,)}
    )
    convert_logical_to_packed(logical.root, tmp_path / "packed")
    manifest = export_llamacpp_checkpoint(tmp_path / "packed", tmp_path / "checkpoint")
    assert manifest.schema_version == 2
    assert manifest.tensor_count == 22  # Three complete factor terms plus an explicit stage count.
    assert open_llamacpp_checkpoint(tmp_path / "checkpoint") == manifest


@pytest.mark.cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 7])
def test_cuda_executes_independent_terms_with_existing_packed_kernel(dtype, tokens):
    from nanoquant.infrastructure.device_lease import acquire_device_lease
    from nanoquant.runtime.cuda_backend import CudaPackedBackend
    from nanoquant.runtime.packed import pack_logical_layer

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = QuantizedLinearSpec(
        "blocks.0.mlp.down_proj", "nanoquant-residual-v2", 96, 64, 64, "float32", "bfloat16", residual_ranks=(32,)
    )
    generator = torch.Generator().manual_seed(83)

    def term(term_spec):
        return LogicalLayerState(
            term_spec,
            torch.randint(0, 2, (64, term_spec.rank), generator=generator).float() * 2 - 1,
            torch.randint(0, 2, (term_spec.rank, 96), generator=generator).float() * 2 - 1,
            torch.rand(96, generator=generator).to(torch.bfloat16),
            torch.rand(term_spec.rank, generator=generator).mul(0.01).to(torch.bfloat16),
            torch.rand(64, generator=generator).to(torch.bfloat16),
        )

    primary = term(replace(spec, logical_format="nanoquant-v1", residual_ranks=()))
    state = replace(primary, spec=spec, residual_stages=(term(stage_spec(spec, 0)),))
    with acquire_device_lease("cuda"):
        inputs = torch.randn(tokens, 96, device="cuda", dtype=dtype)
        backend = CudaPackedBackend()
        actual = backend.linear(inputs, backend.prepare(pack_logical_layer(state), "cuda"))
        reference = FactorizedReferenceBackend()
        expected = reference.linear(inputs, reference.prepare(state, "cuda"))
        torch.testing.assert_close(
            actual,
            expected.float(),
            rtol=0.025 if dtype is torch.bfloat16 else 1e-5,
            atol=0.004 if dtype is torch.bfloat16 else 2e-6,
        )
