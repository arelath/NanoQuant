from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from nanoquant.runtime import (
    GGUF_TENSOR_SUFFIXES,
    PACKED_LAYOUT_VERSION,
    PACKED_REFERENCE_COMMIT,
    PACKED_REFERENCE_CUDA_SHA256,
    PACKED_TENSOR_NAMESPACE,
    LogicalLayerState,
    PackedLayerState,
    PackedLayoutMetadata,
    QuantizedLinearSpec,
    pack_logical_layer,
    pack_sign_matrix,
    packed_row_stride_bytes,
    packed_word_count,
    supports_vector_word_loads,
    unpack_sign_matrix,
)


@pytest.mark.parametrize("columns", (1, 31, 32, 33, 65, 128))
def test_llamacpp_sign_words_roundtrip_arbitrary_tail_width(columns: int) -> None:
    values = torch.where(
        torch.arange(3 * columns).reshape(3, columns) % 3 == 0,
        -torch.ones(3, columns),
        torch.ones(3, columns),
    )

    packed = pack_sign_matrix(values)

    assert packed.dtype == torch.int32
    assert packed.shape == (3, packed_word_count(columns))
    assert torch.equal(unpack_sign_matrix(packed, 3, columns), values)


def test_llamacpp_sign_words_use_lsb_first_and_one_for_negative() -> None:
    values = torch.ones(2, 32)
    values[0, (0, 1, 31)] = -1
    values[1] = -1

    packed = pack_sign_matrix(values)

    assert int(packed[0, 0]) == -(2**31) + 3
    assert int(packed[1, 0]) == -1


def test_llamacpp_sign_words_reject_noncanonical_padding() -> None:
    packed = pack_sign_matrix(torch.ones(1, 33))
    packed[0, -1] = 2

    with pytest.raises(ValueError, match="non-zero padding bit"):
        unpack_sign_matrix(packed, 1, 33)


def test_llamacpp_sign_word_row_stride_controls_optional_vector_loads() -> None:
    scalar_rows = pack_sign_matrix(torch.ones(2, 33))
    vector_rows = pack_sign_matrix(torch.ones(2, 128))

    assert packed_row_stride_bytes(33) == 8
    assert packed_row_stride_bytes(128) == 16
    assert not supports_vector_word_loads(scalar_rows)
    assert supports_vector_word_loads(vector_rows)


def _logical(*, factor_dtype: torch.dtype = torch.float32) -> LogicalLayerState:
    spec = QuantizedLinearSpec(
        "blocks.0.linear",
        "nanoquant-v1",
        35,
        3,
        33,
        str(factor_dtype).removeprefix("torch."),
        "float32",
        outlier_count=2,
        outlier_value_dtype="int8",
        has_outlier_scales=True,
        has_bias=True,
    )
    left = torch.where(
        torch.arange(99).reshape(3, 33) % 2 == 0,
        torch.ones(3, 33),
        -torch.ones(3, 33),
    ).to(factor_dtype)
    right = torch.where(
        torch.arange(33 * 35).reshape(33, 35) % 5 == 0,
        -torch.ones(33, 35),
        torch.ones(33, 35),
    ).to(factor_dtype)
    return LogicalLayerState(
        spec,
        left,
        right,
        torch.cat((torch.tensor([0.5, 0.0]), torch.linspace(0.5, 1.5, 32), torch.zeros(1))),
        torch.linspace(0.75, 1.25, 33),
        torch.linspace(1.0, 1.5, 3),
        torch.tensor([0.1, -0.2, 0.3]),
        torch.tensor([1, 34], dtype=torch.int32),
        torch.tensor([[1, -2], [3, -4], [5, -6]], dtype=torch.int8),
        torch.tensor([0.25, 0.5]),
    )


@pytest.mark.parametrize("factor_dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_packed_layer_roundtrip_preserves_logical_state(factor_dtype: torch.dtype) -> None:
    logical = _logical(factor_dtype=factor_dtype)

    packed = pack_logical_layer(logical)
    restored = packed.to_logical()

    assert packed.layout == PACKED_LAYOUT_VERSION
    assert packed.left_words.shape == (3, 2)
    assert packed.right_words.shape == (33, 2)
    assert restored.spec == logical.spec
    for expected, actual in (
        (logical.left_binary, restored.left_binary),
        (logical.right_binary, restored.right_binary),
        (logical.scale_pre, restored.scale_pre),
        (logical.scale_mid, restored.scale_mid),
        (logical.scale_post, restored.scale_post),
        (logical.bias, restored.bias),
        (logical.outlier_indices, restored.outlier_indices),
        (logical.outlier_values, restored.outlier_values),
        (logical.outlier_scales, restored.outlier_scales),
    ):
        assert torch.equal(expected, actual)


def test_packed_layer_rejects_salient_scales_for_non_int8_values() -> None:
    logical = _logical()
    packed = pack_logical_layer(logical)
    invalid_spec = replace(packed.spec, outlier_value_dtype="float32")

    with pytest.raises(ValueError, match="scales require int8"):
        PackedLayerState(
            invalid_spec,
            packed.layout,
            packed.left_words,
            packed.right_words,
            packed.scale_pre,
            packed.scale_mid,
            packed.scale_post,
            packed.bias,
            packed.outlier_indices,
            packed.outlier_values.float(),
            packed.outlier_scales,
        )


def test_packed_layer_rejects_nonzero_scale_pre_at_salient_columns() -> None:
    logical = _logical()
    scale_pre = logical.scale_pre.clone()
    scale_pre[logical.outlier_indices.long()] = 1

    with pytest.raises(ValueError, match="exactly zero scale_pre"):
        PackedLayerState(
            logical.spec,
            PACKED_LAYOUT_VERSION,
            pack_sign_matrix(logical.left_binary),
            pack_sign_matrix(logical.right_binary),
            scale_pre,
            logical.scale_mid,
            logical.scale_post,
            logical.bias,
            logical.outlier_indices,
            logical.outlier_values,
            logical.outlier_scales,
        )


def test_packed_layout_metadata_matches_reference_kernel_contract() -> None:
    layout = PackedLayoutMetadata()

    assert layout.version == "llama.cpp-i32-lsb-v1"
    assert layout.tensor_namespace == PACKED_TENSOR_NAMESPACE
    assert layout.word_bits == 32
    assert layout.bit_order == "least-significant-bit-first"
    assert (layout.positive_bit, layout.negative_bit, layout.padding_bit) == (0, 1, 0)
    assert (layout.right_sidecar_name, layout.left_sidecar_name) == ("nq_v", "nq_u")
    assert layout.bias_storage == "separate-additive-tensor"
    assert dict(GGUF_TENSOR_SUFFIXES) == {
        "factor_right_words": "nq_v",
        "factor_left_words": "nq_u",
        "scale_pre": "nq_scale_pre",
        "scale_mid": "nq_scale_mid",
        "scale_post": "nq_scale_post",
        "outlier_indices": "nq_salient_idx",
        "outlier_values": "nq_salient_weight",
        "outlier_scales": "nq_salient_scale",
        "bias": "bias",
    }
    assert layout.reference.commit == PACKED_REFERENCE_COMMIT
    assert layout.reference.cuda_sha256 == PACKED_REFERENCE_CUDA_SHA256
