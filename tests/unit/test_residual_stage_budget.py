"""Storage and reconstruction checks for the residual-stage diagnostic."""

import pytest
import torch
from probe_residual_stage_budget import (
    HEADER_BYTES,
    correction_bytes,
    decode_payload,
    encode_payload,
    factor_bytes,
    reconstruct,
)


def test_sign_payload_roundtrip_preserves_non_byte_aligned_shape():
    state = {"a.binary": torch.tensor([[-1, 1, -1], [1, 1, -1]], dtype=torch.int8),
             "a.scale": torch.tensor([0.25, 1.125]), "a.indices": torch.tensor([7, 11], dtype=torch.int32)}
    decoded = decode_payload(encode_payload(state))
    assert all(torch.equal(decoded[key], value) for key, value in state.items())
    with pytest.raises(ValueError, match="only -1"):
        encode_payload({"bad.binary": torch.zeros(2, 2)})


@pytest.mark.parametrize("m,n,r", [(3, 7, 2), (64, 96, 32)])
def test_factor_accounting_matches_actual_serialization(m, n, r):
    tensors = {"left_binary": torch.ones(m, r, dtype=torch.int8),
               "right_binary": torch.ones(r, n, dtype=torch.int8),
               "scale_pre": torch.ones(n), "scale_mid": torch.ones(r), "scale_post": torch.ones(m)}
    assert len(encode_payload(tensors)) == HEADER_BYTES + factor_bytes(m, n, r)


@pytest.mark.parametrize("kind", ["int8", "sign"])
def test_correction_accounting_includes_scales_and_indices(kind):
    tensors = {"indices": torch.arange(5, dtype=torch.int32), "column_scale": torch.ones(5)}
    if kind == "int8":
        tensors["int8"] = torch.ones(7, 5, dtype=torch.int8)
    else:
        tensors["binary"] = torch.ones(7, 5, dtype=torch.int8)
        tensors["row_scale"] = torch.ones(7)
    assert len(encode_payload(tensors)) == HEADER_BYTES + correction_bytes(kind, 7, 5)


def test_independent_stage_scales_and_protected_columns():
    tensors = {}
    for prefix, pre, post in [("first.", [2., 3.], [4., 5.]), ("second.", [7., 11.], [13., 17.])]:
        tensors.update({prefix + "left_binary": torch.ones(2, 1, dtype=torch.int8),
                        prefix + "right_binary": torch.ones(1, 2, dtype=torch.int8),
                        prefix + "scale_mid": torch.ones(1),
                        prefix + "scale_pre": torch.tensor(pre), prefix + "scale_post": torch.tensor(post)})
    tensors["protected.indices"] = torch.tensor([1], dtype=torch.int32)
    tensors["protected.values"] = torch.tensor([[101.], [103.]])
    result = reconstruct(decode_payload(encode_payload(tensors)))
    torch.testing.assert_close(result, torch.tensor([[99., 101.], [129., 103.]], dtype=torch.float64))
