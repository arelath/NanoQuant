"""Check recovered Gemma scale semantics and paired sequence aggregation."""

import pytest
import torch
from probe_residual_stage_functional import aggregate, copy_parameter, split_qkv


def test_gemma_norm_effective_gamma_is_converted_before_bf16_cast():
    weight = torch.zeros(3, dtype=torch.bfloat16)
    gamma = torch.tensor([0.25, 1.0, 1.007])
    copy_parameter(weight, gamma, gemma_norm=True)
    torch.testing.assert_close(weight, (gamma - 1).bfloat16())


def test_parameter_copy_rejects_invalid_source():
    with pytest.raises(ValueError, match="shape"):
        copy_parameter(torch.zeros(2), torch.zeros(3))
    with pytest.raises(ValueError, match="finiteness"):
        copy_parameter(torch.zeros(2), torch.tensor([1., float("nan")]))


def test_sequence_aggregation_weights_by_token_count():
    rows = [{"arm": dict(negative_log_likelihood=2., kl_nats_per_token=0.2, token_count=3)},
            {"arm": dict(negative_log_likelihood=4., kl_nats_per_token=0.6, token_count=1)}]
    result = aggregate("arm", rows)
    assert result.negative_log_likelihood == pytest.approx(2.5)
    assert result.kl_nats_per_token == pytest.approx(0.3)
    assert result.token_count == 4


def test_shared_qkv_preserves_query_key_value_row_order():
    weight = torch.arange(24).reshape(8, 3)
    query, key, value = split_qkv(weight, [4, 2, 2])
    assert torch.equal(query, weight[:4])
    assert torch.equal(key, weight[4:6])
    assert torch.equal(value, weight[6:])
    with pytest.raises(ValueError, match="widths"):
        split_qkv(weight, [4, 4, 4])
