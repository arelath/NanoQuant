"""The 071 ablation changes only the number of trainable outer-scale vectors."""

import pytest
import torch
from probe_071_full_rank_scales import decode_shared, encode_shared, make_model, splice
from probe_residual_rank_split import frozen_payload, payload_weight
from torch import nn


def test_full_rank_arms_start_equal_and_encode_one_shared_pair():
    rng = torch.Generator().manual_seed(12)
    left = torch.randn(4, 8, generator=rng).sign()
    right = torch.randn(8, 6, generator=rng).sign()
    pre, middle, post = torch.ones(6), torch.randn(8, generator=rng), torch.ones(4)
    shared = make_model(left, right, pre, middle, post, (4, 4), True)
    independent = make_model(left, right, pre, middle, post, (4, 4), False)
    assert shared.scale_pre is shared.residual_terms[0].scale_pre
    assert independent.scale_post is not independent.residual_terms[0].scale_post
    assert not shared.left_latent.requires_grad and not independent.residual_terms[0].right_latent.requires_grad
    x = torch.randn(3, 6, generator=rng)
    torch.testing.assert_close(shared(x), independent(x))
    shared(x).square().mean().backward()
    independent(x).square().mean().backward()
    torch.testing.assert_close(shared.scale_post.grad,
                               independent.scale_post.grad + independent.residual_terms[0].scale_post.grad)
    stored = encode_shared(shared)
    assert len(stored) == 8
    assert stored["stage0.scale_pre"].dtype == torch.bfloat16
    torch.testing.assert_close(payload_weight(decode_shared(stored), "cpu"),
                               payload_weight(frozen_payload(independent), "cpu"))
    with pytest.raises(ValueError, match="independent"):
        encode_shared(independent)


def test_splice_restores_original_module_on_failure():
    base = nn.Sequential(nn.Identity())
    original = base[0]
    new = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="check"):
        with splice([(base, "0", new)]):
            assert base[0] is new
            raise RuntimeError("check")
    assert base[0] is original
