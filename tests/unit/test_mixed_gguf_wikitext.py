import hashlib

import pytest
import torch
from benchmark_mixed_gguf_wikitext import verify_tokens


def test_retained_wikitext_identity_includes_masks_and_token_order():
    tokens = torch.tensor([[1, 2, 3], [1, 4, 5]])
    mask = torch.ones_like(tokens)
    digest = hashlib.sha256(tokens.numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    expected = "sha256:" + digest.hexdigest()
    assert verify_tokens(tokens, mask, expected) == expected
    with pytest.raises(ValueError, match="digest differs"):
        verify_tokens(tokens.flip(0), mask, expected)
    with pytest.raises(ValueError, match="digest differs"):
        verify_tokens(tokens, None, expected)
