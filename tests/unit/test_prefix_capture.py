import torch

from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.infrastructure.tiny_model import TinyCausalTransformer


def test_prefix_capture_records_exact_inputs_and_removes_hook_without_replacement() -> None:
    model = TinyCausalTransformer(seed=2)
    first_block = model.blocks[0]
    identity = id(first_block)
    tokens = (torch.tensor([[1, 2, 3]]), torch.tensor([[3, 2, 1]]))
    captures = capture_prefix_invocations(first_block, tuple(lambda value=value: model(value) for value in tokens))
    assert len(captures) == 2
    assert id(model.blocks[0]) == identity
    assert not first_block._forward_pre_hooks
    for capture, token_batch in zip(captures, tokens, strict=True):
        assert torch.equal(capture.positional[0], model.embed(token_batch))
        assert capture.keyword == {}


def test_captured_tensors_do_not_alias_model_intermediates() -> None:
    model = TinyCausalTransformer(seed=3)
    tokens = torch.tensor([[1, 2]])
    capture = capture_prefix_invocations(model.blocks[0], (lambda: model(tokens),))[0]
    expected = capture.positional[0].clone()  # type: ignore[union-attr]
    model.embed.weight.data.zero_()
    assert torch.equal(capture.positional[0], expected)
