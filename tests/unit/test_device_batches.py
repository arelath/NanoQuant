import pytest
import torch

from nanoquant.application.device_batches import iter_device_batches


def test_cpu_device_batches_preserve_order_and_partial_tail() -> None:
    first = torch.arange(15).reshape(5, 3)
    second = first + 100

    batches = tuple(iter_device_batches((first, second), 2, torch.device("cpu")))

    assert [batch[0].shape[0] for batch in batches] == [2, 2, 1]
    assert torch.equal(torch.cat([batch[0] for batch in batches]), first)
    assert torch.equal(torch.cat([batch[1] for batch in batches]), second)


def test_device_batches_reject_an_empty_tensor_set() -> None:
    with pytest.raises(ValueError, match="at least one tensor"):
        tuple(iter_device_batches((), 2, torch.device("cpu")))
