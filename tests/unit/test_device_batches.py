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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned CUDA transfer requires a GPU")
def test_cuda_device_batches_preserve_values_with_bounded_fixed_slots() -> None:
    device = torch.device("cuda")
    rows, sequence, width, batch_size = 64, 1024, 512, 8
    row_ids = torch.arange(rows, dtype=torch.bfloat16).reshape(rows, 1, 1)
    first = row_ids.expand(rows, sequence, width).contiguous().pin_memory()
    second = (first + 100).pin_memory()
    pair_bytes = 2 * batch_size * sequence * width * first.element_size()
    torch.cuda.empty_cache()
    baseline_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    observed_first = []
    observed_second = []

    for first_batch, second_batch in iter_device_batches((first, second), batch_size, device):
        torch.cuda._sleep(10_000_000)  # type: ignore[attr-defined]
        observed_first.append(first_batch[:, 0, 0].clone())
        observed_second.append(second_batch[:, 0, 0].clone())

    assert torch.equal(torch.cat(observed_first).cpu(), torch.arange(rows, dtype=torch.bfloat16))
    assert torch.equal(torch.cat(observed_second).cpu(), torch.arange(rows, dtype=torch.bfloat16) + 100)
    peak_increment = torch.cuda.max_memory_reserved(device) - baseline_reserved
    assert peak_increment <= 4 * pair_bytes
    del first, second, observed_first, observed_second
    torch.cuda.empty_cache()
