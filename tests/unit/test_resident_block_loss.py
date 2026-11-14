import pytest
import torch

from nanoquant.resident_quantization import _weighted_mse


def _expected(prediction: torch.Tensor, target: torch.Tensor, importance: torch.Tensor) -> float:
    return float(((prediction.float() - target.float()).square() * importance).sum() / prediction.numel())


def test_weighted_mse_matches_the_reference_formula() -> None:
    generator = torch.Generator().manual_seed(3)
    prediction = torch.randn(3, 5, 4, generator=generator, dtype=torch.bfloat16)
    target = torch.randn(3, 5, 4, generator=generator, dtype=torch.bfloat16)
    importance = torch.rand(4, generator=generator)

    assert _weighted_mse(prediction, target, importance) == pytest.approx(_expected(prediction, target, importance))
    with pytest.raises(ValueError, match="shapes differ"):
        _weighted_mse(prediction, target[:2], importance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for a device-cached target")
def test_weighted_mse_accepts_a_gpu_cached_teacher_target() -> None:
    generator = torch.Generator().manual_seed(4)
    prediction = torch.randn(3, 5, 4, generator=generator, dtype=torch.bfloat16)
    target = torch.randn(3, 5, 4, generator=generator, dtype=torch.bfloat16)
    importance = torch.rand(4, generator=generator)

    host = _weighted_mse(prediction, target, importance)

    assert _weighted_mse(prediction, target.cuda(), importance) == host
    assert _weighted_mse(prediction, target, importance.cuda()) == host
