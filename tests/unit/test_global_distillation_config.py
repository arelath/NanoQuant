from dataclasses import replace
from pathlib import Path

import pytest

from nanoquant.global_distillation import GlobalDistillationRequest, run_global_topk_distillation


@pytest.mark.parametrize("field", ("initial_cooldown_seconds", "epoch_cooldown_seconds"))
@pytest.mark.parametrize("cooldown", (-1.0, float("inf"), float("nan")))
def test_global_distillation_rejects_invalid_cooldown(
    tmp_path: Path,
    field: str,
    cooldown: float,
) -> None:
    request = GlobalDistillationRequest(
        tmp_path / "run",
        tmp_path / "snapshot",
        "fixture/gemma3",
        "pinned-test-revision",
        ((1,),),
        device="cpu",
    )

    with pytest.raises(ValueError, match="cooldown must be finite and non-negative"):
        run_global_topk_distillation(replace(request, **{field: cooldown}))
