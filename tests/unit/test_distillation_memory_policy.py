"""Storage/checkpoint policy must not penalize models that fit on the device."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import nanoquant.global_distillation as global_distillation
from nanoquant.application.layers import FactorizedReferenceLinear, TrainableFactorizedLinear
from nanoquant.global_distillation import (
    _RESIDENT_SIGN_HEADROOM_BYTES,
    _compact_block_expansion_fits,
    _configure_student_checkpointing,
    _needs_compact_student,
    _retain_decoded_signs,
    _spare_memory_after_step,
    _use_spare_memory_for_signs,
)


def test_small_student_keeps_resident_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FactorizedReferenceLinear(torch.ones(4, 4), torch.ones(4, 4),
                                      torch.ones(4), torch.ones(4), torch.ones(4))
    loaded = SimpleNamespace(model=model)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=1024))
    assert not _needs_compact_student(loaded, "cuda")  # type: ignore[arg-type]
    assert not _needs_compact_student(loaded, "cpu")  # type: ignore[arg-type]
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=64))
    assert _needs_compact_student(loaded, "cuda")  # type: ignore[arg-type]


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("headroom", [False, True])
def test_checkpoint_policy_only_removes_verified_nested_checkpoints(
    compact: bool, enabled: bool, headroom: bool,
) -> None:
    class Student(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.term = TrainableFactorizedLinear(torch.ones(4, 4), torch.ones(4, 4),
                                                 torch.ones(4), torch.ones(4), torch.ones(4),
                                                 immutable_binary_factors=True)
            self.is_gradient_checkpointing = False
            self.options: object = None

        def gradient_checkpointing_enable(self, **kwargs: object) -> None:
            self.is_gradient_checkpointing = True
            self.options = kwargs

    model = Student()
    _configure_student_checkpointing(model, compact=compact, enabled=enabled, allow_block_expansion=headroom)
    assert model.term.checkpoint_fixed_signs is not (compact and enabled and headroom)
    assert model.options == ({"gradient_checkpointing_kwargs": {"use_reentrant": True}}
                             if compact and enabled else {} if enabled else None)


def test_no_block_checkpoint_support_retains_term_checkpoint() -> None:
    term = TrainableFactorizedLinear(torch.ones(4, 4), torch.ones(4, 4),
                                    torch.ones(4), torch.ones(4), torch.ones(4),
                                    immutable_binary_factors=True)
    _configure_student_checkpointing(term, compact=True, enabled=True)
    assert term.checkpoint_fixed_signs


def test_block_expansion_requires_workspace_margin(monkeypatch: pytest.MonkeyPatch) -> None:
    term = TrainableFactorizedLinear(torch.ones(4, 4), torch.ones(4, 4),
                                    torch.ones(4), torch.ones(4), torch.ones(4),
                                    immutable_binary_factors=True)
    student = nn.Module()
    student.model = nn.Module()
    student.model.layers = nn.ModuleList([term])
    boundary = 2 * 1024**3 + 64
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (boundary - 1, 12 * 1024**3))
    assert not _compact_block_expansion_fits(student, "cuda")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (boundary, 12 * 1024**3))
    assert _compact_block_expansion_fits(student, "cuda")


def _packed_term(rows: int, *, block_checkpointed: bool = True) -> TrainableFactorizedLinear:
    term = TrainableFactorizedLinear(torch.ones(rows, 4), torch.ones(4, 4),
                                    torch.ones(4), torch.ones(4), torch.ones(rows),
                                    immutable_binary_factors=True)
    term.pack_fixed_signs()
    term.checkpoint_fixed_signs = not block_checkpointed
    return term


def test_resident_signs_fill_the_budget_smallest_first() -> None:
    student = nn.ModuleList([_packed_term(rows) for rows in (16, 4, 8)])
    costs = [term.fixed_sign_residency_bytes() for term in student]
    assert costs[1] < costs[2] < costs[0]

    assert _retain_decoded_signs(student, costs[1] - 1) == (0, 0)
    assert all(term.fixed_sign_residency_bytes() for term in student)
    # The two smallest fit; the largest would overdraw what is left.
    assert _retain_decoded_signs(student, sum(costs) - 1) == (2, costs[1] + costs[2])
    assert [bool(term.fixed_sign_residency_bytes()) for term in student] == [True, False, False]
    assert _retain_decoded_signs(student, costs[0]) == (1, costs[0])
    assert _retain_decoded_signs(student, 10**9) == (0, 0)


def test_spare_memory_subtracts_the_rehearsed_working_set(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(global_distillation, "release_memory", lambda *_, **__: events.append("release"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _: events.append("reset"))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 300)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _: 1000)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (5000, 12000))

    assert _spare_memory_after_step("cuda", lambda: events.append("step")) == 5000 - (1000 - 300)
    # Cached allocator blocks must be returned before both the baseline and the free reading.
    assert events == ["release", "reset", "step", "release"]


def _measured_spare(monkeypatch: pytest.MonkeyPatch, spare: int) -> list[object]:
    measured: list[object] = []

    def spare_memory(_device: str, rehearse: object) -> int:
        measured.append(rehearse)
        return spare

    monkeypatch.setattr(global_distillation, "_spare_memory_after_step", spare_memory)
    return measured


def _rehearse() -> None:
    return None


@pytest.mark.parametrize(("device", "block_checkpointed"), [("cpu", True), ("cuda", False)])
def test_spare_memory_is_not_measured_for_host_or_term_checkpointed_students(
    monkeypatch: pytest.MonkeyPatch, device: str, block_checkpointed: bool,
) -> None:
    measured = _measured_spare(monkeypatch, 10**12)
    student = nn.ModuleList([_packed_term(4, block_checkpointed=block_checkpointed), _packed_term(8)])

    assert _use_spare_memory_for_signs(student, device, _rehearse) is None
    assert _use_spare_memory_for_signs(nn.ModuleList(), "cuda", _rehearse) is None
    assert measured == []
    assert all(term.fixed_sign_residency_bytes() for term in student)


def test_spare_memory_beyond_the_headroom_keeps_signs_resident(monkeypatch: pytest.MonkeyPatch) -> None:
    student = nn.ModuleList([_packed_term(4), _packed_term(8)])
    costs = [term.fixed_sign_residency_bytes() for term in student]

    measured = _measured_spare(monkeypatch, _RESIDENT_SIGN_HEADROOM_BYTES + costs[0] - 1)
    assert _use_spare_memory_for_signs(student, "cuda", _rehearse) == (0, 2, 0)
    assert measured == [_rehearse]

    _measured_spare(monkeypatch, _RESIDENT_SIGN_HEADROOM_BYTES + costs[0])
    assert _use_spare_memory_for_signs(student, "cuda", _rehearse) == (1, 2, costs[0])
    assert [bool(term.fixed_sign_residency_bytes()) for term in student] == [False, True]
