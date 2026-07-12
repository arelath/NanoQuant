from dataclasses import replace
from pathlib import Path

import pytest

import nanoquant.resident_quantization as resident
from nanoquant.resident_quantization import ResidentQuantizationRequest


def test_resident_algorithm_version_invalidates_commit_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"),
        Path("output"),
        "fixture/model",
        "revision",
        ((1, 2, 3),),
        device="cpu",
    )
    original = resident._resident_config_hash(request)

    monkeypatch.setattr(
        resident,
        "RESIDENT_ALGORITHM_VERSION",
        resident.RESIDENT_ALGORITHM_VERSION + 1,
    )

    assert resident._resident_config_hash(request) != original


def test_legacy_tuning_seed_mode_invalidates_commit_identity() -> None:
    request = ResidentQuantizationRequest(
        Path("snapshot"), Path("output"), "fixture/model", "revision", ((1, 2, 3),), device="cpu"
    )

    assert resident._resident_config_hash(request) != resident._resident_config_hash(
        replace(request, legacy_tuning_seed_reset=True)
    )
