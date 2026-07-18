import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nanoquant.infrastructure.hf_calibration_dataset as calibration_module
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.hf_calibration_dataset import (
    PinnedCalibrationDataset,
    _pack_chat_records,
    _slice_wikitext,
    load_or_prepare_calibration,
)


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
        del messages, kwargs
        return list(range(2, 14))

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.arange(max(40, len(text))).reshape(1, -1))


def test_chat_packing_and_wikitext_slicing_are_exact_length_and_deterministic() -> None:
    records = ({"messages": [{"role": "user", "content": str(index)}]} for index in range(20))
    chat = _pack_chat_records(records, Tokenizer(), count=3, sequence_length=10)
    first = _slice_wikitext("x" * 100, Tokenizer(), 4, 8, random.Random(1))
    second = _slice_wikitext("x" * 100, Tokenizer(), 4, 8, random.Random(1))

    assert len(chat) == 3 and all(len(row) == 10 for row in chat)
    assert all(len(row) == 8 for row in first)
    assert first == second


def _fixture_calibration(artifact_id: str = "sha256-" + "1" * 64) -> PinnedCalibrationDataset:
    tokens = torch.arange(12, dtype=torch.long).reshape(3, 4)
    return PinnedCalibrationDataset(
        ArtifactRef("calibration-dataset-manifest", artifact_id, 1),
        tokens,
        torch.ones_like(tokens, dtype=torch.bool),
        "sha256:fixture",
        (("dataset", "revision"),),
    )


def test_run_local_calibration_is_generated_then_reused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated = _fixture_calibration()
    prepared: list[tuple[object, ...]] = []
    loaded: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        calibration_module,
        "prepare_experiment018_calibration",
        lambda *args, **kwargs: prepared.append((args, kwargs)) or generated,
    )
    monkeypatch.setattr(
        calibration_module,
        "load_pinned_calibration",
        lambda *args: loaded.append(args) or generated,
    )

    first = load_or_prepare_calibration(
        tmp_path / "snapshot",
        tmp_path / "run",
        sample_count=3,
        sequence_length=4,
        seed=7,
        preparation_id="sha256:config",
    )
    second = load_or_prepare_calibration(
        tmp_path / "snapshot",
        tmp_path / "run",
        sample_count=3,
        sequence_length=4,
        seed=7,
        preparation_id="sha256:config",
    )

    assert first is generated and second is generated
    assert len(prepared) == 1
    assert len(loaded) == 1
    assert loaded[0][0] == tmp_path / "run"
    assert loaded[0][1] == generated.reference


def test_run_local_calibration_regenerates_when_run_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _fixture_calibration("sha256-" + "1" * 64)
    second = _fixture_calibration("sha256-" + "2" * 64)
    values = iter((first, second))
    prepared: list[object] = []

    def prepare(*args: object, **kwargs: object) -> PinnedCalibrationDataset:
        prepared.append((args, kwargs))
        return next(values)

    monkeypatch.setattr(calibration_module, "prepare_experiment018_calibration", prepare)

    load_or_prepare_calibration(
        tmp_path / "snapshot",
        tmp_path / "run",
        sample_count=3,
        sequence_length=4,
        preparation_id="sha256:first",
    )
    regenerated = load_or_prepare_calibration(
        tmp_path / "snapshot",
        tmp_path / "run",
        sample_count=3,
        sequence_length=4,
        preparation_id="sha256:second",
    )

    assert regenerated is second
    assert len(prepared) == 2
