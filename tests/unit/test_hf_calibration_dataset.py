import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import nanoquant.infrastructure.hf_calibration_dataset as calibration_module
from nanoquant.config.schema import BehaviorSliceConfig, DatasetConfig, DatasetSourceConfig, ReasoningMode
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.hf_calibration_dataset import (
    PinnedCalibrationDataset,
    _openr1_messages,
    _pack_behavior_records,
    _pack_chat_records,
    _slice_wikitext,
    load_or_prepare_calibration,
    load_pinned_calibration,
    materialize_pinned_calibration,
)
from nanoquant.ports.chat_behavior import RenderedBehaviorRecord


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


def test_behavior_packing_never_splits_records_and_rejects_overlength_units() -> None:
    def record(length: int, value: int) -> RenderedBehaviorRecord:
        return RenderedBehaviorRecord(
            (value,) * length,
            (2,) * length,
            (2,) * length,
            (False,) * length,
            (0.0,) * length,
        )

    windows, receipts, rejected = _pack_behavior_records(
        (record(7, 7), record(4, 4), record(11, 9), record(6, 6)),
        count=2,
        sequence_length=10,
        pad_token_id=0,
    )

    assert windows[0][0] == [7] * 7 + [0] * 3
    assert windows[1][0] == [4] * 4 + [6] * 6
    assert len(receipts) == 3
    assert rejected == 1


def test_behavior_packing_emits_bounded_progress() -> None:
    events: list[tuple[str, dict[str, object]]] = []
    record = RenderedBehaviorRecord((7,), (2,), (2,), (False,), (0.0,))

    _pack_behavior_records(
        (record for _index in range(2_000)),
        count=1,
        sequence_length=2_000,
        pad_token_id=0,
        slice_name="thinking",
        progress=lambda event, fields: events.append((event, dict(fields))),
    )

    progress = [fields for event, fields in events if event == "packing_progress"]
    assert progress
    assert progress[0]["slice"] == "thinking"
    assert progress[0]["rendered_records_considered"] == 1_000
    assert progress[0]["valid_tokens"] == 1_000


def test_openr1_normalization_selects_a_complete_verified_generation() -> None:
    messages = _openr1_messages(
        {
            "problem": "2+2?",
            "generations": [
                "<think>wrong path</think>3",
                "<think>add the terms</think>4",
            ],
            "correctness_math_verify": [False, True],
        }
    )

    assert messages[-1] == {
        "role": "assistant",
        "reasoning_content": "add the terms",
        "content": "4",
    }

    with pytest.raises(ValueError, match="no generation marked correct"):
        _openr1_messages(
            {
                "problem": "2+2?",
                "generations": ["<think>wrong</think>3"],
                "correctness_math_verify": [False],
            }
        )


def test_behavior_preparation_logs_slice_and_artifact_milestones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
            assert return_tensors == "pt"
            return SimpleNamespace(input_ids=torch.arange(max(40, len(text))).reshape(1, -1))

    class Behavior:
        supported_modes = (ReasoningMode.THINKING, ReasoningMode.NON_THINKING)

        @staticmethod
        def policy_identity(_tokenizer: object) -> str:
            return "sha256:fixture-policy"

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    source = DatasetSourceConfig("fixture/raw", revision="pinned")
    config = DatasetConfig(
        behavior_slices=(
            BehaviorSliceConfig("raw", ReasoningMode.RAW, source, "raw_text", 1.0),
        )
    )
    events: list[str] = []
    monkeypatch.setattr(
        calibration_module.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: RawTokenizer(),
    )
    monkeypatch.setattr(calibration_module, "chat_behavior_for_snapshot", lambda _snapshot: Behavior())
    monkeypatch.setattr(
        calibration_module,
        "_load_behavior_source",
        lambda _item, _seed: ({"text": "enough raw fixture text for tokenization"},),
    )

    calibration_module.prepare_behavior_calibration(
        snapshot,
        tmp_path / "run",
        config,
        sample_count=1,
        sequence_length=8,
        seed=0,
        progress=lambda event, _fields: events.append(event),
    )

    assert events == [
        "started",
        "tokenizer_load_started",
        "tokenizer_load_completed",
        "slice_source_load_started",
        "slice_source_load_completed",
        "raw_tokenization_started",
        "slice_completed",
        "artifact_persist_started",
        "completed",
    ]


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


def test_validated_calibration_can_be_materialized_into_a_run_local_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    generated = _fixture_calibration()
    (source / "calibration-input.json").write_text(
        json.dumps(
            {
                "sample_count": 3,
                "sequence_length": 4,
                "seed": 7,
                "artifact_id": generated.reference.artifact_id,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(calibration_module, "load_pinned_calibration", lambda *_args: generated)

    materialized = materialize_pinned_calibration(
        source,
        tmp_path / "destination",
        sample_count=3,
        sequence_length=4,
        seed=7,
        preparation_id="sha256:config",
        tokenizer_identity="sha256:tokenizer",
    )
    loaded = load_pinned_calibration(tmp_path / "destination", materialized.reference)
    receipt = json.loads((tmp_path / "destination" / "calibration-input.json").read_text(encoding="utf-8"))

    assert torch.equal(loaded.input_ids, generated.input_ids)
    assert torch.equal(loaded.attention_mask, generated.attention_mask)
    assert receipt["materialized_from"] == str(source.resolve())
    assert receipt["tokenizer_identity"] == "sha256:tokenizer"
