import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytest import MonkeyPatch

from tools.probe_non_wikitext_kd_quality import (
    _c4_slice_reservation,
    _contiguous_token_windows,
    _load_c4_tokens,
    _parse_arm,
    _parser,
)


class _Tokenizer:
    bos_token_id = 2

    def __call__(self, text: str, **kwargs: object) -> SimpleNamespace:
        assert text == "first second"
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": True}
        return SimpleNamespace(input_ids=torch.arange(18).unsqueeze(0))


def test_arm_parser_keeps_windows_drive_colons_inside_the_run_path() -> None:
    assert _parse_arm(r"candidate=postkd;D:\evidence\candidate") == (
        "candidate",
        "postkd",
        Path(r"D:\evidence\candidate"),
        None,
    )
    assert _parse_arm(
        r"conditional=tuning;D:\evidence\candidate;D:\evidence\primary.json"
    ) == (
        "conditional",
        "tuning",
        Path(r"D:\evidence\candidate"),
        Path(r"D:\evidence\primary.json"),
    )


def test_contiguous_c4_windows_apply_offset_after_tokenization() -> None:
    tokens, bos_token_id = _contiguous_token_windows(
        ["first", "second"],
        _Tokenizer(),
        offset=1,
        samples=2,
        sequence_length=6,
    )

    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))


def test_primary_comparison_is_explicit_in_the_cli_protocol() -> None:
    args = _parser().parse_args(
        [
            "--snapshot",
            "snapshot",
            "--output",
            "output.json",
            "--arm",
            "baseline=postkd;baseline-run",
            "--arm",
            "candidate=postkd;candidate-run",
            "--primary-baseline",
            "baseline",
            "--primary-candidate",
            "candidate",
            "--expected-steps",
            "baseline=256",
            "--expected-steps",
            "candidate=256",
            "--slice-registry",
            "registry.json",
            "--slice-id",
            "candidate-slice",
        ]
    )

    assert args.primary_baseline == "baseline"
    assert args.primary_candidate == "candidate"


def test_c4_slice_reservation_rejects_retired_overlap(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "slices": [
                    {
                        "id": "old",
                        "dataset": "allenai/c4",
                        "split": "validation",
                        "offset": 0,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_start": 0,
                        "token_end": 24576,
                        "token_hash": "old",
                        "status": "retired",
                    },
                    {
                        "id": "new",
                        "dataset": "allenai/c4",
                        "split": "validation",
                        "offset": 24,
                        "samples": 48,
                        "sequence_length": 512,
                        "token_start": 12288,
                        "token_end": 36864,
                        "token_hash": "new",
                        "status": "reserved",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps"):
        _c4_slice_reservation(
            registry,
            "new",
            offset=24,
            samples=48,
            sequence_length=512,
            token_hash="new",
        )


def test_local_c4_shard_uses_packaged_json_loader(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    shard = tmp_path / "c4.json.gz"
    shard.write_bytes(b"retained shard")
    calls: list[tuple[object, ...]] = []

    class _Dataset:
        _fingerprint = "local-fingerprint"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item: slice) -> dict[str, list[str]]:
            assert item == slice(None, 2)
            return {"text": ["first", "second"]}

    def fake_load_dataset(*args: object, **kwargs: object) -> _Dataset:
        calls.append(args + (kwargs,))
        return _Dataset()

    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.load_dataset",
        fake_load_dataset,
    )
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        tmp_path,
        revision="revision",
        data_file=str(shard),
        documents=2,
        offset=1,
        samples=2,
        sequence_length=6,
        local_files_only=True,
    )

    assert calls == [
        (
            "json",
            {
                "data_files": {"validation": str(shard.resolve())},
                "split": "validation",
            },
        )
    ]
    assert fingerprint == "local-fingerprint"
    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))


def test_local_c4_arrow_cache_opens_without_dataset_rebuild(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    arrow = tmp_path / "c4-validation.arrow"
    arrow.write_bytes(b"retained arrow")

    class _Dataset:
        _fingerprint = "arrow-fingerprint"

        def __len__(self) -> int:
            return 2

        def __getitem__(self, item: slice) -> dict[str, list[str]]:
            assert item == slice(None, 2)
            return {"text": ["first", "second"]}

    observed: list[str] = []
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.Dataset.from_file",
        lambda path: observed.append(path) or _Dataset(),
    )
    monkeypatch.setattr(
        "tools.probe_non_wikitext_kd_quality.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )

    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        tmp_path,
        revision="revision",
        data_file=str(arrow),
        documents=2,
        offset=1,
        samples=2,
        sequence_length=6,
        local_files_only=True,
    )

    assert observed == [str(arrow.resolve())]
    assert fingerprint == "arrow-fingerprint"
    assert bos_token_id == 2
    assert torch.equal(tokens, torch.arange(6, 18).reshape(2, 6))
