from pathlib import Path
from types import SimpleNamespace

import torch

from tools.probe_non_wikitext_kd_quality import (
    _contiguous_token_windows,
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
        ]
    )

    assert args.primary_baseline == "baseline"
    assert args.primary_candidate == "candidate"
