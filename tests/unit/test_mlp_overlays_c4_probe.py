import argparse
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))

import probe_mlp_overlays_c4 as probe  # noqa: E402


def test_overlay_parser_accepts_composed_nonempty_paths() -> None:
    assert probe._parse_overlay("candidate=base/path,addition/path") == (
        "candidate",
        (Path("base/path"), Path("addition/path")),
    )


def test_overlay_parser_rejects_missing_paths() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        probe._parse_overlay("candidate=")


def test_overlay_replacements_accepts_attention_paths() -> None:
    replacements = probe._overlay_replacements(
        {"model.layers.2.self_attn.q_proj.weight": torch.ones(3, 4)}
    )
    assert next(iter(replacements)).path == "self_attn.q_proj"
