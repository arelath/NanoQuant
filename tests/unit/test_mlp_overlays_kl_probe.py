from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


def _probe_module() -> object:
    tools = str(Path(__file__).resolve().parents[2] / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module("probe_mlp_overlays_kl")


def test_overlay_parser_uses_named_path_form() -> None:
    probe = _probe_module()

    assert probe._parse_overlay("incumbent=some/path") == (
        "incumbent",
        (Path("some/path"),),
    )
    assert probe._parse_overlay("candidate=base/path,addition/path") == (
        "candidate",
        (Path("base/path"), Path("addition/path")),
    )
    with pytest.raises(Exception, match="name=path"):
        probe._parse_overlay("missing-path=")
