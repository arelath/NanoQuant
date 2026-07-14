from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.support.vram_budget import VramBudgetFactory


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--vram-report", type=Path, default=None, help="append VRAM budget results as JSONL")


@pytest.fixture
def vram_budget(request: pytest.FixtureRequest) -> VramBudgetFactory:
    configured = request.config.getoption("--vram-report")
    environment = os.environ.get("NANOQUANT_VRAM_REPORT")
    report = configured if configured is not None else (Path(environment) if environment else None)
    return VramBudgetFactory(request.node.nodeid, report)
