"""Standalone-tool import roots for repository-local experiment definitions."""

from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS = str(Path(__file__).resolve().parent.parent / "experiments")
if _EXPERIMENTS not in sys.path:
    sys.path.insert(0, _EXPERIMENTS)
