"""Render a retained quality-evaluation JSON result as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from nanoquant.infrastructure.io_utils import atomic_write_text
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = cast(Any, json.loads(args.input.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError("quality evaluation result must be a JSON object")
    output = args.output or args.input.with_suffix(".md")
    atomic_write_text(output, render_quality_evaluation_markdown(payload))


if __name__ == "__main__":
    main()
