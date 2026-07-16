"""Rebuild and publish a running compression's live weight-error table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from nanoquant.infrastructure.live_reconstruction import rebuild_live_weight_error_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_output", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    run = arguments.run_output.resolve()
    manifest = cast(
        dict[str, Any],
        json.loads((run / "manifest.json").read_text(encoding="utf-8")),
    )
    launcher = cast(dict[str, Any], manifest.get("launcher"))
    experiment_number = launcher.get("experiment_number")
    if not isinstance(experiment_number, int):
        raise ValueError("run manifest does not identify a numbered experiment")
    run_status = str(manifest.get("status", "running"))
    status = "compression complete" if run_status == "completed" else run_status
    report = rebuild_live_weight_error_report(
        arguments.repository_root,
        experiment_number,
        run,
        status=status,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
