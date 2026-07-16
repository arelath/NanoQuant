"""Publish existing experiment result files into the top-level Results index."""

from __future__ import annotations

import argparse
from pathlib import Path

from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_number", type=int)
    parser.add_argument("--model", action="append", type=Path, default=[])
    parser.add_argument("--statistics", action="append", type=Path, default=[])
    parser.add_argument("--report", action="append", type=Path, default=[])
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    root = Path(__file__).resolve().parent.parent
    artifacts = tuple(
        PublishableArtifact(path, kind)
        for kind, paths in (
            (PublishableArtifactKind.MODEL, arguments.model),
            (PublishableArtifactKind.STATISTICS, arguments.statistics),
            (PublishableArtifactKind.REPORT, arguments.report),
        )
        for path in paths
    )
    result = publish_experiment_artifacts(root, arguments.experiment_number, artifacts)
    print(result.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
