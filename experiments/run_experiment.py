"""Run one declarative YAML/JSON experiment definition."""

from __future__ import annotations

import argparse
from pathlib import Path

from recipes import load_declarative_experiment, run_experiment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition", type=Path)
    arguments = parser.parse_args()
    definition = load_declarative_experiment(arguments.definition)
    return run_experiment(definition, launcher_path=arguments.definition)


if __name__ == "__main__":
    raise SystemExit(main())
