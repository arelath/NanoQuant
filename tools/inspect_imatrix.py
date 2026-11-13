"""Inspect and pin llama.cpp imatrix input without starting compression."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.imatrix import imatrix_config, load_imatrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    config = imatrix_config(args.path)
    data = load_imatrix(args.path, config.sha256)
    print(json.dumps({'calibration': {'imatrix': to_dict(config)}, 'format': data.format,
                      'tensors': {name: list(value.shape) for name, value in data.entries.items()}}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
