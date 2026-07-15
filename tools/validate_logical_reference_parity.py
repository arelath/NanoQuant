"""Compare dense and factorized reference execution for every logical artifact layer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nanoquant.runtime import validate_logical_reference_parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.03125)
    args = parser.parse_args()
    result = validate_logical_reference_parity(
        args.artifact,
        absolute_tolerance=args.absolute_tolerance,
    )
    payload = asdict(result)
    payload["artifact"] = str(result.artifact)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
