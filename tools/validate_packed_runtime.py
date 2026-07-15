"""Validate exact conversion and reference execution for a packed NanoQuant artifact."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nanoquant.runtime import validate_packed_conversion, validate_packed_reference_parity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-artifact", type=Path, required=True)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    args = parser.parse_args()
    conversion = asdict(
        validate_packed_conversion(args.logical_artifact, args.packed_artifact)
    )
    parity = asdict(
        validate_packed_reference_parity(
            args.logical_artifact,
            args.packed_artifact,
            absolute_tolerance=args.absolute_tolerance,
        )
    )
    for payload in (conversion, parity):
        for key in ("logical_artifact", "packed_artifact"):
            payload[key] = str(payload[key])
    print(json.dumps({"conversion": conversion, "reference_parity": parity}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
