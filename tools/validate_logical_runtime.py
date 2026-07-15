"""Prove a logical runtime artifact exactly matches a committed frozen run."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nanoquant.infrastructure.runtime_export import validate_frozen_run_logical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, required=True)
    parser.add_argument("--no-global-tuning", action="store_true")
    parser.add_argument(
        "--use-validation-cache",
        action="store_true",
        help="Trust unchanged source artifact signatures instead of freshly hashing every member.",
    )
    args = parser.parse_args()
    result = validate_frozen_run_logical(
        args.run_output,
        args.artifact,
        args.expected_blocks,
        use_global_tuning=not args.no_global_tuning,
        fresh_validation=not args.use_validation_cache,
    )
    payload = asdict(result)
    payload["output"] = str(result.output)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
