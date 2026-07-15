"""Export a complete committed NanoQuant run to the deployment logical format."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nanoquant.infrastructure.runtime_export import export_frozen_run_logical
from nanoquant.runtime import RuntimeModelMetadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-blocks", type=int, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--config-hash", required=True)
    parser.add_argument("--tokenizer-hash", required=True)
    parser.add_argument("--no-global-tuning", action="store_true")
    parser.add_argument(
        "--use-validation-cache",
        action="store_true",
        help="Trust unchanged source artifact signatures instead of freshly hashing every member.",
    )
    args = parser.parse_args()
    result = export_frozen_run_logical(
        args.run_output,
        args.output,
        RuntimeModelMetadata(
            args.source,
            args.revision,
            args.family,
            args.config_hash,
            args.tokenizer_hash,
        ),
        args.expected_blocks,
        use_global_tuning=not args.no_global_tuning,
        fresh_validation=not args.use_validation_cache,
    )
    payload = asdict(result)
    payload["output"] = str(result.output)
    print(json.dumps(payload, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
