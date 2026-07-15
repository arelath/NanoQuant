"""Export a self-contained runtime bundle from packed weights and a source model shell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanoquant.runtime import write_runtime_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    bundle = write_runtime_bundle(
        args.output,
        args.packed_artifact,
        args.model,
        replace=args.replace,
    )
    result = {
        "output": str(bundle.root),
        "model": bundle.manifest.model.source,
        "revision": bundle.manifest.model.revision,
        "packed_layers": bundle.packed.manifest.layer_count,
        "shell_tensors": len(bundle.manifest.shell_tensors),
        "excluded_linears": len(bundle.manifest.excluded_linear_modules),
        "members": len(bundle.manifest.members),
        "member_bytes": bundle.manifest.total_member_bytes,
        "passed": True,
    }
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
