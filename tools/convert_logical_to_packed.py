"""Convert a logical NanoQuant artifact to the llama.cpp-compatible packed layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanoquant.runtime import convert_logical_to_packed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packed = convert_logical_to_packed(args.logical_artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(packed.root),
                "layout": packed.manifest.layout.version,
                "logical_descriptor_sha256": packed.manifest.logical_descriptor_sha256,
                "block_count": len(packed.manifest.blocks),
                "layer_count": packed.manifest.layer_count,
                "weight_bytes": packed.manifest.weight_bytes,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
