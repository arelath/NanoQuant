"""Export packed Gemma layers for the pinned modified llama.cpp GGUF converter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nanoquant.runtime.llamacpp import export_llamacpp_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_llamacpp_checkpoint(args.packed_artifact, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_packed_descriptor_sha256": manifest.source_packed_descriptor_sha256,
                "block_count": len(manifest.shards),
                "layer_count": manifest.layer_count,
                "tensor_count": manifest.tensor_count,
                "weight_bytes": manifest.weight_bytes,
                "reference_converter_sha256": manifest.reference.converter_sha256,
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
