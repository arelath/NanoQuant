"""Create a validated Hugging Face model-card README for a NanoQuant GGUF."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from nanoquant.infrastructure.huggingface_model_card import (
    HuggingFaceModelCardMetadata,
    load_huggingface_model_card_metadata,
    write_huggingface_model_card,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--body", type=Path)
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument(
        "--pipeline-tag",
        choices=("text-generation", "image-text-to-text"),
        default="text-generation",
    )
    args = parser.parse_args(argv)
    metadata = (
        HuggingFaceModelCardMetadata(args.base_model, args.base_revision)
        if args.source_snapshot is None
        else load_huggingface_model_card_metadata(
            args.base_model,
            args.base_revision,
            args.source_snapshot,
        )
    )
    write_huggingface_model_card(
        metadata,
        args.output,
        model_name=args.model_name,
        body_source=args.body,
        pipeline_tag=args.pipeline_tag,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
