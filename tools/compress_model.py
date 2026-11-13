"""Interactively compress, validate, optionally benchmark, and publish a model."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
from recipes import INTERACTIVE_RECOMMENDED_MODELS

from nanoquant.interactive_compression import run_interactive_launcher


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imatrix", type=Path, help="llama.cpp .dat or GGUF importance file for a new run")
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parent.parent
    try:
        return run_interactive_launcher(
            repository_root,
            __file__,
            INTERACTIVE_RECOMMENDED_MODELS,
            imatrix_path=args.imatrix,
        )
    except KeyboardInterrupt:
        print("\nInteractive compression interrupted. Run the script again to continue.", flush=True)
        return 130
    except BaseException as exc:
        print(f"Interactive compression failed: {type(exc).__name__}: {exc}", flush=True)
        print("Run the script again to continue from persisted settings.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
