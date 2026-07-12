"""Schema-driven CLI input adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nanoquant.config.codec import apply_overrides, load_config, parse_override, to_dict
from nanoquant.config.help import schema_reference
from nanoquant.config.validation import raise_for_issues, validate


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="nanoquant")
    subcommands = result.add_subparsers(dest="command", required=True)
    inspect = subcommands.add_parser("inspect", help="decode, validate, and print a fully resolved recipe view")
    inspect.add_argument("recipe", type=Path)
    inspect.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="sparse schema-aware override; available paths come from the canonical schema",
    )
    subcommands.add_parser("config-reference", help="emit canonical configuration reference data")
    return result


def main(arguments: list[str] | None = None) -> int:
    args = parser().parse_args(arguments)
    if args.command == "config-reference":
        print(json.dumps(schema_reference(), indent=2, default=str))
        return 0
    config = load_config(args.recipe)
    config = apply_overrides(config, dict(parse_override(item) for item in args.set))
    raise_for_issues(validate(config))
    print(json.dumps(to_dict(config), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
