"""Materialize per-block and full component overlays for a multiplier initializer."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import _paths  # noqa: F401
import torch

from nanoquant.application.foldable_mlp_multipliers import (
    InstalledMultipliers,
    install_global_mlp_multipliers,
    rescaled_global_mlp_components,
    seed_global_mlp_multipliers,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.foldable_mlp_initializer import load_foldable_mlp_initializer
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--initializer-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-source", default="google/gemma-3-1b-it")
    parser.add_argument(
        "--model-revision",
        default="dcc83ea841ab6100d6b47a070329e1ba4cf78752",
    )
    parser.add_argument("--initializer-multiplier-limit", type=float, default=128.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--composition",
        action="append",
        default=[],
        help="Additional named block subset in name=block,block form.",
    )
    parser.add_argument(
        "--winsorize-original-bounds",
        action="store_true",
        help="Clamp seed axes to the original gate/up/down fitting bounds.",
    )
    return parser


def _parse_composition(value: str) -> tuple[str, set[int]]:
    name, separator, block_text = value.partition("=")
    try:
        blocks = {int(item) for item in block_text.split(",") if item}
    except ValueError as error:
        raise ValueError("initializer composition blocks must be integers") from error
    if not separator or not name or not blocks or any(character.isspace() for character in name):
        raise ValueError("initializer composition must use name=block,block")
    return name, blocks


def _winsorize_original_bounds(installed: InstalledMultipliers) -> dict[str, int]:
    counts = {"lower": 0, "upper": 0}
    with torch.no_grad():
        for (_block, path), wrapper in installed.wrappers.items():
            minimum, maximum = {
                "mlp.gate_proj": (0.25, 2.0),
                "mlp.up_proj": (0.1, 8.0),
                "mlp.down_proj": (0.25, 4.0),
            }[path]
            for parameter in (
                wrapper.log_input_multiplier,
                wrapper.log_output_multiplier,
            ):
                if parameter is None:
                    continue
                counts["lower"] += int((parameter < math.log(minimum)).sum())
                counts["upper"] += int((parameter > math.log(maximum)).sum())
                parameter.clamp_(min=math.log(minimum), max=math.log(maximum))
    return counts


def _write_overlay(
    root: Path,
    tensors: dict[str, torch.Tensor],
    *,
    blocks: set[int],
    identity: dict[str, str],
    global_tuning: object,
    initializer_sha256: str,
    policy: dict[str, object],
    replaced_bytes: int,
) -> dict[str, object]:
    root.mkdir(parents=True)
    tensor_path = root / "components.safetensors"
    SAFETENSORS.save(tensors, tensor_path)
    digest = hash_file(tensor_path)
    manifest = {
        "schema_version": 2,
        "semantics": "replace-existing-factorized-components",
        "source_dense_tensor_sha256": f"initializer-transfer:{initializer_sha256}",
        "frozen_identity": identity,
        "global_tuning": global_tuning,
        "policy": {str(block): policy[str(block)] for block in sorted(blocks)},
        "tensor_sha256": digest,
        "tensor_count": len(tensors),
        "replaced_payload_bytes": replaced_bytes,
        "replacement_payload_bytes": replaced_bytes,
        "payload_byte_delta": 0,
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
            for name, value in sorted(tensors.items())
        },
    }
    atomic_write_json(root / "manifest.json", manifest)
    return {"directory": str(root), **manifest}


def run(args: argparse.Namespace) -> int:
    if args.initializer_multiplier_limit <= 1:
        raise ValueError("initializer multiplier limit must exceed one")
    initializer = load_foldable_mlp_initializer(
        args.initializer,
        expected_sha256=args.initializer_sha256,
        model_source=args.model_source,
        model_revision=args.model_revision,
    )
    policy = initializer.manifest.get("policy")
    if not isinstance(policy, dict) or not policy:
        raise ValueError("initializer does not declare a block policy")
    blocks = {int(block) for block in policy}
    compositions = tuple(_parse_composition(value) for value in args.composition)
    if len({name for name, _blocks in compositions}) != len(compositions):
        raise ValueError("initializer composition names must be unique")
    if any(not selected.issubset(blocks) for _name, selected in compositions):
        raise ValueError("initializer composition selects a block outside the seed")
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=args.model_source,
            revision=args.model_revision,
            device=args.device,
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=True,
        )
        installed = install_global_mlp_multipliers(loaded.model)
        consumed = seed_global_mlp_multipliers(
            installed,
            initializer.tensors,
            log_limit=math.log(args.initializer_multiplier_limit),
        )
        winsorized = (
            _winsorize_original_bounds(installed)
            if args.winsorize_original_bounds
            else None
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        with atomic_workspace(args.output_root, replace_existing=True) as temporary:
            arms: dict[str, object] = {}
            for block in sorted(blocks):
                tensors, replaced_bytes = rescaled_global_mlp_components(
                    installed,
                    blocks={block},
                )
                name = f"block-{block:02d}"
                arms[name] = _write_overlay(
                    temporary / name,
                    tensors,
                    blocks={block},
                    identity=identity,
                    global_tuning=global_tuning,
                    initializer_sha256=initializer.tensor_sha256,
                    policy=policy,
                    replaced_bytes=replaced_bytes,
                )
            tensors, replaced_bytes = rescaled_global_mlp_components(
                installed,
                blocks=blocks,
            )
            arms["full"] = _write_overlay(
                temporary / "full",
                tensors,
                blocks=blocks,
                identity=identity,
                global_tuning=global_tuning,
                initializer_sha256=initializer.tensor_sha256,
                policy=policy,
                replaced_bytes=replaced_bytes,
            )
            for name, selected in compositions:
                if name in arms:
                    raise ValueError(f"initializer composition name is reserved: {name}")
                tensors, replaced_bytes = rescaled_global_mlp_components(
                    installed,
                    blocks=selected,
                )
                arms[name] = _write_overlay(
                    temporary / name,
                    tensors,
                    blocks=selected,
                    identity=identity,
                    global_tuning=global_tuning,
                    initializer_sha256=initializer.tensor_sha256,
                    policy=policy,
                    replaced_bytes=replaced_bytes,
                )
            atomic_write_json(
                temporary / "summary.json",
                {
                    "schema_version": 1,
                    "role": "analysis-only foldable MLP initializer materialization",
                    "run_output": str(args.run_output.resolve()),
                    "initializer": {
                        "directory": str(args.initializer.resolve()),
                        "tensor_sha256": initializer.tensor_sha256,
                        "seeded_axes": list(consumed),
                        "winsorized_original_bounds": winsorized,
                    },
                    "frozen_identity": identity,
                    "global_tuning": global_tuning,
                    "arms": arms,
                },
            )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
