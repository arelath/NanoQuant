"""Screen raw and winsorized multiplier-seed transfer on a retained post-KD run."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload, _select_token_window
from probe_factorized_component_overlays_kl import _arm_result, _dtype
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from torch import nn

from nanoquant.application.kl_budget import KlSequenceResult, causal_kl_nll_per_sequence_from_logits
from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.linear_math import rescale_factorized_terms
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.factorized_component_overlay import load_factorized_component_overlay
from nanoquant.infrastructure.foldable_mlp_initializer import load_foldable_mlp_initializer
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_model_protocol import HuggingFaceModel
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash


@dataclass(frozen=True, slots=True)
class SeedArm:
    name: str
    blocks: tuple[int, ...]
    winsorized: bool


def _parse_blocks(value: str) -> tuple[int, ...]:
    blocks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not blocks or len(blocks) != len(set(blocks)) or any(block < 0 for block in blocks):
        raise argparse.ArgumentTypeError("prefix order must contain unique non-negative blocks")
    return blocks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument(
        "--prefix-order",
        type=_parse_blocks,
        help="Evaluate cumulative raw-seed prefixes in this predeclared block order.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--component-overlay", type=Path)
    parser.add_argument("--only-component-overlay", action="store_true")
    return parser


def _decoder(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose decoder blocks")
    return layers


def _module_at_path(block: nn.Module, path: str) -> FactorizedReferenceLinear:
    current = block
    for part in path.split("."):
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"seed target path is unavailable: {path}")
        current = child
    if not isinstance(current, FactorizedReferenceLinear):
        raise TypeError(f"seed target is not factorized: {path}")
    return current


def _parse_seed_name(name: str) -> tuple[int, str, str]:
    prefix = "model.layers."
    suffixes = {
        ".input_log_multiplier": "input",
        ".output_log_multiplier": "output",
    }
    if not name.startswith(prefix):
        raise ValueError(f"invalid seed tensor name: {name}")
    for suffix, axis in suffixes.items():
        if name.endswith(suffix):
            logical = name.removeprefix(prefix).removesuffix(suffix)
            block_text, path = logical.split(".", maxsplit=1)
            return int(block_text), path, axis
    raise ValueError(f"invalid seed tensor axis: {name}")


def _fit_bounds(path: str, axis: str) -> tuple[float, float]:
    if path == "mlp.gate_proj" and axis == "output":
        return 0.25, 2.0
    if path == "mlp.up_proj" and axis == "output":
        return 0.1, 8.0
    if path == "mlp.down_proj" and axis in {"input", "output"}:
        return 0.25, 4.0
    raise ValueError(f"unsupported seed family: {path}:{axis}")


def _variant_logs(
    tensors: dict[str, torch.Tensor],
    arm: SeedArm,
) -> tuple[dict[tuple[int, str], dict[str, torch.Tensor]], dict[str, object]]:
    grouped: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    values = []
    lower_hits = upper_hits = 0
    for name, log_value in sorted(tensors.items()):
        block, path, axis = _parse_seed_name(name)
        if block not in arm.blocks:
            continue
        multiplier = log_value.exp()
        if arm.winsorized:
            lower, upper = _fit_bounds(path, axis)
            lower_hits += int((multiplier < lower).sum())
            upper_hits += int((multiplier > upper).sum())
            multiplier = multiplier.clamp(min=lower, max=upper)
        grouped.setdefault((block, path), {})[axis] = multiplier
        values.append(multiplier.reshape(-1))
    if not values:
        raise ValueError(f"seed arm selects no initializer tensors: {arm.name}")
    flat = torch.cat(values)
    return grouped, {
        "blocks": list(arm.blocks),
        "winsorized": arm.winsorized,
        "axis_count": len(values),
        "multiplier_count": flat.numel(),
        "minimum": float(flat.min()),
        "median": float(torch.quantile(flat, 0.5)),
        "maximum": float(flat.max()),
        "lower_winsorized_count": lower_hits,
        "upper_winsorized_count": upper_hits,
    }


def _module_components(prefix: str, module: FactorizedReferenceLinear) -> dict[str, torch.Tensor]:
    values = {
        "scale_pre": module.scale_pre,
        "scale_post": module.scale_post,
        "outlier_values": module.outlier_values,
        "patch_left": module.patch_left,
        "patch_right": module.patch_right,
    }
    return {
        f"{prefix}.{name}": value.detach().cpu().clone()
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }


def _replacement_components(
    model: nn.Module,
    grouped: dict[tuple[int, str], dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    replacements: dict[str, torch.Tensor] = {}
    decoder = _decoder(model)
    for (block, path), axes in sorted(grouped.items()):
        module = _module_at_path(decoder[block], path)
        input_multiplier = axes.get("input")
        output_multiplier = axes.get("output")
        scaled = rescale_factorized_terms(
            module.scale_pre,
            module.scale_post,
            input_multiplier=(
                None
                if input_multiplier is None
                else input_multiplier.to(device=module.scale_pre.device)
            ),
            output_multiplier=(
                None
                if output_multiplier is None
                else output_multiplier.to(device=module.scale_post.device)
            ),
            outlier_indices=module.outlier_indices,
            outlier_values=module.outlier_values,
            patch_left=module.patch_left,
            patch_right=module.patch_right,
        )
        prefix = f"model.layers.{block}.{path}"
        values = {
            "scale_pre": scaled.scale_pre,
            "scale_post": scaled.scale_post,
            "outlier_values": scaled.outlier_values,
            "patch_left": scaled.patch_left,
            "patch_right": scaled.patch_right,
        }
        replacements.update(
            {
                f"{prefix}.{name}": value.detach().cpu()
                for name, value in values.items()
                if isinstance(value, torch.Tensor)
            }
        )
    return replacements


def _copy_components(model: nn.Module, tensors: dict[str, torch.Tensor]) -> None:
    decoder = _decoder(model)
    with torch.no_grad():
        for name, value in tensors.items():
            logical = name.removeprefix("model.layers.")
            block_text, remainder = logical.split(".", maxsplit=1)
            path, component = remainder.rsplit(".", maxsplit=1)
            target = getattr(_module_at_path(decoder[int(block_text)], path), component)
            if not isinstance(target, torch.Tensor) or target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError(f"seed replacement component differs: {name}")
            target.copy_(value.to(device=target.device))


@torch.no_grad()
def _evaluate_arms(
    teacher: nn.Module,
    student: nn.Module,
    tokens: torch.Tensor,
    replacements: dict[str, dict[str, torch.Tensor]],
    source: dict[str, torch.Tensor],
    *,
    device: str,
    token_chunk_size: int,
) -> dict[str, tuple[KlSequenceResult, ...]]:
    sequences: dict[str, list[KlSequenceResult]] = {
        name: [] for name in ("baseline", *replacements)
    }
    teacher.eval()
    student.eval()
    for index in range(tokens.shape[0]):
        batch = tokens[index : index + 1].to(device)
        teacher_logits = cast(HuggingFaceModel, teacher)(input_ids=batch, use_cache=False).logits
        for name in sequences:
            _copy_components(student, source)
            if name != "baseline":
                _copy_components(student, replacements[name])
            student_logits = cast(HuggingFaceModel, student)(input_ids=batch, use_cache=False).logits
            sequences[name].extend(
                causal_kl_nll_per_sequence_from_logits(
                    teacher_logits,
                    student_logits,
                    batch,
                    token_chunk_size=token_chunk_size,
                )
            )
            del student_logits
        del teacher_logits, batch
    _copy_components(student, source)
    return {name: tuple(values) for name, values in sequences.items()}


def run(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.sequence_length <= 1 or args.wikitext_offset < 0 or args.token_chunk_size <= 0:
        raise ValueError("foldable MLP initializer transfer protocol is invalid")
    if args.only_component_overlay and args.component_overlay is None:
        raise ValueError("only-component-overlay requires a component overlay")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    manifest = json.loads((args.initializer / "manifest.json").read_text(encoding="utf-8"))
    initializer = load_foldable_mlp_initializer(
        args.initializer,
        expected_sha256=str(manifest["tensor_sha256"]),
        model_source=MODEL_SOURCE,
        model_revision=args.model_revision,
    )
    blocks = tuple(sorted(int(block) for block in initializer.manifest["policy"]))
    arms = (
        ()
        if args.only_component_overlay
        else tuple(SeedArm(f"block{block}_raw", (block,), False) for block in blocks)
        + (
            SeedArm("full_raw", blocks, False),
            SeedArm("full_winsorized", blocks, True),
        )
    )
    prefix_arms: tuple[SeedArm, ...] = ()
    if args.prefix_order is not None:
        if not set(args.prefix_order).issubset(blocks):
            raise ValueError("prefix order contains a block absent from the initializer")
        prefix_arms = tuple(
            SeedArm(
                "prefix_" + "_".join(str(block) for block in args.prefix_order[:length]),
                args.prefix_order[:length],
                False,
            )
            for length in range(1, len(args.prefix_order) + 1)
        )
        arms += prefix_arms
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = _select_token_window(all_tokens, offset=args.wikitext_offset, samples=args.samples)
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device=args.device,
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=True,
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        arm_reports: dict[str, object] = {}
        replacements: dict[str, dict[str, torch.Tensor]] = {}
        for arm in arms:
            grouped, report = _variant_logs(initializer.tensors, arm)
            replacements[arm.name] = _replacement_components(loaded.model, grouped)
            arm_reports[arm.name] = report
        if args.component_overlay is not None:
            component = load_factorized_component_overlay(
                args.component_overlay,
                frozen_identity=identity,
                global_tuning=loaded.global_tuning,
            )
            replacements["component_overlay"] = component.tensors
            arm_reports["component_overlay"] = {
                "directory": str(args.component_overlay),
                "tensor_sha256": component.manifest["tensor_sha256"],
                "policy": component.manifest["policy"],
            }
        affected = set().union(*(set(values) for values in replacements.values()))
        source = {}
        decoder = _decoder(loaded.model)
        for name in affected:
            logical = name.removeprefix("model.layers.")
            block_text, remainder = logical.split(".", maxsplit=1)
            path, _component = remainder.rsplit(".", maxsplit=1)
            prefix = f"model.layers.{block_text}.{path}"
            source.update(_module_components(prefix, _module_at_path(decoder[int(block_text)], path)))
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        sequence_results = _evaluate_arms(
            teacher,
            loaded.model,
            tokens,
            replacements,
            source,
            device=args.device,
            token_chunk_size=args.token_chunk_size,
        )
        del teacher, loaded
        gc.collect()
        torch.cuda.empty_cache()
    results = {name: _arm_result(name, values) for name, values in sequence_results.items()}
    baseline = results["baseline"]
    comparisons = {
        name: {
            "nll": _paired_metric_payload(baseline, result, "negative_log_likelihood"),
            "kl": _paired_payload(baseline, result, seed=0),
        }
        for name, result in results.items()
        if name != "baseline"
    }
    prefix_marginals = {}
    previous_name = "baseline"
    for arm in prefix_arms:
        prefix_marginals[f"{arm.name}_minus_{previous_name}"] = {
            "nll": _paired_metric_payload(
                results[previous_name],
                results[arm.name],
                "negative_log_likelihood",
            ),
            "kl": _paired_payload(results[previous_name], results[arm.name], seed=0),
        }
        previous_name = arm.name
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only cross-artifact foldable MLP initializer transfer",
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "initializer": {
                "directory": str(args.initializer),
                "tensor_sha256": initializer.tensor_sha256,
                "policy": initializer.manifest["policy"],
            },
            "protocol": {
                "wikitext_split": args.wikitext_split,
                "wikitext_offset": args.wikitext_offset,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "token_chunk_size": args.token_chunk_size,
                "dataset_fingerprint": fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "arms": arm_reports,
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_candidate_minus_baseline": comparisons,
            "paired_prefix_marginals": prefix_marginals,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
