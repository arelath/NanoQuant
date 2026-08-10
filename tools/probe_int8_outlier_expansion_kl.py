"""Test rate-matched INT8 outlier expansion on a completed frozen run.

This analysis-only probe keeps the committed factors and tuned continuous state
fixed.  It quantizes the existing outlier sidecar per column, then spends the
saved value bits on residual-selected INT8 correction columns.  Dense splice
evaluation measures the resulting held-out NLL and teacher KL without changing
the committed run or its artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import (
    _paired_payload,
    _replace_weights,
    _select_token_window,
)
from probe_mlp_overlays_kl import _split_tokens
from probe_tuned_mlp_scale_refit import MODEL_SOURCE, PINNED_MODEL_REVISION
from safetensors import safe_open

from nanoquant.application.layers import FrozenReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.calibration_math import shrink_importance
from nanoquant.domain.models import LayerId
from nanoquant.domain.outliers import quantize_int8_columns
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstructionSet,
    collect_splice_reconstructions,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash


@dataclass(frozen=True, slots=True)
class Int8OutlierPatch:
    same_count_weight: torch.Tensor
    expanded_weight: torch.Tensor
    additional_indices: torch.Tensor
    existing_count: int
    expanded_count: int
    bf16_sidecar_bits: int
    int8_sidecar_bits: int


def _stored_int8_columns(
    values: torch.Tensor,
    *,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    quantized, scales = quantize_int8_columns(values)
    stored_scales = scales.to(scale_dtype).float()
    return quantized.float() * stored_scales.reshape(1, -1)


def _sidecar_bits(
    *,
    out_features: int,
    in_features: int,
    count: int,
    value_bits: int,
    scale_bits: int,
) -> int:
    index_bits = max(1, math.ceil(math.log2(max(2, in_features))))
    return count * (out_features * value_bits + index_bits + scale_bits)


def _rate_matched_int8_patch(
    source: torch.Tensor,
    compressed: torch.Tensor,
    existing_indices: torch.Tensor,
    existing_values: torch.Tensor,
    *,
    scale_bits: int = 16,
    scale_dtype: torch.dtype = torch.bfloat16,
    input_importance: torch.Tensor | None = None,
    output_importance: torch.Tensor | None = None,
) -> Int8OutlierPatch:
    """Quantize existing columns and add the best residual columns at equal rate."""

    if source.ndim != 2 or compressed.shape != source.shape:
        raise ValueError("outlier patch source and compressed matrices must match")
    indices = existing_indices.detach().long().reshape(-1).cpu()
    values = existing_values.detach().float().cpu()
    if values.shape != (source.shape[0], indices.numel()):
        raise ValueError("existing outlier values differ from their matrix extent")
    if indices.numel() <= 0 or torch.unique(indices).numel() != indices.numel():
        raise ValueError("outlier patch requires unique existing columns")
    if int(indices.min()) < 0 or int(indices.max()) >= source.shape[1]:
        raise ValueError("existing outlier index is outside the source matrix")

    source32 = source.detach().float().cpu()
    compressed32 = compressed.detach().float().cpu()
    same_count = compressed32.clone()
    same_count[:, indices] += _stored_int8_columns(
        values,
        scale_dtype=scale_dtype,
    ) - values

    bf16_bits = _sidecar_bits(
        out_features=source.shape[0],
        in_features=source.shape[1],
        count=indices.numel(),
        value_bits=16,
        scale_bits=0,
    )
    int8_column_bits = _sidecar_bits(
        out_features=source.shape[0],
        in_features=source.shape[1],
        count=1,
        value_bits=8,
        scale_bits=scale_bits,
    )
    expanded_count = bf16_bits // int8_column_bits
    if expanded_count < indices.numel():
        raise ValueError("INT8 metadata leaves no rate-matched existing sidecar")
    additional_count = expanded_count - indices.numel()

    residual = source32 - same_count
    if (input_importance is None) != (output_importance is None):
        raise ValueError("weighted selection requires paired input/output importance")
    if input_importance is None:
        scores = residual.square().sum(dim=0)
    else:
        assert output_importance is not None
        if input_importance.numel() != source.shape[1] or output_importance.numel() != source.shape[0]:
            raise ValueError("selection importance differs from the matrix dimensions")
        scores = (
            residual.square()
            * output_importance.detach().float().cpu().reshape(-1, 1)
        ).sum(dim=0) * input_importance.detach().float().cpu().reshape(-1)
    scores[indices] = -torch.inf
    additional = (
        torch.empty(0, dtype=torch.long)
        if additional_count == 0
        else torch.topk(scores, additional_count, sorted=True).indices.sort().values
    )
    expanded = same_count.clone()
    if additional.numel():
        correction = residual[:, additional]
        expanded[:, additional] += _stored_int8_columns(
            correction,
            scale_dtype=scale_dtype,
        )
    int8_bits = expanded_count * int8_column_bits
    return Int8OutlierPatch(
        same_count.to(compressed.dtype),
        expanded.to(compressed.dtype),
        additional,
        indices.numel(),
        expanded_count,
        bf16_bits,
        int8_bits,
    )


def _decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError("model does not expose a supported decoder layer stack")
    return tuple(layers)


def _module_at_path(block: torch.nn.Module, path: str) -> torch.nn.Module:
    current = block
    for part in path.split("."):
        child = getattr(current, part, None)
        if not isinstance(child, torch.nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _error_metrics(source: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    error = (source.float() - candidate.float()).square().sum()
    target = source.float().square().sum()
    return {
        "error_energy": float(error),
        "normalized_rmse": math.sqrt(float(error / target.clamp_min(1e-30))),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path)
    parser.add_argument(
        "--selection",
        choices=("raw", "fisher-weighted"),
        default="raw",
    )
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.samples <= 0 or args.sequence_length <= 1 or args.wikitext_offset < 0:
        raise ValueError("INT8 outlier KL protocol is invalid")
    if args.output.exists():
        raise ValueError("INT8 outlier KL output already exists")
    if (args.selection == "fisher-weighted") != (args.calibration_state is not None):
        raise ValueError("Fisher-weighted selection requires exactly one calibration state")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = _select_token_window(
        all_tokens,
        offset=args.wikitext_offset,
        samples=args.samples,
    )

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
        baseline = collect_splice_reconstructions(loaded)
        baseline_by_layer = {item.layer: item.weight for item in baseline.layers}
        blocks = _decoder_layers(loaded.model)
        same_replacements: dict[LayerId, torch.Tensor] = {}
        expanded_replacements: dict[LayerId, torch.Tensor] = {}
        layer_results = []
        calibration_layers: dict[str, int] = {}
        calibration_samples = 0
        if args.calibration_state is not None:
            calibration_manifest = json.loads(
                (args.calibration_state / "manifest.json").read_text(encoding="utf-8")
            )
            calibration_samples = int(calibration_manifest["sample_count"])
            calibration_layers = {
                str(item["path"]): index
                for index, item in enumerate(calibration_manifest["layers"])
            }
        calibration_manager = (
            safe_open(
                str(args.calibration_state / "state.safetensors"),
                framework="pt",
                device="cpu",
            )
            if args.calibration_state is not None
            else nullcontext(None)
        )
        with (
            safe_open(str(args.model), framework="pt", device="cpu") as handle,
            calibration_manager as calibration_context,
        ):
            for block_index, block in enumerate(blocks):
                layer = LayerId(block=loaded.blocks[block_index].block, path="mlp.down_proj")
                module = _module_at_path(block, layer.path)
                if not isinstance(module, FrozenReferenceLinear):
                    raise TypeError("down projection is not a frozen factorized linear")
                if module.outlier_indices is None or module.outlier_values is None:
                    raise ValueError("down projection has no retained outlier sidecar")
                source = handle.get_tensor(
                    f"model.layers.{block_index}.mlp.down_proj.weight"
                )
                compressed = baseline_by_layer[layer]
                values = module.outlier_values.detach().float().cpu()
                if module.outlier_scales is not None:
                    values *= module.outlier_scales.detach().float().cpu().reshape(1, -1)
                input_importance = None
                output_importance = None
                if calibration_context is not None:
                    profile_path = f"block.{block_index}.mlp.down_proj"
                    profile_index = calibration_layers.get(profile_path)
                    if profile_index is None or calibration_samples <= 0:
                        raise ValueError(f"calibration state is missing {profile_path}")
                    input_importance = shrink_importance(
                        calibration_context.get_tensor(
                            f"layer_{profile_index}.inputs.total"
                        ).float()
                        / calibration_samples,
                        0.6,
                    )
                    output_importance = shrink_importance(
                        calibration_context.get_tensor(
                            f"layer_{profile_index}.outputs.total"
                        ).float()
                        / calibration_samples,
                        0.6,
                    )
                patch = _rate_matched_int8_patch(
                    source,
                    compressed,
                    module.outlier_indices,
                    values,
                    input_importance=input_importance,
                    output_importance=output_importance,
                )
                same_replacements[layer] = patch.same_count_weight
                expanded_replacements[layer] = patch.expanded_weight
                layer_results.append(
                    {
                        "block": block_index,
                        "existing_count": patch.existing_count,
                        "expanded_count": patch.expanded_count,
                        "additional_indices": patch.additional_indices.tolist(),
                        "bf16_sidecar_bits": patch.bf16_sidecar_bits,
                        "int8_sidecar_bits": patch.int8_sidecar_bits,
                        "baseline": _error_metrics(source, compressed),
                        "int8_same_count": _error_metrics(source, patch.same_count_weight),
                        "int8_rate_matched": _error_metrics(source, patch.expanded_weight),
                    }
                )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        del loaded
        gc.collect()
        torch.cuda.empty_cache()

        arms: dict[str, SpliceReconstructionSet] = {
            "bf16_existing": baseline,
            "int8_same_count": _replace_weights(baseline, same_replacements),
            "int8_rate_matched": _replace_weights(baseline, expanded_replacements),
        }
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype={
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(cast(str, config.get("torch_dtype")), torch.float32),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        results = {}
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        for name, reconstruction in arms.items():
            print(f"evaluating {name}", flush=True)
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction,
                tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if not teacher_cache:
                teacher_nll, teacher_cache = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(teacher_nll, teacher_cache)
            results[name] = evaluator("full")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del teacher

    names = tuple(arms)
    paired = {}
    for before, after in zip(names, names[1:], strict=False):
        paired[f"{after}_minus_{before}"] = {
            "nll": _paired_metric_payload(
                results[before], results[after], "negative_log_likelihood"
            ),
            "kl": _paired_payload(results[before], results[after], seed=0),
        }
    baseline_name = names[0]
    paired_to_baseline = {
        f"{name}_minus_{baseline_name}": {
            "nll": _paired_metric_payload(
                results[baseline_name],
                results[name],
                "negative_log_likelihood",
            ),
            "kl": _paired_payload(results[baseline_name], results[name], seed=0),
        }
        for name in names[1:]
    }
    total_bf16 = sum(
        cast(int, item["bf16_sidecar_bits"]) for item in layer_results
    )
    total_int8 = sum(
        cast(int, item["int8_sidecar_bits"]) for item in layer_results
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only rate-matched INT8 down-projection outlier screen",
            "run_output": str(args.run_output.resolve()),
            "model_revision": args.model_revision,
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "protocol": {
                "layers": "all mlp.down_proj",
                "selection": args.selection,
                "calibration_state": (
                    None
                    if args.calibration_state is None
                    else str(args.calibration_state.resolve())
                ),
                "int8_quantizer": "symmetric per-column absmax/127",
                "scale_dtype": "bfloat16",
                "wikitext_split": args.wikitext_split,
                "wikitext_offset": args.wikitext_offset,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "dataset_fingerprint": fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "rate": {
                "bf16_existing_bits": total_bf16,
                "int8_rate_matched_bits": total_int8,
                "saved_bits": total_bf16 - total_int8,
            },
            "layers": layer_results,
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_adjacent_arms": paired,
            "paired_to_baseline": paired_to_baseline,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
