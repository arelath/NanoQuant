"""Gate the best sparse-corrected sign codebook with paired held-out splice KL."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_sign_word_codebook import (
    PINNED_MODEL_REVISION,
    PROJECTION_PATHS,
    _load_profile,
    _logical_seed,
    _metrics,
    _prepare_fixed_outliers,
    _restore_fixed_outliers,
)
from safetensors import safe_open

from nanoquant.application.kl_budget import paired_bootstrap_kl_delta
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import BinaryFactorSearchConfig
from nanoquant.domain.binary_factor_search import (
    BinaryFactorSearchResult,
    refine_binary_factors_control_then_tabu,
)
from nanoquant.domain.codebook_payload_search import (
    SignWordPayloadSearchConfig,
    SignWordPayloadSearchResult,
    refine_sign_word_payloads,
)
from nanoquant.domain.factorization import (
    AdmmParameters,
    factorize_admm_with_parameters,
)
from nanoquant.domain.mlp_operator_refit import (
    coupled_mlp_output_normalized_rmse,
    fit_coupled_mlp_output_scales,
    fit_linear_input_scales,
    fit_linear_output_scales,
    linear_input_scale_normalized_rmse,
    linear_output_normalized_rmse,
)
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.domain.planning import factor_bit_cost, outlier_bit_cost
from nanoquant.domain.scale_fit import MaterializedScaleFitResult, fit_scales, reconstruct
from nanoquant.domain.sign_word_codebook import (
    FullSignCodebook,
    ProductSignCodebook,
    codebook_index_metrics,
    corrected_asymmetric_codebook_bit_cost,
    decode_product_codebook,
    factorize_sign_word_codebook_admm,
    maximum_corrected_asymmetric_rank_for_budget,
    mixed_right_corrected_codebook_bit_cost,
    mixed_right_linear_codebook_bit_cost,
    mixed_right_product_codebook_bit_cost,
)
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import (
    atomic_workspace,
    atomic_write_json,
    hash_file,
)
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.probe_reconstruction_cache import (
    ProbeReconstructionCache,
    ProbeReconstructionCacheEntry,
)
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens
from nanoquant.runtime.packed import pack_sign_matrix
from nanoquant.runtime.product_codebook import (
    pack_product_codebook_indices,
    pack_product_half_table,
)

MODEL_SOURCE = "google/gemma-3-1b-it"
RECONSTRUCTION_CACHE_ALGORITHM_VERSION = 6
PRODUCT_COMPONENT_SCHEMA_VERSION = 1


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("blocks must be unique non-negative integers")
    return result


def _parse_floats(value: str) -> tuple[float, ...]:
    result = tuple(
        float(item.strip()) for item in value.split(",") if item.strip()
    )
    if (
        not result
        or len(result) != len(set(result))
        or any(item <= 0 for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "selection thresholds must be unique positive fractions"
        )
    return result


def _parse_projections(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if (
        not result
        or len(result) != len(set(result))
        or any(item not in PROJECTION_PATHS for item in result)
    ):
        raise argparse.ArgumentTypeError(
            "projections must be unique known projection names"
        )
    return result


def _parse_projection_free_rows(
    value: str,
) -> tuple[tuple[str, int], ...]:
    result: list[tuple[str, int]] = []
    for item in value.split(","):
        parts = item.strip().split(":", maxsplit=1)
        if len(parts) != 2 or parts[0] not in PROJECTION_PATHS:
            raise argparse.ArgumentTypeError(
                "projection free rows must use known-projection:count entries"
            )
        try:
            free_rows = int(parts[1])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "projection free-row counts must be integers"
            ) from error
        if free_rows < 0:
            raise argparse.ArgumentTypeError(
                "projection free-row counts must not be negative"
            )
        result.append((parts[0], free_rows))
    if not result or len({projection for projection, _rows in result}) != len(result):
        raise argparse.ArgumentTypeError(
            "projection free-row entries must be nonempty and unique"
        )
    return tuple(result)


def _resolve_projection_free_rows(
    projections: tuple[str, ...],
    default: int,
    overrides: tuple[tuple[str, int], ...] | None,
) -> dict[str, int]:
    if overrides is None:
        return {projection: default for projection in projections}
    by_projection = dict(overrides)
    if set(by_projection) != set(projections):
        raise ValueError(
            "projection free-row overrides must choose every requested projection"
        )
    return by_projection


def _parse_block_policy(value: str) -> tuple[tuple[int, str], ...]:
    choices = {"base", "operator", "output", "input", "joint"}
    result = []
    for item in value.split(","):
        parts = item.strip().split(":", maxsplit=1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "block policy entries must use block:choice"
            )
        try:
            block = int(parts[0])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "block policy indices must be integers"
            ) from error
        choice = parts[1].strip()
        if block < 0 or choice not in choices:
            raise argparse.ArgumentTypeError(
                "block policy choices are base, operator, output, input, or joint"
            )
        result.append((block, choice))
    if not result or len({block for block, _choice in result}) != len(result):
        raise argparse.ArgumentTypeError(
            "block policy must contain unique block indices"
        )
    return tuple(result)


def _parse_representation_policy(
    value: str,
) -> tuple[tuple[int, str], ...]:
    result = []
    for item in value.split(","):
        parts = item.strip().split(":", maxsplit=1)
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "representation policy entries must use block:choice"
            )
        try:
            block = int(parts[0])
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "representation policy indices must be integers"
            ) from error
        choice = parts[1].strip()
        if block < 0 or choice not in {"free", "mixed"}:
            raise argparse.ArgumentTypeError(
                "representation policy choices are free or mixed"
            )
        result.append((block, choice))
    if not result or len({block for block, _choice in result}) != len(result):
        raise argparse.ArgumentTypeError(
            "representation policy must contain unique block indices"
        )
    return tuple(result)


def _dtype(config: dict[str, object]) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(cast(str, config.get("torch_dtype")), torch.float32)


def _reconstruction_cache_identity(
    args: argparse.Namespace,
    *,
    model_sha256: str,
    calibration_manifest_sha256: str,
    calibration_state_sha256: str,
    block: int,
    projection: str,
    projection_path: str,
    transposed: bool,
    factorization_shape: tuple[int, int],
    rank: int,
    right_free_rows: int | None = None,
) -> dict[str, object]:
    return {
        "schema": "corrected-codebook-splice-reconstruction",
        "algorithm_version": RECONSTRUCTION_CACHE_ALGORITHM_VERSION,
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "model_sha256": model_sha256,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "calibration_state_sha256": calibration_state_sha256,
        "block": block,
        "projection": projection,
        "projection_path": projection_path,
        "transposed": transposed,
        "factorization_shape": list(factorization_shape),
        "baseline_rank": args.baseline_rank,
        "candidate_rank": rank,
        "right_free_rows": (
            args.right_free_rows
            if right_free_rows is None
            else right_free_rows
        ),
        "codebook_mode": args.codebook_mode,
        "index_width": args.index_width,
        "corrections_per_word": args.corrections_per_word,
        "correction_bits": args.correction_bits,
        "fixed_outlier_indices": list(args.fixed_outlier_indices),
        "outer_iterations": args.outer_iterations,
        "inner_iterations": args.inner_iterations,
        "regularization": args.regularization,
        "penalty_schedule": args.penalty_schedule,
        "convergence_check_interval": (
            args.convergence_check_interval
        ),
        "codebook_update_interval": args.codebook_update_interval,
        "codebook_freeze_fraction": args.codebook_freeze_fraction,
        "assignment_batch_words": args.assignment_batch_words,
        "linear_assignment_sweeps": args.linear_assignment_sweeps,
        "corrected_assignment_candidates": (
            args.corrected_assignment_candidates
        ),
        "scale_fit_passes": args.scale_fit_passes,
        "binary_search": args.binary_search,
        "payload_search": args.payload_search,
        "payload_search_passes": args.payload_search_passes,
        "payload_search_max_words": args.payload_search_max_words,
        "payload_search_scale_passes": args.payload_search_scale_passes,
        "payload_search_batch_words": args.payload_search_batch_words,
        "payload_search_table_chunk": args.payload_search_table_chunk,
        "functional_payload_search": args.functional_payload_search,
        "functional_payload_fit_offset": args.functional_payload_fit_offset,
        "functional_payload_fit_samples": args.functional_payload_fit_samples,
        "functional_payload_validation_offset": (
            args.functional_payload_validation_offset
        ),
        "functional_payload_validation_samples": (
            args.functional_payload_validation_samples
        ),
        "functional_payload_candidate_words": (
            args.functional_payload_candidate_words
        ),
        "calibration_shrinkage": args.calibration_shrinkage,
        "seed": args.seed,
    }


def _binary_search_factors(
    weight: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    fitted: MaterializedScaleFitResult,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    right_free_rows: int | None,
) -> tuple[BinaryFactorSearchResult, dict[str, object]]:
    settings = BinaryFactorSearchConfig(enabled=True)
    common_fit = fit_scales(
        weight,
        left_binary,
        right_binary,
        fitted.scale_pre,
        fitted.scale_mid,
        fitted.scale_post,
        input_importance,
        output_importance,
        alternating_passes=settings.scale_passes,
    )
    right_mutable = None
    if right_free_rows is not None:
        right_mutable = torch.zeros(
            right_binary.shape[0],
            dtype=torch.bool,
            device=right_binary.device,
        )
        right_mutable[:right_free_rows] = True
    control, tabu = refine_binary_factors_control_then_tabu(
        weight,
        left_binary,
        right_binary,
        common_fit.scale_pre,
        common_fit.scale_mid,
        common_fit.scale_post,
        input_importance,
        output_importance,
        scale_passes=settings.scale_passes,
        control_outer_passes=settings.control_outer_passes,
        one_bit_passes=settings.one_bit_passes,
        one_bit_fraction=settings.one_bit_fraction,
        max_one_bit_vectors=settings.max_one_bit_vectors,
        variable_depth_passes=settings.variable_depth_passes,
        variable_depth_length=settings.variable_depth_length,
        tabu_outer_passes=settings.tabu_outer_passes,
        tabu_passes=settings.tabu_passes,
        tabu_steps=settings.tabu_steps,
        tabu_tenure=settings.tabu_tenure,
        tabu_tenure_jitter=settings.tabu_tenure_jitter,
        right_mutable_components=right_mutable,
    )
    return tabu, {
        "settings": asdict(settings),
        "common_refit_error": common_fit.after_error,
        "control_after_error": control.after_error,
        "tabu_after_error": tabu.after_error,
        "left_sign_distance": int((tabu.left_binary != left_binary).sum()),
        "right_sign_distance": int((tabu.right_binary != right_binary).sum()),
    }


def _payload_search_factors(
    args: argparse.Namespace,
    weight: torch.Tensor,
    left_binary: torch.Tensor,
    right_binary: torch.Tensor,
    scale_pre: torch.Tensor,
    scale_mid: torch.Tensor,
    scale_post: torch.Tensor,
    input_importance: torch.Tensor,
    output_importance: torch.Tensor,
    *,
    free_rows: int,
    codebook: FullSignCodebook | None,
    right_indices: torch.Tensor | None,
    right_flip_positions: torch.Tensor | None,
    functional_fit_inputs: torch.Tensor | None,
    functional_held_out_inputs: torch.Tensor | None,
) -> tuple[SignWordPayloadSearchResult, dict[str, object]]:
    settings = SignWordPayloadSearchConfig(
        enabled=True,
        outer_passes=args.payload_search_passes,
        max_words_per_pass=args.payload_search_max_words,
        scale_passes=args.payload_search_scale_passes,
        candidate_batch_words=args.payload_search_batch_words,
        table_chunk_size=args.payload_search_table_chunk,
        functional_candidate_words_per_pass=(
            args.functional_payload_candidate_words
            if args.functional_payload_search
            else 0
        ),
    )
    result = refine_sign_word_payloads(
        weight,
        left_binary,
        right_binary,
        scale_pre,
        scale_mid,
        scale_post,
        input_importance,
        output_importance,
        free_rows=free_rows,
        codebook=codebook,
        right_indices=right_indices,
        right_flip_positions=right_flip_positions,
        config=settings,
        functional_fit_inputs=functional_fit_inputs,
        functional_held_out_inputs=functional_held_out_inputs,
    )
    return result, {
        "settings": asdict(settings),
        "before_error": result.before_error,
        "after_error": result.after_error,
        "accepted_outer_passes": result.accepted_outer_passes,
        "candidate_words_evaluated": result.candidate_words_evaluated,
        "codebook_patterns_evaluated": result.codebook_patterns_evaluated,
        "selected_words": result.selected_words,
        "accepted_words": result.accepted_words,
        "sign_updates": result.sign_updates,
        "functional_candidates_ranked": result.functional_candidates_ranked,
        "functional_fit_error_before": result.functional_fit_error_before,
        "functional_fit_error_after": result.functional_fit_error_after,
        "functional_held_out_error_before": (
            result.functional_held_out_error_before
        ),
        "functional_held_out_error_after": (
            result.functional_held_out_error_after
        ),
    }


def _module_at_path(block: torch.nn.Module, path: str) -> torch.nn.Module:
    current = block
    for part in path.split("."):
        child = (
            current[part]
            if isinstance(current, torch.nn.ModuleDict)
            else getattr(current, part, None)
        )
        if not isinstance(child, torch.nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current


def _decoder_blocks(model: torch.nn.Module) -> tuple[torch.nn.Module, ...]:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError("model does not expose a supported decoder block stack")
    return tuple(layers)


@torch.inference_mode()
def _capture_linear_inputs(
    model: torch.nn.Module,
    module: torch.nn.Module,
    tokens: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    captured = []

    def hook(
        _module: torch.nn.Module,
        inputs: tuple[object, ...],
    ) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise TypeError("MLP projection input hook did not receive a tensor")
        value = inputs[0]
        captured.append(
            value.detach()
            .reshape(-1, value.shape[-1])
            .to(device="cpu", dtype=torch.bfloat16)
        )

    handle = module.register_forward_pre_hook(hook)
    try:
        for index in range(tokens.shape[0]):
            cast(Any, model)(
                input_ids=tokens[index : index + 1].to(device),
                use_cache=False,
            )
    finally:
        handle.remove()
    if len(captured) != tokens.shape[0]:
        raise ValueError("MLP input capture did not cover every requested sequence")
    return torch.cat(captured)


@torch.inference_mode()
def _linear_outputs(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    *,
    device: str,
    row_batch: int = 256,
) -> torch.Tensor:
    if inputs.ndim != 2 or weight.ndim != 2 or inputs.shape[1] != weight.shape[1]:
        raise ValueError("operator-refit linear dimensions do not match")
    if row_batch <= 0:
        raise ValueError("operator-refit row batch must be positive")
    weight_device = weight.to(device=device, dtype=torch.bfloat16)
    outputs = []
    for start in range(0, inputs.shape[0], row_batch):
        outputs.append(
            inputs[start : start + row_batch]
            .to(device=device, dtype=torch.bfloat16)
            .matmul(weight_device.mT)
        )
    return torch.cat(outputs)


@torch.inference_mode()
def _gated_outputs(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    gate = _linear_outputs(inputs, gate_weight, device=device)
    up = _linear_outputs(inputs, up_weight, device=device)
    result = torch.nn.functional.silu(gate.float()) * up.float()
    del gate, up
    return result


@torch.inference_mode()
def _gated_down_outputs(
    inputs: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    gated = _gated_outputs(
        inputs,
        gate_weight,
        up_weight,
        device=device,
    )
    result = _linear_outputs(gated, down_weight, device=device)
    del gated
    return result


def _replace_weights(
    reconstructions: SpliceReconstructionSet,
    replacements: dict[LayerId, torch.Tensor],
) -> SpliceReconstructionSet:
    layers = tuple(
        SpliceReconstruction(
            item.layer,
            replacements.get(item.layer, item.weight),
            item.bias,
            item.weighted_normalized_squared_error,
        )
        for item in reconstructions.layers
    )
    if set(replacements) != {
        item.layer for item in layers if item.layer in replacements
    }:
        raise ValueError("operator refit replacement inventory is incomplete")
    return SpliceReconstructionSet(
        layers,
        reconstructions.unit_members,
        reconstructions.unit_weighted_normalized_squared_errors,
    )


def _multiplier_summary(
    multiplier: torch.Tensor,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, object]:
    values = multiplier.detach().float().reshape(-1).cpu()
    if values.numel() <= 0 or not torch.isfinite(values).all():
        raise ValueError("multiplier summary requires finite values")
    quantiles = torch.quantile(
        values,
        torch.tensor((0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)),
    )
    lower_count = int((values <= minimum).sum())
    upper_count = int((values >= maximum).sum())
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                quantiles,
                strict=True,
            )
        },
        "lower_bound_count": lower_count,
        "upper_bound_count": upper_count,
        "lower_bound_fraction": lower_count / values.numel(),
        "upper_bound_fraction": upper_count / values.numel(),
    }


def _operator_refit_sets(
    teacher: torch.nn.Module,
    blocks: tuple[int, ...],
    fit_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    *,
    device: str,
    gate_grid_points: int,
    minimum_gate_multiplier: float,
    maximum_gate_multiplier: float,
    minimum_up_multiplier: float,
    maximum_up_multiplier: float,
    fit_inputs_by_block: dict[int, torch.Tensor] | None = None,
    validation_inputs_by_block: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, SpliceReconstructionSet], dict[str, Any]]:
    if (fit_inputs_by_block is None) != (validation_inputs_by_block is None):
        raise ValueError("external operator-refit inputs must be paired")
    if fit_inputs_by_block is not None:
        assert validation_inputs_by_block is not None
        if (
            set(blocks) - fit_inputs_by_block.keys()
            or set(blocks) - validation_inputs_by_block.keys()
        ):
            raise ValueError("external operator-refit input inventory is incomplete")
    decoder = _decoder_blocks(teacher)
    result = dict(reconstruction_sets)
    metrics: dict[str, Any] = {}
    for arm in ("free_words", "corrected_codebook"):
        replacements: dict[LayerId, torch.Tensor] = {}
        arm_metrics: dict[str, Any] = {}
        by_layer = {item.layer: item for item in reconstruction_sets[arm].layers}
        for block_index in blocks:
            gate_id = LayerId(BlockId(block_index), PROJECTION_PATHS["gate"])
            up_id = LayerId(BlockId(block_index), PROJECTION_PATHS["up"])
            gate_item = by_layer.get(gate_id)
            up_item = by_layer.get(up_id)
            if gate_item is None or up_item is None:
                raise ValueError("operator refit requires gate and up reconstructions")
            block = decoder[block_index]
            gate_module = _module_at_path(block, PROJECTION_PATHS["gate"])
            up_module = _module_at_path(block, PROJECTION_PATHS["up"])
            if not isinstance(gate_module, torch.nn.Linear) or not isinstance(
                up_module,
                torch.nn.Linear,
            ):
                raise TypeError("operator refit targets must be dense linear modules")
            fit_inputs = (
                _capture_linear_inputs(
                    teacher,
                    gate_module,
                    fit_tokens,
                    device=device,
                )
                if fit_inputs_by_block is None
                else fit_inputs_by_block[block_index]
            )
            validation_inputs = (
                _capture_linear_inputs(
                    teacher,
                    gate_module,
                    validation_tokens,
                    device=device,
                )
                if validation_inputs_by_block is None
                else validation_inputs_by_block[block_index]
            )
            teacher_fit_gate = _linear_outputs(
                fit_inputs,
                gate_module.weight,
                device=device,
            )
            teacher_fit_up = _linear_outputs(
                fit_inputs,
                up_module.weight,
                device=device,
            )
            candidate_fit_gate = _linear_outputs(
                fit_inputs,
                gate_item.weight,
                device=device,
            )
            candidate_fit_up = _linear_outputs(
                fit_inputs,
                up_item.weight,
                device=device,
            )
            fitted = fit_coupled_mlp_output_scales(
                teacher_fit_gate,
                teacher_fit_up,
                candidate_fit_gate,
                candidate_fit_up,
                gate_grid_points=gate_grid_points,
                minimum_gate_multiplier=minimum_gate_multiplier,
                maximum_gate_multiplier=maximum_gate_multiplier,
                minimum_up_multiplier=minimum_up_multiplier,
                maximum_up_multiplier=maximum_up_multiplier,
            )
            teacher_validation_gate = _linear_outputs(
                validation_inputs,
                gate_module.weight,
                device=device,
            )
            teacher_validation_up = _linear_outputs(
                validation_inputs,
                up_module.weight,
                device=device,
            )
            candidate_validation_gate = _linear_outputs(
                validation_inputs,
                gate_item.weight,
                device=device,
            )
            candidate_validation_up = _linear_outputs(
                validation_inputs,
                up_item.weight,
                device=device,
            )
            validation_before = coupled_mlp_output_normalized_rmse(
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
            )
            validation_after = coupled_mlp_output_normalized_rmse(
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
                gate_multiplier=fitted.gate_multiplier,
                up_multiplier=fitted.up_multiplier,
            )
            replacements[gate_id] = (
                gate_item.weight.float()
                * fitted.gate_multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            replacements[up_id] = (
                up_item.weight.float()
                * fitted.up_multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            arm_metrics[str(block_index)] = {
                "fit_before_normalized_rmse": fitted.before_normalized_rmse,
                "fit_after_normalized_rmse": fitted.after_normalized_rmse,
                "validation_before_normalized_rmse": validation_before,
                "validation_after_normalized_rmse": validation_after,
                "validation_change_fraction": (
                    validation_after / validation_before - 1
                ),
                "gate_multiplier_minimum": float(
                    fitted.gate_multiplier.min()
                ),
                "gate_multiplier_maximum": float(
                    fitted.gate_multiplier.max()
                ),
                "up_multiplier_minimum": float(fitted.up_multiplier.min()),
                "up_multiplier_maximum": float(fitted.up_multiplier.max()),
                "gate_multiplier_distribution": _multiplier_summary(
                    fitted.gate_multiplier,
                    minimum=minimum_gate_multiplier,
                    maximum=maximum_gate_multiplier,
                ),
                "up_multiplier_distribution": _multiplier_summary(
                    fitted.up_multiplier,
                    minimum=minimum_up_multiplier,
                    maximum=maximum_up_multiplier,
                ),
            }
            del (
                fit_inputs,
                validation_inputs,
                teacher_fit_gate,
                teacher_fit_up,
                candidate_fit_gate,
                candidate_fit_up,
                teacher_validation_gate,
                teacher_validation_up,
                candidate_validation_gate,
                candidate_validation_up,
            )
            torch.cuda.empty_cache()
        result[f"{arm}_operator_refit"] = _replace_weights(
            reconstruction_sets[arm],
            replacements,
        )
        metrics[arm] = arm_metrics
    return result, metrics


def _downstream_refit_sets(
    teacher: torch.nn.Module,
    blocks: tuple[int, ...],
    fit_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    *,
    device: str,
    minimum_multiplier: float,
    maximum_multiplier: float,
    fit_inputs_by_block: dict[int, torch.Tensor] | None = None,
    validation_inputs_by_block: dict[int, torch.Tensor] | None = None,
    fit_targets_by_block: dict[int, torch.Tensor] | None = None,
    validation_targets_by_block: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, SpliceReconstructionSet], dict[str, Any]]:
    if (fit_inputs_by_block is None) != (validation_inputs_by_block is None):
        raise ValueError("external downstream-refit inputs must be paired")
    if (fit_targets_by_block is None) != (validation_targets_by_block is None):
        raise ValueError("external downstream-refit targets must be paired")
    for inventory, label in (
        (fit_inputs_by_block, "fit inputs"),
        (validation_inputs_by_block, "validation inputs"),
        (fit_targets_by_block, "fit targets"),
        (validation_targets_by_block, "validation targets"),
    ):
        if inventory is not None and set(blocks) - inventory.keys():
            raise ValueError(f"external downstream-refit {label} inventory is incomplete")
    decoder = _decoder_blocks(teacher)
    source_arms = (
        "free_words_operator_refit",
        "corrected_codebook_operator_refit",
    )
    replacements_by_arm: dict[str, dict[LayerId, torch.Tensor]] = {
        arm: {} for arm in source_arms
    }
    metrics: dict[str, dict[str, Any]] = {
        arm: {} for arm in source_arms
    }
    layers_by_arm = {
        arm: {item.layer: item for item in reconstruction_sets[arm].layers}
        for arm in source_arms
    }
    for block_index in blocks:
        gate_id = LayerId(BlockId(block_index), PROJECTION_PATHS["gate"])
        up_id = LayerId(BlockId(block_index), PROJECTION_PATHS["up"])
        down_id = LayerId(BlockId(block_index), PROJECTION_PATHS["down"])
        block = decoder[block_index]
        gate_module = _module_at_path(block, PROJECTION_PATHS["gate"])
        up_module = _module_at_path(block, PROJECTION_PATHS["up"])
        down_module = _module_at_path(block, PROJECTION_PATHS["down"])
        if not all(
            isinstance(module, torch.nn.Linear)
            for module in (gate_module, up_module, down_module)
        ):
            raise TypeError("downstream refit targets must be dense linear modules")
        assert isinstance(gate_module, torch.nn.Linear)
        assert isinstance(up_module, torch.nn.Linear)
        assert isinstance(down_module, torch.nn.Linear)
        fit_inputs = (
            _capture_linear_inputs(
                teacher,
                gate_module,
                fit_tokens,
                device=device,
            )
            if fit_inputs_by_block is None
            else fit_inputs_by_block[block_index]
        )
        validation_inputs = (
            _capture_linear_inputs(
                teacher,
                gate_module,
                validation_tokens,
                device=device,
            )
            if validation_inputs_by_block is None
            else validation_inputs_by_block[block_index]
        )
        teacher_fit = (
            _gated_down_outputs(
                fit_inputs,
                gate_module.weight,
                up_module.weight,
                down_module.weight,
                device=device,
            )
            if fit_targets_by_block is None
            else fit_targets_by_block[block_index].to(device=device)
        )
        teacher_validation = (
            _gated_down_outputs(
                validation_inputs,
                gate_module.weight,
                up_module.weight,
                down_module.weight,
                device=device,
            )
            if validation_targets_by_block is None
            else validation_targets_by_block[block_index].to(device=device)
        )
        for arm in source_arms:
            by_layer = layers_by_arm[arm]
            gate_item = by_layer.get(gate_id)
            up_item = by_layer.get(up_id)
            down_item = by_layer.get(down_id)
            if gate_item is None or up_item is None or down_item is None:
                raise ValueError(
                    "downstream refit requires gate, up, and down reconstructions"
                )
            candidate_fit = _gated_down_outputs(
                fit_inputs,
                gate_item.weight,
                up_item.weight,
                down_item.weight,
                device=device,
            )
            fitted = fit_linear_output_scales(
                teacher_fit,
                candidate_fit,
                minimum_multiplier=minimum_multiplier,
                maximum_multiplier=maximum_multiplier,
            )
            candidate_validation = _gated_down_outputs(
                validation_inputs,
                gate_item.weight,
                up_item.weight,
                down_item.weight,
                device=device,
            )
            validation_before = linear_output_normalized_rmse(
                teacher_validation,
                candidate_validation,
            )
            validation_after = linear_output_normalized_rmse(
                teacher_validation,
                candidate_validation,
                multiplier=fitted.multiplier,
            )
            replacements_by_arm[arm][down_id] = (
                down_item.weight.float()
                * fitted.multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            metrics[arm][str(block_index)] = {
                "fit_before_normalized_rmse": fitted.before_normalized_rmse,
                "fit_after_normalized_rmse": fitted.after_normalized_rmse,
                "validation_before_normalized_rmse": validation_before,
                "validation_after_normalized_rmse": validation_after,
                "validation_change_fraction": (
                    validation_after / validation_before - 1
                ),
                "multiplier_minimum": float(fitted.multiplier.min()),
                "multiplier_maximum": float(fitted.multiplier.max()),
                "multiplier_distribution": _multiplier_summary(
                    fitted.multiplier,
                    minimum=minimum_multiplier,
                    maximum=maximum_multiplier,
                ),
            }
            del candidate_fit, candidate_validation
        del fit_inputs, validation_inputs, teacher_fit, teacher_validation
        torch.cuda.empty_cache()
    result = dict(reconstruction_sets)
    for arm in source_arms:
        result[f"{arm.removesuffix('_refit')}_downstream_refit"] = (
            _replace_weights(
                reconstruction_sets[arm],
                replacements_by_arm[arm],
            )
        )
    return result, metrics


def _downstream_input_refit_sets(
    teacher: torch.nn.Module,
    blocks: tuple[int, ...],
    fit_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    *,
    device: str,
    minimum_multiplier: float,
    maximum_multiplier: float,
    iterations: int,
    learning_rate: float,
    fit_inputs_by_block: dict[int, torch.Tensor] | None = None,
    validation_inputs_by_block: dict[int, torch.Tensor] | None = None,
    fit_targets_by_block: dict[int, torch.Tensor] | None = None,
    validation_targets_by_block: dict[int, torch.Tensor] | None = None,
) -> tuple[dict[str, SpliceReconstructionSet], dict[str, Any]]:
    if (fit_inputs_by_block is None) != (validation_inputs_by_block is None):
        raise ValueError("external downstream input-refit inputs must be paired")
    if (fit_targets_by_block is None) != (validation_targets_by_block is None):
        raise ValueError("external downstream input-refit targets must be paired")
    for inventory, label in (
        (fit_inputs_by_block, "fit inputs"),
        (validation_inputs_by_block, "validation inputs"),
        (fit_targets_by_block, "fit targets"),
        (validation_targets_by_block, "validation targets"),
    ):
        if inventory is not None and set(blocks) - inventory.keys():
            raise ValueError(
                f"external downstream input-refit {label} inventory is incomplete"
            )
    decoder = _decoder_blocks(teacher)
    source_arms = (
        "free_words_operator_refit",
        "corrected_codebook_operator_refit",
    )
    input_replacements: dict[str, dict[LayerId, torch.Tensor]] = {
        arm: {} for arm in source_arms
    }
    joint_replacements: dict[str, dict[LayerId, torch.Tensor]] = {
        arm: {} for arm in source_arms
    }
    metrics: dict[str, dict[str, Any]] = {
        arm: {} for arm in source_arms
    }
    layers_by_arm = {
        arm: {item.layer: item for item in reconstruction_sets[arm].layers}
        for arm in source_arms
    }
    for block_index in blocks:
        gate_id = LayerId(BlockId(block_index), PROJECTION_PATHS["gate"])
        up_id = LayerId(BlockId(block_index), PROJECTION_PATHS["up"])
        down_id = LayerId(BlockId(block_index), PROJECTION_PATHS["down"])
        block = decoder[block_index]
        gate_module = _module_at_path(block, PROJECTION_PATHS["gate"])
        up_module = _module_at_path(block, PROJECTION_PATHS["up"])
        down_module = _module_at_path(block, PROJECTION_PATHS["down"])
        if not all(
            isinstance(module, torch.nn.Linear)
            for module in (gate_module, up_module, down_module)
        ):
            raise TypeError(
                "downstream input-refit targets must be dense linear modules"
            )
        assert isinstance(gate_module, torch.nn.Linear)
        assert isinstance(up_module, torch.nn.Linear)
        assert isinstance(down_module, torch.nn.Linear)
        fit_inputs = (
            _capture_linear_inputs(
                teacher,
                gate_module,
                fit_tokens,
                device=device,
            )
            if fit_inputs_by_block is None
            else fit_inputs_by_block[block_index]
        )
        validation_inputs = (
            _capture_linear_inputs(
                teacher,
                gate_module,
                validation_tokens,
                device=device,
            )
            if validation_inputs_by_block is None
            else validation_inputs_by_block[block_index]
        )
        teacher_fit_gated = None
        teacher_validation_gated = None
        if fit_targets_by_block is None:
            teacher_fit_gated = _gated_outputs(
                fit_inputs,
                gate_module.weight,
                up_module.weight,
                device=device,
            )
            teacher_fit = _linear_outputs(
                teacher_fit_gated,
                down_module.weight,
                device=device,
            )
        else:
            teacher_fit = fit_targets_by_block[block_index].to(device=device)
        if validation_targets_by_block is None:
            teacher_validation_gated = _gated_outputs(
                validation_inputs,
                gate_module.weight,
                up_module.weight,
                device=device,
            )
            teacher_validation = _linear_outputs(
                teacher_validation_gated,
                down_module.weight,
                device=device,
            )
        else:
            teacher_validation = validation_targets_by_block[block_index].to(
                device=device
            )
        for arm in source_arms:
            by_layer = layers_by_arm[arm]
            gate_item = by_layer.get(gate_id)
            up_item = by_layer.get(up_id)
            down_item = by_layer.get(down_id)
            if gate_item is None or up_item is None or down_item is None:
                raise ValueError(
                    "downstream input refit requires gate, up, and down"
                )
            candidate_fit_gated = _gated_outputs(
                fit_inputs,
                gate_item.weight,
                up_item.weight,
                device=device,
            )
            fitted = fit_linear_input_scales(
                teacher_fit,
                candidate_fit_gated,
                down_item.weight.to(device),
                minimum_multiplier=minimum_multiplier,
                maximum_multiplier=maximum_multiplier,
                iterations=iterations,
                learning_rate=learning_rate,
            )
            input_weight = (
                down_item.weight.float().to(device)
                * fitted.multiplier.reshape(1, -1)
            )
            candidate_fit = _linear_outputs(
                candidate_fit_gated,
                input_weight,
                device=device,
            )
            output_fitted = fit_linear_output_scales(
                teacher_fit,
                candidate_fit,
                minimum_multiplier=minimum_multiplier,
                maximum_multiplier=maximum_multiplier,
            )
            candidate_validation_gated = _gated_outputs(
                validation_inputs,
                gate_item.weight,
                up_item.weight,
                device=device,
            )
            validation_before = linear_input_scale_normalized_rmse(
                teacher_validation,
                candidate_validation_gated,
                down_item.weight.to(device),
            )
            validation_input = linear_input_scale_normalized_rmse(
                teacher_validation,
                candidate_validation_gated,
                down_item.weight.to(device),
                multiplier=fitted.multiplier,
            )
            candidate_validation_input = _linear_outputs(
                candidate_validation_gated,
                input_weight,
                device=device,
            )
            validation_joint = linear_output_normalized_rmse(
                teacher_validation,
                candidate_validation_input,
                multiplier=output_fitted.multiplier,
            )
            input_replacements[arm][down_id] = (
                down_item.weight.float()
                * fitted.multiplier.detach().cpu().reshape(1, -1)
            ).to(torch.bfloat16)
            joint_replacements[arm][down_id] = (
                input_replacements[arm][down_id].float()
                * output_fitted.multiplier.detach().cpu().reshape(-1, 1)
            ).to(torch.bfloat16)
            metrics[arm][str(block_index)] = {
                "fit_before_normalized_rmse": fitted.before_normalized_rmse,
                "fit_input_normalized_rmse": fitted.after_normalized_rmse,
                "fit_joint_normalized_rmse": (
                    output_fitted.after_normalized_rmse
                ),
                "validation_before_normalized_rmse": validation_before,
                "validation_input_normalized_rmse": validation_input,
                "validation_joint_normalized_rmse": validation_joint,
                "validation_input_change_fraction": (
                    validation_input / validation_before - 1
                ),
                "validation_joint_change_fraction": (
                    validation_joint / validation_before - 1
                ),
                "accepted_iterations": fitted.accepted_iterations,
                "input_multiplier_minimum": float(
                    fitted.multiplier.min()
                ),
                "input_multiplier_maximum": float(
                    fitted.multiplier.max()
                ),
                "output_multiplier_minimum": float(
                    output_fitted.multiplier.min()
                ),
                "output_multiplier_maximum": float(
                    output_fitted.multiplier.max()
                ),
                "input_multiplier_distribution": _multiplier_summary(
                    fitted.multiplier,
                    minimum=minimum_multiplier,
                    maximum=maximum_multiplier,
                ),
                "output_multiplier_distribution": _multiplier_summary(
                    output_fitted.multiplier,
                    minimum=minimum_multiplier,
                    maximum=maximum_multiplier,
                ),
            }
            del (
                candidate_fit_gated,
                candidate_fit,
                candidate_validation_gated,
                candidate_validation_input,
                input_weight,
            )
        del fit_inputs, validation_inputs, teacher_fit, teacher_validation
        if teacher_fit_gated is not None:
            del teacher_fit_gated
        if teacher_validation_gated is not None:
            del teacher_validation_gated
        torch.cuda.empty_cache()
    result = dict(reconstruction_sets)
    for arm in source_arms:
        prefix = arm.removesuffix("_refit")
        result[f"{prefix}_downstream_input_refit"] = _replace_weights(
            reconstruction_sets[arm],
            input_replacements[arm],
        )
        result[f"{prefix}_downstream_joint_refit"] = _replace_weights(
            reconstruction_sets[arm],
            joint_replacements[arm],
        )
    return result, metrics


def _downstream_policy_sets(
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    policy: tuple[tuple[int, str], ...],
) -> dict[str, SpliceReconstructionSet]:
    policy_by_block = dict(policy)
    result = dict(reconstruction_sets)
    for prefix in ("free_words", "corrected_codebook"):
        names = {
            "base": prefix,
            "operator": f"{prefix}_operator_refit",
            "output": f"{prefix}_operator_downstream_refit",
            "input": f"{prefix}_operator_downstream_input_refit",
            "joint": f"{prefix}_operator_downstream_joint_refit",
        }
        required_names = {
            names[choice] for choice in policy_by_block.values()
        }
        missing = required_names - reconstruction_sets.keys()
        if missing:
            raise ValueError(
                f"downstream policy requires unavailable arms: {sorted(missing)}"
            )
        base = reconstruction_sets[prefix]
        source_layers = {
            choice: {
                item.layer: item
                for item in reconstruction_sets[names[choice]].layers
            }
            for choice in set(policy_by_block.values())
        }
        replacements = {}
        for item in base.layers:
            choice = policy_by_block.get(item.layer.block.index)
            if choice is None:
                continue
            source = source_layers[choice].get(item.layer)
            if source is None:
                raise ValueError(
                    "downstream policy source inventory is incomplete"
                )
            replacements[item.layer] = source.weight
        result[f"{prefix}_operator_policy_refit"] = _replace_weights(
            base,
            replacements,
        )
    return result


def _hybrid_representation_set(
    reconstruction_sets: dict[str, SpliceReconstructionSet],
    policy: tuple[tuple[int, str], ...],
) -> SpliceReconstructionSet:
    names = {
        "free": "free_words_operator_policy_refit",
        "mixed": "corrected_codebook_operator_policy_refit",
    }
    missing = set(names.values()) - reconstruction_sets.keys()
    if missing:
        raise ValueError(
            f"representation policy requires unavailable arms: {sorted(missing)}"
        )
    policy_by_block = dict(policy)
    base = reconstruction_sets[names["mixed"]]
    source_layers = {
        choice: {
            item.layer: item
            for item in reconstruction_sets[name].layers
        }
        for choice, name in names.items()
    }
    replacements = {}
    for item in base.layers:
        choice = policy_by_block.get(item.layer.block.index)
        if choice is None:
            continue
        source = source_layers[choice].get(item.layer)
        if source is None:
            raise ValueError(
                "representation policy source inventory is incomplete"
            )
        replacements[item.layer] = source.weight
    return _replace_weights(base, replacements)


def _export_reconstruction_set(
    destination: Path,
    arm: str,
    reconstructions: SpliceReconstructionSet,
) -> dict[str, object]:
    tensors = {
        (
            f"model.layers.{item.layer.block.index}."
            f"{item.layer.path}.weight"
        ): (
            item.weight.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
        )
        for item in reconstructions.layers
    }
    if not tensors or len(tensors) != len(reconstructions.layers):
        raise ValueError(
            "reconstruction export requires unique non-empty layers"
        )
    with atomic_workspace(destination) as temporary:
        tensor_path = temporary / "weights.safetensors"
        SAFETENSORS.save(tensors, tensor_path)
        manifest = {
            "schema_version": 1,
            "arm": arm,
            "layer_count": len(tensors),
            "blocks": sorted(
                {
                    item.layer.block.index
                    for item in reconstructions.layers
                }
            ),
            "tensor_sha256": hash_file(tensor_path),
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
                for name, value in tensors.items()
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return {
        "directory": str(destination),
        **manifest,
    }


def _export_product_components(
    destination: Path,
    components: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist exact product-code factors before dense reconstruction."""

    if destination.exists():
        raise FileExistsError(f"product component export already exists: {destination}")
    if not components:
        raise ValueError("product component export requires at least one layer")
    with atomic_workspace(destination) as temporary:
        tensors: dict[str, torch.Tensor] = {}
        layers = []
        for index, component in enumerate(components):
            metadata = cast(dict[str, Any], component["metadata"])
            values = cast(dict[str, torch.Tensor], component["tensors"])
            inventory = {}
            for role, value in sorted(values.items()):
                key = f"layers.{index}.{role}"
                copied = value.detach().cpu().contiguous()
                tensors[key] = copied
                inventory[role] = {
                    "key": key,
                    "shape": list(copied.shape),
                    "dtype": str(copied.dtype).removeprefix("torch."),
                }
            layers.append({**metadata, "tensors": inventory})
        tensor_path = temporary / "components.safetensors"
        SAFETENSORS.save(tensors, tensor_path)
        manifest = {
            "schema_version": PRODUCT_COMPONENT_SCHEMA_VERSION,
            "format": "product-codebook-factor-components",
            "layer_count": len(layers),
            "layers": layers,
            "tensor_sha256": hash_file(tensor_path),
            "tensor_bytes": tensor_path.stat().st_size,
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return {
        "directory": str(destination.resolve()),
        "manifest": str((destination / "manifest.json").resolve()),
        "tensor_sha256": manifest["tensor_sha256"],
        "layer_count": manifest["layer_count"],
    }


def _reconstruction_set(
    values: list[tuple[LayerId, torch.Tensor, float]],
) -> SpliceReconstructionSet:
    layers = []
    units = []
    errors = []
    for layer, reconstruction, weighted_rmse in values:
        unit = f"{layer.block.index}:{layer.path}"
        squared_error = weighted_rmse**2
        layers.append(
            SpliceReconstruction(
                layer,
                reconstruction.detach().to(device="cpu", dtype=torch.bfloat16),
                None,
                squared_error,
            )
        )
        units.append((unit, (layer,)))
        errors.append((unit, squared_error))
    return SpliceReconstructionSet(
        tuple(layers),
        tuple(units),
        tuple(errors),
    )


def _select_blocks(
    reconstructions: SpliceReconstructionSet,
    blocks: tuple[int, ...],
) -> SpliceReconstructionSet:
    selected = frozenset(blocks)
    layers = tuple(
        layer
        for layer in reconstructions.layers
        if layer.layer.block.index in selected
    )
    if {layer.layer.block.index for layer in layers} != selected:
        raise ValueError("selected splice blocks do not map to reconstruction layers")
    units = tuple(
        (unit, members)
        for unit, members in reconstructions.unit_members
        if members and all(member.block.index in selected for member in members)
    )
    errors = tuple(
        (unit, error)
        for unit, error in reconstructions.unit_weighted_normalized_squared_errors
        if any(candidate == unit for candidate, _members in units)
    )
    return SpliceReconstructionSet(layers, units, errors)


def _select_token_window(
    tokens: torch.Tensor,
    *,
    offset: int,
    samples: int,
) -> torch.Tensor:
    if tokens.ndim != 2 or offset < 0 or samples <= 0:
        raise ValueError("held-out token window is invalid")
    selected = tokens[offset : offset + samples]
    if selected.shape[0] != samples:
        raise ValueError("held-out token inventory is shorter than requested")
    return selected


def _paired_payload(
    before: Any,
    after: Any,
    *,
    seed: int,
) -> dict[str, float | int | bool]:
    interval = paired_bootstrap_kl_delta(before, after, seed=seed)
    return {
        "point_delta": interval.point_delta,
        "relative_delta": (
            interval.point_delta / before.kl_nats_per_token
        ),
        "lower_delta": interval.lower_delta,
        "upper_delta": interval.upper_delta,
        "confidence": interval.confidence,
        "resamples": interval.resamples,
        "improved_with_confidence": (
            interval.point_delta < 0 and interval.upper_delta < 0
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-reconstruction-set", type=Path)
    parser.add_argument("--export-arm")
    parser.add_argument("--export-product-components", type=Path)
    parser.add_argument("--reconstruction-cache", type=Path)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--projection", choices=tuple(PROJECTION_PATHS), default="down")
    parser.add_argument("--projections", type=_parse_projections)
    parser.add_argument("--transpose-matrix", action="store_true")
    parser.add_argument("--block", type=int, default=12)
    parser.add_argument("--blocks", type=_parse_ints)
    parser.add_argument("--selection-thresholds", type=_parse_floats)
    parser.add_argument("--baseline-rank", type=int, default=970)
    parser.add_argument("--candidate-rank", type=int)
    parser.add_argument("--right-free-rows", type=int, default=0)
    parser.add_argument(
        "--right-free-rows-by-projection",
        type=_parse_projection_free_rows,
        help=(
            "comma-separated projection:count overrides for joint splices; "
            "must cover every requested projection"
        ),
    )
    parser.add_argument(
        "--fixed-outlier-indices",
        type=_parse_ints,
        default=(),
        help=(
            "comma-separated exact input-column indices applied symmetrically "
            "to both reconstruction arms"
        ),
    )
    parser.add_argument("--index-width", type=int, default=10)
    parser.add_argument(
        "--codebook-mode",
        choices=("corrected-full", "product", "linear"),
        default="corrected-full",
    )
    parser.add_argument("--corrections-per-word", type=int, default=2)
    parser.add_argument("--correction-bits", type=int, default=9)
    parser.add_argument("--outer-iterations", type=int, default=800)
    parser.add_argument("--inner-iterations", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=3e-2)
    parser.add_argument("--penalty-schedule", default="cubic")
    parser.add_argument("--convergence-check-interval", type=int, default=100)
    parser.add_argument("--codebook-update-interval", type=int, default=10)
    parser.add_argument("--codebook-freeze-fraction", type=float, default=0.5)
    parser.add_argument("--assignment-batch-words", type=int, default=8192)
    parser.add_argument("--linear-assignment-sweeps", type=int, default=2)
    parser.add_argument("--corrected-assignment-candidates", type=int, default=16)
    parser.add_argument("--scale-fit-passes", type=int, default=2)
    parser.add_argument("--binary-search", action="store_true")
    parser.add_argument("--payload-search", action="store_true")
    parser.add_argument("--payload-search-passes", type=int, default=8)
    parser.add_argument("--payload-search-max-words", type=int, default=8_192)
    parser.add_argument("--payload-search-scale-passes", type=int, default=64)
    parser.add_argument("--payload-search-batch-words", type=int, default=2_048)
    parser.add_argument("--payload-search-table-chunk", type=int, default=128)
    parser.add_argument("--functional-payload-search", action="store_true")
    parser.add_argument("--functional-payload-fit-offset", type=int, default=48)
    parser.add_argument("--functional-payload-fit-samples", type=int, default=4)
    parser.add_argument(
        "--functional-payload-validation-offset", type=int, default=52
    )
    parser.add_argument(
        "--functional-payload-validation-samples", type=int, default=4
    )
    parser.add_argument(
        "--functional-payload-candidate-words", type=int, default=256
    )
    parser.add_argument("--calibration-shrinkage", type=float, default=0.6)
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--wikitext-offset", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--operator-scale-refit", action="store_true")
    parser.add_argument("--operator-refit-offset", type=int, default=48)
    parser.add_argument("--operator-refit-samples", type=int, default=4)
    parser.add_argument("--operator-validation-offset", type=int, default=52)
    parser.add_argument("--operator-validation-samples", type=int, default=4)
    parser.add_argument("--operator-gate-grid-points", type=int, default=41)
    parser.add_argument("--operator-minimum-gate-multiplier", type=float, default=0.5)
    parser.add_argument("--operator-maximum-gate-multiplier", type=float, default=1.5)
    parser.add_argument("--operator-minimum-up-multiplier", type=float, default=0.25)
    parser.add_argument("--operator-maximum-up-multiplier", type=float, default=4.0)
    parser.add_argument("--downstream-scale-refit", action="store_true")
    parser.add_argument("--downstream-input-scale-refit", action="store_true")
    parser.add_argument(
        "--downstream-minimum-output-multiplier",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--downstream-maximum-output-multiplier",
        type=float,
        default=4.0,
    )
    parser.add_argument("--downstream-input-iterations", type=int, default=20)
    parser.add_argument(
        "--downstream-input-learning-rate",
        type=float,
        default=0.25,
    )
    parser.add_argument("--downstream-policy", type=_parse_block_policy)
    parser.add_argument(
        "--representation-policy",
        type=_parse_representation_policy,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    if args.linear_assignment_sweeps <= 0:
        raise ValueError("linear assignment sweeps must be positive")
    if (
        args.codebook_mode == "corrected-full"
        and args.corrections_per_word not in {1, 2, 3}
    ) or (
        args.codebook_mode in {"product", "linear"}
        and args.corrections_per_word != 0
    ):
        raise ValueError(
            "corrected-full mode requires one to three corrections and compact modes require zero"
        )
    if (args.export_reconstruction_set is None) != (args.export_arm is None):
        raise ValueError(
            "reconstruction export requires both destination and arm"
        )
    if args.export_product_components is not None and (
        args.codebook_mode != "product"
        or args.index_width != 16
        or args.corrections_per_word != 0
        or not args.binary_search
        or args.payload_search
    ):
        raise ValueError(
            "product component export requires k16 product mode with binary search and no payload search"
        )
    if args.candidate_rank is not None and args.candidate_rank <= 0:
        raise ValueError("candidate rank/free-row configuration is invalid")
    if args.payload_search and not args.binary_search:
        raise ValueError("payload search requires the matched binary-search control")
    if args.codebook_mode in {"product", "linear"} and args.payload_search:
        raise ValueError("payload search does not support compact codebooks")
    if args.functional_payload_search and not args.payload_search:
        raise ValueError("functional payload search requires payload search")
    if (
        args.wikitext_samples <= 0
        or args.wikitext_offset < 0
        or args.sequence_length < 2
    ):
        raise ValueError("held-out token dimensions are invalid")
    config_payload = json.loads(
        (args.snapshot / "config.json").read_text(encoding="utf-8")
    )
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = cast(dict[str, object], config_payload)
    adapter = adapter_for_config(config)
    projections = (
        (args.projection,)
        if args.projections is None
        else args.projections
    )
    right_free_rows_by_projection = _resolve_projection_free_rows(
        projections,
        args.right_free_rows,
        args.right_free_rows_by_projection,
    )
    if args.candidate_rank is not None and any(
        free_rows < 0 or free_rows >= args.candidate_rank
        for free_rows in right_free_rows_by_projection.values()
    ):
        raise ValueError("candidate rank/free-row configuration is invalid")
    projection_paths = tuple(PROJECTION_PATHS[item] for item in projections)
    downstream_requested = (
        args.downstream_scale_refit
        or args.downstream_input_scale_refit
    )
    expected_operator_projections = (
        {"gate", "up", "down"}
        if downstream_requested
        else {"gate", "up"}
    )
    if (
        args.operator_scale_refit
        and set(projections) != expected_operator_projections
    ):
        raise ValueError(
            "operator scale refit requires its complete projection group"
        )
    if downstream_requested and (
        not args.operator_scale_refit
        or not args.transpose_matrix
        or set(projections) != {"gate", "up", "down"}
    ):
        raise ValueError(
            "downstream scale refit requires transposed gate/up plus down"
        )
    if (
        args.operator_refit_offset < 0
        or args.operator_refit_samples <= 0
        or args.operator_validation_offset < 0
        or args.operator_validation_samples <= 0
        or args.operator_gate_grid_points < 2
        or not 0
        < args.operator_minimum_gate_multiplier
        <= 1
        <= args.operator_maximum_gate_multiplier
        or not 0
        < args.operator_minimum_up_multiplier
        <= 1
        <= args.operator_maximum_up_multiplier
        or not 0
        < args.downstream_minimum_output_multiplier
        <= 1
        <= args.downstream_maximum_output_multiplier
        or args.downstream_input_iterations <= 0
        or args.downstream_input_learning_rate <= 0
    ):
        raise ValueError("operator scale-refit dataset settings are invalid")
    fit_inventory = set(
        range(
            args.operator_refit_offset,
            args.operator_refit_offset + args.operator_refit_samples,
        )
    )
    validation_inventory = set(
        range(
            args.operator_validation_offset,
            args.operator_validation_offset
            + args.operator_validation_samples,
        )
    )
    evaluation_inventory = set(
        range(
            args.wikitext_offset,
            args.wikitext_offset + args.wikitext_samples,
        )
    )
    functional_fit_inventory = set(
        range(
            args.functional_payload_fit_offset,
            args.functional_payload_fit_offset
            + args.functional_payload_fit_samples,
        )
    )
    functional_validation_inventory = set(
        range(
            args.functional_payload_validation_offset,
            args.functional_payload_validation_offset
            + args.functional_payload_validation_samples,
        )
    )
    if args.functional_payload_search and (
        args.functional_payload_fit_offset < 0
        or args.functional_payload_fit_samples <= 0
        or args.functional_payload_validation_offset < 0
        or args.functional_payload_validation_samples <= 0
        or args.functional_payload_candidate_words <= 0
        or functional_fit_inventory & functional_validation_inventory
        or functional_fit_inventory & evaluation_inventory
        or functional_validation_inventory & evaluation_inventory
    ):
        raise ValueError(
            "functional payload fit, validation, and KL windows must be valid and disjoint"
        )
    if args.operator_scale_refit and (
        fit_inventory & validation_inventory
        or fit_inventory & evaluation_inventory
        or validation_inventory & evaluation_inventory
    ):
        raise ValueError(
            "operator fit, validation, and KL evaluation windows must be disjoint"
        )
    blocks = (args.block,) if args.blocks is None else args.blocks
    if args.downstream_policy is not None:
        policy_blocks = {
            block for block, _choice in args.downstream_policy
        }
        if policy_blocks != set(blocks):
            raise ValueError(
                "downstream policy must choose every requested block"
            )
        policy_choices = {
            choice for _block, choice in args.downstream_policy
        }
        if (
            (
                bool(policy_choices - {"base"})
                and not args.operator_scale_refit
            )
            or (
                "output" in policy_choices
                and not args.downstream_scale_refit
            )
            or (
                bool({"input", "joint"} & policy_choices)
                and not args.downstream_input_scale_refit
            )
        ):
            raise ValueError(
                "downstream policy requires each selected refit arm"
            )
    if args.representation_policy is not None:
        representation_blocks = {
            block for block, _choice in args.representation_policy
        }
        if (
            representation_blocks != set(blocks)
            or args.downstream_policy is None
        ):
            raise ValueError(
                "representation policy requires a downstream choice "
                "for every requested block"
            )
    if any(
        block >= adapter.decoder_block_count_from_config(config)
        for block in blocks
    ):
        raise ValueError("requested block is outside the model")
    parameters = AdmmParameters(
        outer_iterations=args.outer_iterations,
        inner_iterations=args.inner_iterations,
        regularization=args.regularization,
        penalty_schedule=args.penalty_schedule,
        convergence_check_interval=args.convergence_check_interval,
        transpose_wide=True,
    )
    reconstruction_cache = (
        ProbeReconstructionCache(args.reconstruction_cache)
        if args.reconstruction_cache is not None
        else None
    )
    cache_model_sha256 = ""
    cache_calibration_manifest_sha256 = ""
    cache_calibration_state_sha256 = ""
    if reconstruction_cache is not None:
        calibration_manifest = args.calibration_state / "manifest.json"
        calibration_tensors = args.calibration_state / "state.safetensors"
        if (
            not calibration_manifest.is_file()
            or not calibration_tensors.is_file()
        ):
            raise ValueError(
                "reconstruction cache requires a complete calibration state"
            )
        cache_model_sha256 = hash_file(args.model)
        cache_calibration_manifest_sha256 = hash_file(
            calibration_manifest
        )
        cache_calibration_state_sha256 = hash_file(calibration_tensors)
    cache_hits = 0
    cache_misses = 0
    cache_keys: dict[str, str] = {}
    functional_fit_tokens = None
    functional_validation_tokens = None
    if args.functional_payload_search:
        functional_required_samples = max(
            args.functional_payload_fit_offset
            + args.functional_payload_fit_samples,
            args.functional_payload_validation_offset
            + args.functional_payload_validation_samples,
        )
        functional_tokens, _functional_fingerprint, _functional_bos = (
            _wikitext_tokens(
                args.snapshot,
                samples=functional_required_samples,
                sequence_length=args.sequence_length,
                local_files_only=args.local_files_only,
            )
        )
        functional_fit_tokens = _select_token_window(
            functional_tokens,
            offset=args.functional_payload_fit_offset,
            samples=args.functional_payload_fit_samples,
        )
        functional_validation_tokens = _select_token_window(
            functional_tokens,
            offset=args.functional_payload_validation_offset,
            samples=args.functional_payload_validation_samples,
        )
    with acquire_device_lease(args.device), safe_open(
        str(args.model),
        framework="pt",
        device="cpu",
    ) as handle:
        functional_inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        if args.functional_payload_search:
            assert functional_fit_tokens is not None
            assert functional_validation_tokens is not None
            functional_teacher = load_causal_language_model(
                args.snapshot,
                torch_dtype=_dtype(config),
                attention_implementation=adapter.attention_implementation,
                local_files_only=args.local_files_only,
            ).to(args.device)
            functional_teacher.eval()
            functional_blocks = _decoder_blocks(functional_teacher)
            for block in blocks:
                for projection, projection_path in zip(
                    projections, projection_paths, strict=True
                ):
                    module = _module_at_path(
                        functional_blocks[block], projection_path
                    )
                    key = (
                        str(block)
                        if len(projections) == 1
                        else f"{block}:{projection}"
                    )
                    functional_inputs[key] = (
                        _capture_linear_inputs(
                            functional_teacher,
                            module,
                            functional_fit_tokens,
                            device=args.device,
                        ),
                        _capture_linear_inputs(
                            functional_teacher,
                            module,
                            functional_validation_tokens,
                            device=args.device,
                        ),
                    )
            del functional_blocks, functional_teacher
            gc.collect()
            torch.cuda.empty_cache()
        baseline_entries: list[tuple[LayerId, torch.Tensor, float]] = []
        candidate_entries: list[tuple[LayerId, torch.Tensor, float]] = []
        reconstruction_metrics: dict[str, dict[str, dict[str, float]]] = {}
        search_metrics: dict[str, dict[str, object]] = {}
        candidate_index_metrics: dict[
            str,
            dict[str, dict[str, float | int | bool]],
        ] = {}
        product_components: list[dict[str, Any]] = []
        rank = 0
        matrix_shape = (0, 0)
        transpose_by_projection: dict[str, bool] = {}
        for block in blocks:
            for projection, projection_path in zip(
                projections,
                projection_paths,
                strict=True,
            ):
                tensor_name = f"model.layers.{block}.{projection_path}.weight"
                calibration_path = f"block.{block}.{projection_path}"
                transpose_current = (
                    args.transpose_matrix
                    and not (
                        downstream_requested
                        and projection == "down"
                    )
                )
                transpose_by_projection[projection_path] = transpose_current
                source_weight = handle.get_tensor(tensor_name)
                current_shape = (
                    (
                        int(source_weight.shape[1]),
                        int(source_weight.shape[0]),
                    )
                    if transpose_current
                    else (
                        int(source_weight.shape[0]),
                        int(source_weight.shape[1]),
                    )
                )
                if matrix_shape != (0, 0) and current_shape != matrix_shape:
                    raise ValueError(
                        "joint projection splices require one factorization shape"
                    )
                matrix_shape = current_shape
                unit_key = (
                    str(block)
                    if len(projections) == 1
                    else f"{block}:{projection}"
                )
                current_right_free_rows = right_free_rows_by_projection[
                    projection
                ]
                target_bits = factor_bit_cost(
                    current_shape[0],
                    current_shape[1],
                    args.baseline_rank,
                    scale_bits=16,
                ).total
                if args.codebook_mode in {"product", "linear"}:
                    if args.candidate_rank is None:
                        raise ValueError(
                            "compact codebook splice requires an explicit candidate rank"
                        )
                    rank = args.candidate_rank
                else:
                    rank = maximum_corrected_asymmetric_rank_for_budget(
                        current_shape[0],
                        current_shape[1],
                        target_bits,
                        left_index_width=None,
                        right_index_width=args.index_width,
                        right_flip_bits=args.correction_bits,
                        rank_multiple=32,
                        scale_width=16,
                    )
                if args.candidate_rank is not None:
                    rank = args.candidate_rank
                cache_identity = (
                    _reconstruction_cache_identity(
                        args,
                        model_sha256=cache_model_sha256,
                        calibration_manifest_sha256=(
                            cache_calibration_manifest_sha256
                        ),
                        calibration_state_sha256=(
                            cache_calibration_state_sha256
                        ),
                        block=block,
                        projection=projection,
                        projection_path=projection_path,
                        transposed=transpose_current,
                        factorization_shape=current_shape,
                        rank=rank,
                        right_free_rows=current_right_free_rows,
                    )
                    if reconstruction_cache is not None
                    else None
                )
                cached = (
                    reconstruction_cache.load(cache_identity)
                    if (
                        reconstruction_cache is not None
                        and cache_identity is not None
                        and args.export_product_components is None
                    )
                    else None
                )
                layer = LayerId(BlockId(block), projection_path)
                if cached is not None:
                    assert reconstruction_cache is not None
                    assert cache_identity is not None
                    if (
                        cached.rank != rank
                        or tuple(cached.baseline.shape)
                        != tuple(source_weight.shape)
                    ):
                        raise ValueError(
                            "cached reconstruction shape or rank is incompatible"
                        )
                    cache_hits += 1
                    cache_keys[unit_key] = reconstruction_cache.key(
                        cache_identity
                    )
                    baseline_metrics = cast(
                        dict[str, float],
                        cached.baseline_metrics,
                    )
                    candidate_metrics = cast(
                        dict[str, float],
                        cached.candidate_metrics,
                    )
                    candidate_index_metrics[unit_key] = cast(
                        dict[
                            str,
                            dict[str, float | int | bool],
                        ],
                        cached.candidate_index_metrics,
                    )
                    baseline_entries.append(
                        (
                            layer,
                            cached.baseline,
                            float(
                                baseline_metrics[
                                    "weighted_normalized_rmse"
                                ]
                            ),
                        )
                    )
                    candidate_entries.append(
                        (
                            layer,
                            cached.candidate,
                            float(
                                candidate_metrics[
                                    "weighted_normalized_rmse"
                                ]
                            ),
                        )
                    )
                    reconstruction_metrics[unit_key] = {
                        "free_words": baseline_metrics,
                        "corrected_codebook": candidate_metrics,
                    }
                    del source_weight
                    continue
                if reconstruction_cache is not None:
                    cache_misses += 1
                input_cpu, output_cpu = _load_profile(
                    args.calibration_state,
                    calibration_path,
                    args.calibration_shrinkage,
                )
                weight = source_weight.to(args.device)
                input_importance = input_cpu.to(args.device).float()
                output_importance = output_cpu.to(args.device).float()
                if transpose_current:
                    weight = weight.mT.contiguous()
                    input_importance, output_importance = (
                        output_importance,
                        input_importance,
                    )
                metric_weight = weight
                fixed_axis = 0 if transpose_current else 1
                weight, outlier_indices, outlier_values = _prepare_fixed_outliers(
                    weight,
                    args.fixed_outlier_indices,
                    axis=fixed_axis,
                )
                baseline_generator = torch.Generator(device=args.device).manual_seed(
                    _logical_seed(args.seed, "free-word-baseline")
                )
                baseline_factors = factorize_admm_with_parameters(
                    weight,
                    input_importance,
                    output_importance,
                    args.baseline_rank,
                    baseline_generator,
                    parameters,
                )
                baseline_fit = fit_scales(
                    weight,
                    baseline_factors.left_binary,
                    baseline_factors.right_binary,
                    baseline_factors.scale_pre,
                    baseline_factors.scale_mid,
                    baseline_factors.scale_post,
                    input_importance,
                    output_importance,
                    alternating_passes=args.scale_fit_passes,
                )
                baseline_left = baseline_factors.left_binary
                baseline_right = baseline_factors.right_binary
                baseline_pre = baseline_fit.scale_pre
                baseline_mid = baseline_fit.scale_mid
                baseline_post = baseline_fit.scale_post
                unit_search_metrics: dict[str, object] = {}
                if args.binary_search:
                    baseline_search, baseline_search_metrics = (
                        _binary_search_factors(
                            weight,
                            baseline_left,
                            baseline_right,
                            baseline_fit,
                            input_importance,
                            output_importance,
                            right_free_rows=None,
                        )
                    )
                    baseline_left = baseline_search.left_binary
                    baseline_right = baseline_search.right_binary
                    baseline_pre = baseline_search.scale_pre
                    baseline_mid = baseline_search.scale_mid
                    baseline_post = baseline_search.scale_post
                    unit_search_metrics["free_binary"] = baseline_search_metrics
                if args.payload_search:
                    functional_pair = functional_inputs.get(unit_key)
                    baseline_payload, baseline_payload_metrics = (
                        _payload_search_factors(
                            args,
                            weight,
                            baseline_left,
                            baseline_right,
                            baseline_pre,
                            baseline_mid,
                            baseline_post,
                            input_importance,
                            output_importance,
                            free_rows=args.baseline_rank,
                            codebook=None,
                            right_indices=None,
                            right_flip_positions=None,
                            functional_fit_inputs=(
                                None
                                if functional_pair is None
                                else functional_pair[0]
                            ),
                            functional_held_out_inputs=(
                                None
                                if functional_pair is None
                                else functional_pair[1]
                            ),
                        )
                    )
                    baseline_right = baseline_payload.right_binary
                    baseline_pre = baseline_payload.scale_pre
                    baseline_mid = baseline_payload.scale_mid
                    baseline_post = baseline_payload.scale_post
                    unit_search_metrics["free_payload"] = baseline_payload_metrics
                baseline_reconstruction = reconstruct(
                    baseline_left,
                    baseline_right,
                    baseline_pre,
                    baseline_mid,
                    baseline_post,
                )
                baseline_reconstruction = _restore_fixed_outliers(
                    baseline_reconstruction,
                    outlier_indices,
                    outlier_values,
                    axis=fixed_axis,
                )
                baseline_metrics = _metrics(
                    metric_weight,
                    baseline_reconstruction,
                    input_importance,
                    output_importance,
                )
                candidate_generator = torch.Generator(device=args.device).manual_seed(
                    _logical_seed(
                        args.seed,
                        f"{args.codebook_mode}-flip{args.corrections_per_word}-"
                        f"{args.index_width}-rank-{rank}",
                    )
                )
                candidate_factors = factorize_sign_word_codebook_admm(
                    weight,
                    input_importance,
                    output_importance,
                    rank,
                    candidate_generator,
                    index_bits=args.index_width,
                    outer_iterations=args.outer_iterations,
                    inner_iterations=args.inner_iterations,
                    regularization=args.regularization,
                    penalty_schedule=args.penalty_schedule,
                    convergence_check_interval=args.convergence_check_interval,
                    codebook_update_interval=args.codebook_update_interval,
                    codebook_freeze_fraction=args.codebook_freeze_fraction,
                    assignment_batch_words=args.assignment_batch_words,
                    linear_assignment_sweeps=args.linear_assignment_sweeps,
                    corrected_assignment_candidates=(
                        args.corrected_assignment_candidates
                    ),
                    codebook_mode=(
                        "product"
                        if args.codebook_mode == "product"
                        else "linear"
                        if args.codebook_mode == "linear"
                        else "full"
                    ),
                    constrain_left=False,
                    right_flips_per_word=args.corrections_per_word,
                    right_free_rows=current_right_free_rows,
                )
                candidate_fit = fit_scales(
                    weight,
                    candidate_factors.factors.left_binary,
                    candidate_factors.factors.right_binary,
                    candidate_factors.factors.scale_pre,
                    candidate_factors.factors.scale_mid,
                    candidate_factors.factors.scale_post,
                    input_importance,
                    output_importance,
                    alternating_passes=args.scale_fit_passes,
                )
                candidate_left = candidate_factors.factors.left_binary
                candidate_right = candidate_factors.factors.right_binary
                candidate_pre = candidate_fit.scale_pre
                candidate_mid = candidate_fit.scale_mid
                candidate_post = candidate_fit.scale_post
                if args.binary_search:
                    candidate_search, candidate_search_metrics = (
                        _binary_search_factors(
                            weight,
                            candidate_left,
                            candidate_right,
                            candidate_fit,
                            input_importance,
                            output_importance,
                            right_free_rows=current_right_free_rows,
                        )
                    )
                    candidate_left = candidate_search.left_binary
                    candidate_right = candidate_search.right_binary
                    candidate_pre = candidate_search.scale_pre
                    candidate_mid = candidate_search.scale_mid
                    candidate_post = candidate_search.scale_post
                    unit_search_metrics["candidate_binary"] = (
                        candidate_search_metrics
                    )
                if args.payload_search:
                    if not isinstance(
                        candidate_factors.right_codebook, FullSignCodebook
                    ):
                        raise ValueError("payload splice requires one full codebook")
                    candidate_payload, candidate_payload_metrics = (
                        _payload_search_factors(
                            args,
                            weight,
                            candidate_left,
                            candidate_right,
                            candidate_pre,
                            candidate_mid,
                            candidate_post,
                            input_importance,
                            output_importance,
                            free_rows=current_right_free_rows,
                            codebook=candidate_factors.right_codebook,
                            right_indices=candidate_factors.right_indices,
                            right_flip_positions=(
                                candidate_factors.right_flip_positions
                            ),
                            functional_fit_inputs=(
                                None
                                if functional_pair is None
                                else functional_pair[0]
                            ),
                            functional_held_out_inputs=(
                                None
                                if functional_pair is None
                                else functional_pair[1]
                            ),
                        )
                    )
                    candidate_right = candidate_payload.right_binary
                    candidate_pre = candidate_payload.scale_pre
                    candidate_mid = candidate_payload.scale_mid
                    candidate_post = candidate_payload.scale_post
                    unit_search_metrics["candidate_payload"] = (
                        candidate_payload_metrics
                    )
                if args.export_product_components is not None:
                    codebook = candidate_factors.right_codebook
                    indices = candidate_factors.right_indices
                    if not isinstance(codebook, ProductSignCodebook) or indices is None:
                        raise ValueError("product component export requires retained product assignments")
                    decoded_tail = decode_product_codebook(
                        indices,
                        codebook,
                        current_shape[1],
                    )
                    if not torch.equal(
                        decoded_tail,
                        candidate_right[current_right_free_rows :],
                    ):
                        raise ValueError("binary search changed a coded product row")
                    physical_outlier_values = (
                        outlier_values.mT.contiguous()
                        if transpose_current
                        else outlier_values.contiguous()
                    )
                    product_components.append(
                        {
                            "metadata": {
                                "block": block,
                                "projection": projection,
                                "projection_path": projection_path,
                                "tensor_name": tensor_name,
                                "physical_shape": list(source_weight.shape),
                                "factorization_shape": list(current_shape),
                                "factorization_transposed": transpose_current,
                                "rank": rank,
                                "right_free_rows": current_right_free_rows,
                                "fixed_outlier_indices": list(args.fixed_outlier_indices),
                            },
                            "tensors": {
                                "factor_left_words": pack_sign_matrix(candidate_left),
                                "factor_right_free_words": pack_sign_matrix(
                                    candidate_right[:current_right_free_rows].contiguous()
                                ),
                                "factor_right_coded_payload": pack_product_codebook_indices(
                                    indices.to(torch.int32).contiguous()
                                ),
                                "factor_right_first_half_words": pack_product_half_table(
                                    codebook.first.contiguous()
                                ),
                                "factor_right_second_half_words": pack_product_half_table(
                                    codebook.second.contiguous()
                                ),
                                "factor_scale_pre": candidate_pre.detach()
                                .float()
                                .cpu()
                                .contiguous(),
                                "factor_scale_mid": candidate_mid.detach()
                                .float()
                                .cpu()
                                .contiguous(),
                                "factor_scale_post": candidate_post.detach()
                                .float()
                                .cpu()
                                .contiguous(),
                                "outlier_indices": outlier_indices.detach()
                                .to(device="cpu", dtype=torch.int32)
                                .contiguous(),
                                "outlier_values": physical_outlier_values.detach()
                                .to(device="cpu", dtype=torch.bfloat16)
                                .contiguous(),
                            },
                        }
                    )
                candidate_reconstruction = reconstruct(
                    candidate_left,
                    candidate_right,
                    candidate_pre,
                    candidate_mid,
                    candidate_post,
                )
                candidate_reconstruction = _restore_fixed_outliers(
                    candidate_reconstruction,
                    outlier_indices,
                    outlier_values,
                    axis=fixed_axis,
                )
                candidate_metrics = _metrics(
                    metric_weight,
                    candidate_reconstruction,
                    input_importance,
                    output_importance,
                )
                candidate_index_metrics[unit_key] = codebook_index_metrics(
                    candidate_factors
                )
                search_metrics[unit_key] = unit_search_metrics
                if transpose_current:
                    baseline_reconstruction = baseline_reconstruction.mT.contiguous()
                    candidate_reconstruction = candidate_reconstruction.mT.contiguous()
                if (
                    reconstruction_cache is not None
                    and cache_identity is not None
                ):
                    cache_keys[unit_key] = reconstruction_cache.store(
                        cache_identity,
                        ProbeReconstructionCacheEntry(
                            baseline=baseline_reconstruction,
                            candidate=candidate_reconstruction,
                            baseline_metrics=cast(
                                dict[str, object],
                                baseline_metrics,
                            ),
                            candidate_metrics=cast(
                                dict[str, object],
                                candidate_metrics,
                            ),
                            candidate_index_metrics=cast(
                                dict[str, object],
                                candidate_index_metrics[unit_key],
                            ),
                            rank=rank,
                            matrix_shape=(
                                int(baseline_reconstruction.shape[0]),
                                int(baseline_reconstruction.shape[1]),
                            ),
                        ),
                    )
                baseline_entries.append(
                    (
                        layer,
                        baseline_reconstruction,
                        float(baseline_metrics["weighted_normalized_rmse"]),
                    )
                )
                candidate_entries.append(
                    (
                        layer,
                        candidate_reconstruction,
                        float(candidate_metrics["weighted_normalized_rmse"]),
                    )
                )
                reconstruction_metrics[unit_key] = {
                    "free_words": baseline_metrics,
                    "corrected_codebook": candidate_metrics,
                }
                del (
                    weight,
                    input_importance,
                    output_importance,
                    baseline_factors,
                    baseline_fit,
                    candidate_fit,
                    candidate_factors,
                    source_weight,
                )
                gc.collect()
                torch.cuda.empty_cache()
        all_reconstruction_sets = {
            "free_words": _reconstruction_set(baseline_entries),
            "corrected_codebook": _reconstruction_set(candidate_entries),
        }
        selection_specs: list[tuple[str, float | None, tuple[int, ...]]] = []
        if args.selection_thresholds is None:
            selection_specs.append(("full", None, blocks))
        else:
            for threshold in args.selection_thresholds:
                selected = tuple(
                    block
                    for block in blocks
                    if all(
                        (
                            1
                            - reconstruction_metrics[
                                (
                                    str(block)
                                    if len(projections) == 1
                                    else f"{block}:{projection}"
                                )
                            ]["corrected_codebook"]["weighted_normalized_rmse"]
                            / reconstruction_metrics[
                                (
                                    str(block)
                                    if len(projections) == 1
                                    else f"{block}:{projection}"
                                )
                            ]["free_words"]["weighted_normalized_rmse"]
                        )
                        >= threshold
                        for projection in projections
                    )
                )
                if not selected:
                    raise ValueError(
                        f"selection threshold {threshold} selects no blocks"
                    )
                selection_specs.append(
                    (f"weighted_gain_ge_{threshold:.6g}", threshold, selected)
                )

        required_samples = args.wikitext_offset + args.wikitext_samples
        if args.operator_scale_refit:
            required_samples = max(
                required_samples,
                args.operator_refit_offset + args.operator_refit_samples,
                args.operator_validation_offset
                + args.operator_validation_samples,
            )
        all_tokens, dataset_fingerprint, _bos = _wikitext_tokens(
            args.snapshot,
            samples=required_samples,
            sequence_length=args.sequence_length,
            local_files_only=args.local_files_only,
        )
        tokens = _select_token_window(
            all_tokens,
            offset=args.wikitext_offset,
            samples=args.wikitext_samples,
        )
        operator_fit_tokens = (
            _select_token_window(
                all_tokens,
                offset=args.operator_refit_offset,
                samples=args.operator_refit_samples,
            )
            if args.operator_scale_refit
            else None
        )
        operator_validation_tokens = (
            _select_token_window(
                all_tokens,
                offset=args.operator_validation_offset,
                samples=args.operator_validation_samples,
            )
            if args.operator_scale_refit
            else None
        )
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        operator_refit_metrics: dict[str, Any] = {}
        downstream_refit_metrics: dict[str, Any] = {}
        downstream_input_refit_metrics: dict[str, Any] = {}
        if args.operator_scale_refit:
            assert operator_fit_tokens is not None
            assert operator_validation_tokens is not None
            all_reconstruction_sets, operator_refit_metrics = (
                _operator_refit_sets(
                    teacher,
                    blocks,
                    operator_fit_tokens,
                    operator_validation_tokens,
                    all_reconstruction_sets,
                    device=args.device,
                    gate_grid_points=args.operator_gate_grid_points,
                    minimum_gate_multiplier=(
                        args.operator_minimum_gate_multiplier
                    ),
                    maximum_gate_multiplier=(
                        args.operator_maximum_gate_multiplier
                    ),
                    minimum_up_multiplier=(
                        args.operator_minimum_up_multiplier
                    ),
                    maximum_up_multiplier=(
                        args.operator_maximum_up_multiplier
                    ),
                )
            )
        if args.downstream_scale_refit:
            assert operator_fit_tokens is not None
            assert operator_validation_tokens is not None
            all_reconstruction_sets, downstream_refit_metrics = (
                _downstream_refit_sets(
                    teacher,
                    blocks,
                    operator_fit_tokens,
                    operator_validation_tokens,
                    all_reconstruction_sets,
                    device=args.device,
                    minimum_multiplier=(
                        args.downstream_minimum_output_multiplier
                    ),
                    maximum_multiplier=(
                        args.downstream_maximum_output_multiplier
                    ),
                )
            )
        if args.downstream_input_scale_refit:
            assert operator_fit_tokens is not None
            assert operator_validation_tokens is not None
            (
                all_reconstruction_sets,
                downstream_input_refit_metrics,
            ) = _downstream_input_refit_sets(
                teacher,
                blocks,
                operator_fit_tokens,
                operator_validation_tokens,
                all_reconstruction_sets,
                device=args.device,
                minimum_multiplier=(
                    args.downstream_minimum_output_multiplier
                ),
                maximum_multiplier=(
                    args.downstream_maximum_output_multiplier
                ),
                iterations=args.downstream_input_iterations,
                learning_rate=args.downstream_input_learning_rate,
            )
        if args.downstream_policy is not None:
            all_reconstruction_sets = _downstream_policy_sets(
                all_reconstruction_sets,
                args.downstream_policy,
            )
        if args.representation_policy is not None:
            all_reconstruction_sets[
                "hybrid_operator_policy_refit"
            ] = _hybrid_representation_set(
                all_reconstruction_sets,
                args.representation_policy,
            )
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        selection_results: dict[str, dict[str, Any]] = {}
        for name, threshold, selected_blocks in selection_specs:
            reconstruction_sets = {
                arm: _select_blocks(reconstructions, selected_blocks)
                for arm, reconstructions in all_reconstruction_sets.items()
            }
            kl_results = {}
            for arm, reconstructions in reconstruction_sets.items():
                evaluator = DenseKlSpliceEvaluator(
                    teacher,
                    reconstructions,
                    tokens,
                    device=args.device,
                    batch_size=1,
                    token_chunk_size=128,
                    teacher_cache_mode="cpu",
                )
                if not teacher_cache:
                    teacher_nll, teacher_cache = evaluator.teacher_cache_state()
                else:
                    evaluator.install_teacher_cache(
                        teacher_nll,
                        teacher_cache,
                    )
                kl_results[arm] = evaluator("full")
                del evaluator
            baseline_kl = kl_results["free_words"]
            candidate_kl = kl_results["corrected_codebook"]
            selection_result: dict[str, Any] = {
                "minimum_weighted_rmse_gain_fraction": threshold,
                "blocks": list(selected_blocks),
                "kl": {
                    arm: to_dict(value) for arm, value in kl_results.items()
                },
                "paired_candidate_minus_free_words": _paired_payload(
                    baseline_kl,
                    candidate_kl,
                    seed=args.seed,
                ),
            }
            if args.operator_scale_refit:
                baseline_refit = kl_results["free_words_operator_refit"]
                candidate_refit = kl_results[
                    "corrected_codebook_operator_refit"
                ]
                selection_result[
                    "paired_operator_candidate_minus_operator_free_words"
                ] = _paired_payload(
                    baseline_refit,
                    candidate_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_refit_minus_free_words"
                ] = _paired_payload(
                    baseline_kl,
                    baseline_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_refit_minus_candidate"
                ] = _paired_payload(
                    candidate_kl,
                    candidate_refit,
                    seed=args.seed,
                )
            if args.downstream_scale_refit:
                baseline_downstream_refit = kl_results[
                    "free_words_operator_downstream_refit"
                ]
                candidate_downstream_refit = kl_results[
                    "corrected_codebook_operator_downstream_refit"
                ]
                selection_result[
                    "paired_downstream_candidate_minus_downstream_free_words"
                ] = _paired_payload(
                    baseline_downstream_refit,
                    candidate_downstream_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_downstream_refit_minus_operator_refit"
                ] = _paired_payload(
                    baseline_refit,
                    baseline_downstream_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_downstream_refit_minus_operator_refit"
                ] = _paired_payload(
                    candidate_refit,
                    candidate_downstream_refit,
                    seed=args.seed,
                )
            if args.downstream_input_scale_refit:
                baseline_input_refit = kl_results[
                    "free_words_operator_downstream_input_refit"
                ]
                candidate_input_refit = kl_results[
                    "corrected_codebook_operator_downstream_input_refit"
                ]
                baseline_joint_refit = kl_results[
                    "free_words_operator_downstream_joint_refit"
                ]
                candidate_joint_refit = kl_results[
                    "corrected_codebook_operator_downstream_joint_refit"
                ]
                selection_result[
                    "paired_input_candidate_minus_input_free_words"
                ] = _paired_payload(
                    baseline_input_refit,
                    candidate_input_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_joint_candidate_minus_joint_free_words"
                ] = _paired_payload(
                    baseline_joint_refit,
                    candidate_joint_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_input_refit_minus_operator_refit"
                ] = _paired_payload(
                    baseline_refit,
                    baseline_input_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_input_refit_minus_operator_refit"
                ] = _paired_payload(
                    candidate_refit,
                    candidate_input_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_joint_refit_minus_operator_refit"
                ] = _paired_payload(
                    baseline_refit,
                    baseline_joint_refit,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_joint_refit_minus_operator_refit"
                ] = _paired_payload(
                    candidate_refit,
                    candidate_joint_refit,
                    seed=args.seed,
                )
            if args.downstream_policy is not None:
                baseline_policy = kl_results[
                    "free_words_operator_policy_refit"
                ]
                candidate_policy = kl_results[
                    "corrected_codebook_operator_policy_refit"
                ]
                selection_result[
                    "paired_policy_candidate_minus_policy_free_words"
                ] = _paired_payload(
                    baseline_policy,
                    candidate_policy,
                    seed=args.seed,
                )
                selection_result[
                    "paired_free_words_policy_minus_operator_refit"
                ] = _paired_payload(
                    baseline_refit,
                    baseline_policy,
                    seed=args.seed,
                )
                selection_result[
                    "paired_candidate_policy_minus_operator_refit"
                ] = _paired_payload(
                    candidate_refit,
                    candidate_policy,
                    seed=args.seed,
                )
            if args.representation_policy is not None:
                hybrid_policy = kl_results[
                    "hybrid_operator_policy_refit"
                ]
                selection_result[
                    "paired_hybrid_minus_free_words_policy"
                ] = _paired_payload(
                    baseline_policy,
                    hybrid_policy,
                    seed=args.seed,
                )
                selection_result[
                    "paired_hybrid_minus_corrected_codebook_policy"
                ] = _paired_payload(
                    candidate_policy,
                    hybrid_policy,
                    seed=args.seed,
                )
            selection_results[name] = selection_result
        del teacher

    def candidate_cost(free_rows: int) -> Any:
        if args.codebook_mode == "product":
            return mixed_right_product_codebook_bit_cost(
                matrix_shape[0],
                matrix_shape[1],
                rank,
                right_free_rows=free_rows,
                right_index_width=args.index_width,
                scale_width=16,
            )
        if args.codebook_mode == "linear":
            return mixed_right_linear_codebook_bit_cost(
                matrix_shape[0],
                matrix_shape[1],
                rank,
                right_free_rows=free_rows,
                right_index_width=args.index_width,
                scale_width=16,
            )
        if free_rows:
            return mixed_right_corrected_codebook_bit_cost(
                matrix_shape[0],
                matrix_shape[1],
                rank,
                right_free_rows=free_rows,
                right_index_width=args.index_width,
                right_flip_bits=args.correction_bits,
                scale_width=16,
            )
        return corrected_asymmetric_codebook_bit_cost(
            matrix_shape[0],
            matrix_shape[1],
            rank,
            left_index_width=None,
            right_index_width=args.index_width,
            right_flip_bits=args.correction_bits,
            scale_width=16,
        )

    costs_by_projection = {
        projection: candidate_cost(right_free_rows_by_projection[projection])
        for projection in projections
    }
    uniform_free_rows = len(set(right_free_rows_by_projection.values())) == 1
    represented_projection_count = 1 if uniform_free_rows else len(projections)
    cost = next(iter(costs_by_projection.values()))
    fixed_factorization_axis = (
        0
        if transpose_by_projection
        and all(transpose_by_projection.values())
        else 1
    )
    outlier_value_extent = (
        matrix_shape[1] if fixed_factorization_axis == 0 else matrix_shape[0]
    )
    outlier_index_extent = (
        matrix_shape[0] if fixed_factorization_axis == 0 else matrix_shape[1]
    )
    outlier_cost = outlier_bit_cost(
        outlier_value_extent,
        len(args.fixed_outlier_indices),
        value_bits=16,
        index_bits=max(1, (outlier_index_extent - 1).bit_length()),
    )
    baseline_cost = factor_bit_cost(
        matrix_shape[0],
        matrix_shape[1],
        args.baseline_rank,
        scale_bits=16,
    )
    candidate_bit_cost_by_projection = {
        projection: asdict(projection_cost)
        for projection, projection_cost in costs_by_projection.items()
    }
    candidate_bit_cost = asdict(cost) if uniform_free_rows else None
    baseline_bit_cost = asdict(baseline_cost)
    bit_costs = [*candidate_bit_cost_by_projection.values(), baseline_bit_cost]
    if candidate_bit_cost is not None:
        bit_costs.append(candidate_bit_cost)
    for bit_cost in bit_costs:
        bit_cost["outlier_value_bits"] = outlier_cost.outlier_value_bits
        bit_cost["outlier_index_bits"] = outlier_cost.outlier_index_bits
    candidate_total_bits = (
        cost.total + outlier_cost.total
        if uniform_free_rows
        else sum(
            projection_cost.total + outlier_cost.total
            for projection_cost in costs_by_projection.values()
        )
    )
    baseline_total_bits = represented_projection_count * (
        baseline_cost.total + outlier_cost.total
    )
    represented_elements = (
        represented_projection_count * matrix_shape[0] * matrix_shape[1]
    )
    reconstruction_export = None
    if args.export_reconstruction_set is not None:
        if args.export_arm not in all_reconstruction_sets:
            raise ValueError(
                f"reconstruction export arm is unavailable: {args.export_arm}"
            )
        reconstruction_export = _export_reconstruction_set(
            args.export_reconstruction_set,
            args.export_arm,
            all_reconstruction_sets[args.export_arm],
        )
    product_component_export = None
    if args.export_product_components is not None:
        product_component_export = _export_product_components(
            args.export_product_components,
            product_components,
        )

    output: dict[str, Any] = {
        "schema_version": 18,
        "status": "completed",
        "role": "analysis-only sign-code splice gate",
        "reconstruction_export": reconstruction_export,
        "product_component_export": product_component_export,
        "reconstruction_cache": {
            "enabled": reconstruction_cache is not None,
            "root": (
                str(args.reconstruction_cache)
                if args.reconstruction_cache is not None
                else None
            ),
            "hits": cache_hits,
            "misses": cache_misses,
            "keys_by_unit": cache_keys,
            "model_sha256": (
                cache_model_sha256
                if reconstruction_cache is not None
                else None
            ),
            "calibration_manifest_sha256": (
                cache_calibration_manifest_sha256
                if reconstruction_cache is not None
                else None
            ),
            "calibration_state_sha256": (
                cache_calibration_state_sha256
                if reconstruction_cache is not None
                else None
            ),
        },
        "model_source": MODEL_SOURCE,
        "model_revision": args.model_revision,
        "blocks": list(blocks),
        "projection": projection_paths[0] if len(projection_paths) == 1 else None,
        "projections": list(projection_paths),
        "transposed_for_factorization": (
            next(iter(transpose_by_projection.values()))
            if len(set(transpose_by_projection.values())) == 1
            else None
        ),
        "transposed_by_projection": transpose_by_projection,
        "factorization_shape": list(matrix_shape),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_slice_hash": _token_hash(tokens),
        "wikitext_samples": args.wikitext_samples,
        "wikitext_offset": args.wikitext_offset,
        "sequence_length": args.sequence_length,
        "teacher_baseline_nll": teacher_nll,
        "baseline": {
            "rank": args.baseline_rank,
            "bit_cost": baseline_bit_cost,
            "total_bits": baseline_total_bits,
            "actual_bpw": baseline_total_bits / represented_elements,
        },
        "candidate": {
            "codebook_mode": args.codebook_mode,
            "index_width": args.index_width,
            "corrections_per_word": args.corrections_per_word,
            "correction_bits": args.correction_bits,
            "rank": rank,
            "right_free_rows": (
                next(iter(right_free_rows_by_projection.values()))
                if uniform_free_rows
                else None
            ),
            "right_free_rows_by_projection": right_free_rows_by_projection,
            "linear_assignment_sweeps": args.linear_assignment_sweeps,
            "corrected_assignment_candidates": (
                args.corrected_assignment_candidates
            ),
            "bit_cost": candidate_bit_cost,
            "bit_cost_by_projection": candidate_bit_cost_by_projection,
            "total_bits": candidate_total_bits,
            "actual_bpw": candidate_total_bits / represented_elements,
            "index_metrics_by_block": candidate_index_metrics,
        },
        "fixed_outlier_columns": {
            "count": len(args.fixed_outlier_indices),
            "indices": list(args.fixed_outlier_indices),
            "factorization_axis": (
                "row" if fixed_factorization_axis == 0 else "column"
            ),
            "bit_cost": asdict(outlier_cost),
            "projection_count_charged": represented_projection_count,
            "applied_to": ["free_words", "corrected_codebook"],
        },
        "reconstruction_by_block": reconstruction_metrics,
        "search_by_block": search_metrics,
        "functional_payload_search": {
            "enabled": args.functional_payload_search,
            "fit_offset": args.functional_payload_fit_offset,
            "fit_samples": args.functional_payload_fit_samples,
            "validation_offset": args.functional_payload_validation_offset,
            "validation_samples": args.functional_payload_validation_samples,
            "candidate_words_per_pass": args.functional_payload_candidate_words,
        },
        "selection_results": selection_results,
        "operator_scale_refit": {
            "enabled": args.operator_scale_refit,
            "fit_offset": args.operator_refit_offset,
            "fit_samples": args.operator_refit_samples,
            "validation_offset": args.operator_validation_offset,
            "validation_samples": args.operator_validation_samples,
            "gate_grid_points": args.operator_gate_grid_points,
            "minimum_gate_multiplier": (
                args.operator_minimum_gate_multiplier
            ),
            "maximum_gate_multiplier": (
                args.operator_maximum_gate_multiplier
            ),
            "minimum_up_multiplier": args.operator_minimum_up_multiplier,
            "maximum_up_multiplier": args.operator_maximum_up_multiplier,
            "metrics": operator_refit_metrics,
        },
        "downstream_scale_refit": {
            "enabled": args.downstream_scale_refit,
            "minimum_output_multiplier": (
                args.downstream_minimum_output_multiplier
            ),
            "maximum_output_multiplier": (
                args.downstream_maximum_output_multiplier
            ),
            "metrics": downstream_refit_metrics,
        },
        "downstream_input_scale_refit": {
            "enabled": args.downstream_input_scale_refit,
            "minimum_multiplier": (
                args.downstream_minimum_output_multiplier
            ),
            "maximum_multiplier": (
                args.downstream_maximum_output_multiplier
            ),
            "iterations": args.downstream_input_iterations,
            "learning_rate": args.downstream_input_learning_rate,
            "metrics": downstream_input_refit_metrics,
        },
        "downstream_policy": (
            {
                str(block): choice
                for block, choice in args.downstream_policy
            }
            if args.downstream_policy is not None
            else None
        ),
        "representation_policy": (
            {
                str(block): choice
                for block, choice in args.representation_policy
            }
            if args.representation_policy is not None
            else None
        ),
    }
    if args.selection_thresholds is None:
        full = selection_results["full"]
        output["kl"] = full["kl"]
        output["paired_candidate_minus_free_words"] = full[
            "paired_candidate_minus_free_words"
        ]
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "reconstruction_by_block": output["reconstruction_by_block"],
                "selection_results": output["selection_results"],
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
