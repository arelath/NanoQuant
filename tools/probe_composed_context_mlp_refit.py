"""Compare teacher- and composed-student-context zero-bit MLP refits."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_corrected_codebook_splice import (
    _downstream_input_refit_sets,
    _downstream_refit_sets,
    _dtype,
    _export_reconstruction_set,
    _gated_down_outputs,
    _module_at_path,
    _operator_refit_sets,
    _paired_payload,
    _replace_weights,
    _select_token_window,
)
from probe_sign_word_codebook import PROJECTION_PATHS
from probe_tuned_mlp_scale_refit import (
    MODEL_SOURCE,
    PINNED_MODEL_REVISION,
)
from torch.utils.hooks import RemovableHandle

from nanoquant.application.kl_budget import KlBudgetArmResult, KlSequenceResult
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
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
from nanoquant.quality_evaluation import _wikitext_tokens

CONTEXTS = ("teacher_function", "student_function", "student_state")
MLP_PATHS = (
    PROJECTION_PATHS["gate"],
    PROJECTION_PATHS["up"],
    PROJECTION_PATHS["down"],
)
DEFAULT_POLICIES = (
    (0, "output"),
    (18, "joint"),
    (23, "joint"),
    (4, "joint"),
    (19, "joint"),
    (21, "joint"),
    (25, "joint"),
)


@dataclass(frozen=True, slots=True)
class MlpContextCapture:
    mlp_inputs: dict[int, torch.Tensor]
    post_attention_residuals: dict[int, torch.Tensor]
    mlp_outputs: dict[int, torch.Tensor]
    block_outputs: dict[int, torch.Tensor]
    post_feedforward_norm_kinds: dict[int, str]
    post_feedforward_norm_multipliers: dict[int, torch.Tensor]
    post_feedforward_norm_epsilons: dict[int, float]


def _parse_policy(value: str) -> tuple[tuple[int, str], ...]:
    choices = {"operator", "output", "input", "joint"}
    result = []
    for item in value.split(","):
        if not item.strip():
            continue
        block_text, choice = item.split(":", maxsplit=1)
        block = int(block_text)
        if block < 0 or choice not in choices:
            raise argparse.ArgumentTypeError("context policy contains an invalid placement")
        result.append((block, choice))
    if not result or len({block for block, _choice in result}) != len(result):
        raise argparse.ArgumentTypeError("context policy blocks must be non-empty and unique")
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-directory", type=Path)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument(
        "--policy",
        type=_parse_policy,
        default=DEFAULT_POLICIES,
    )
    parser.add_argument("--fit-offset", type=int, default=380)
    parser.add_argument("--fit-samples", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=384)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--evaluation-offset", type=int, default=388)
    parser.add_argument("--evaluation-samples", type=int, default=24)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gate-grid-points", type=int, default=71)
    parser.add_argument("--minimum-gate-multiplier", type=float, default=0.25)
    parser.add_argument("--maximum-gate-multiplier", type=float, default=2.0)
    parser.add_argument("--minimum-up-multiplier", type=float, default=0.1)
    parser.add_argument("--maximum-up-multiplier", type=float, default=8.0)
    parser.add_argument("--minimum-down-multiplier", type=float, default=0.25)
    parser.add_argument("--maximum-down-multiplier", type=float, default=4.0)
    parser.add_argument("--input-iterations", type=int, default=50)
    parser.add_argument("--input-learning-rate", type=float, default=0.25)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-global-tuning", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace, block_count: int) -> None:
    inventories = (
        set(range(args.fit_offset, args.fit_offset + args.fit_samples)),
        set(range(args.validation_offset, args.validation_offset + args.validation_samples)),
        set(range(args.evaluation_offset, args.evaluation_offset + args.evaluation_samples)),
    )
    blocks = tuple(block for block, _choice in args.policy)
    if (
        min(
            args.fit_samples,
            args.validation_samples,
            args.evaluation_samples,
            args.sequence_length - 1,
        )
        <= 0
        or min(args.fit_offset, args.validation_offset, args.evaluation_offset) < 0
        or inventories[0] & inventories[1]
        or inventories[0] & inventories[2]
        or inventories[1] & inventories[2]
        or any(block >= block_count for block in blocks)
    ):
        raise ValueError("composed-context probe protocol is invalid")


def _tensor_from_output(output: object) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError("decoder block hook did not receive a tensor output")
    return value


@torch.inference_mode()
def _capture_mlp_context(
    model: torch.nn.Module,
    blocks: tuple[int, ...],
    tokens: torch.Tensor,
    *,
    device: str,
) -> MlpContextCapture:
    base = getattr(model, "model", None)
    decoder = getattr(base, "layers", None)
    if not isinstance(decoder, torch.nn.ModuleList):
        raise TypeError("model does not expose a supported decoder block stack")
    captured: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {block: [] for block in blocks}
        for name in (
            "mlp_inputs",
            "post_attention_residuals",
            "mlp_outputs",
            "block_outputs",
        )
    }
    norm_kinds: dict[int, str] = {}
    norm_multipliers: dict[int, torch.Tensor] = {}
    norm_epsilons: dict[int, float] = {}
    handles: list[RemovableHandle] = []

    def append(kind: str, block: int, value: torch.Tensor) -> None:
        captured[kind][block].append(
            value.detach().reshape(-1, value.shape[-1]).to(
                device="cpu",
                dtype=torch.bfloat16,
            )
        )

    for block_index in blocks:
        block = decoder[block_index]
        gate = _module_at_path(block, PROJECTION_PATHS["gate"])
        mlp = getattr(block, "mlp", None)
        residual_capture_module = getattr(block, "pre_feedforward_layernorm", None)
        if residual_capture_module is None:
            residual_capture_module = getattr(block, "post_attention_layernorm", None)
        if not isinstance(residual_capture_module, torch.nn.Module):
            raise TypeError("decoder block has no pre-MLP residual boundary")
        if not isinstance(mlp, torch.nn.Module):
            raise TypeError("decoder block has no MLP module")
        post_feedforward_norm = getattr(block, "post_feedforward_layernorm", None)
        if post_feedforward_norm is None:
            norm_kinds[block_index] = "identity"
            norm_multipliers[block_index] = torch.ones(1)
            norm_epsilons[block_index] = 0.0
        elif "rmsnorm" in post_feedforward_norm.__class__.__name__.lower():
            weight = getattr(post_feedforward_norm, "weight", None)
            epsilon = getattr(post_feedforward_norm, "eps", None)
            if not isinstance(weight, torch.Tensor) or not isinstance(epsilon, float):
                raise TypeError("post-feed-forward RMS norm has an unsupported form")
            norm_kinds[block_index] = "rms"
            multiplier = weight.detach().float().cpu()
            if "gemma" in post_feedforward_norm.__class__.__name__.lower():
                multiplier = 1 + multiplier
            norm_multipliers[block_index] = multiplier
            norm_epsilons[block_index] = epsilon
        else:
            raise TypeError("post-feed-forward normalization is unsupported")

        def gate_hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            *,
            index: int = block_index,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("MLP input hook did not receive a tensor")
            append("mlp_inputs", index, inputs[0])

        def residual_hook(
            _module: torch.nn.Module,
            inputs: tuple[object, ...],
            *,
            index: int = block_index,
        ) -> None:
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                raise TypeError("post-attention residual hook did not receive a tensor")
            append("post_attention_residuals", index, inputs[0])

        def block_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            index: int = block_index,
        ) -> None:
            append("block_outputs", index, _tensor_from_output(output))

        def mlp_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: object,
            *,
            index: int = block_index,
        ) -> None:
            append("mlp_outputs", index, _tensor_from_output(output))

        handles.extend(
            (
                gate.register_forward_pre_hook(gate_hook),
                residual_capture_module.register_forward_pre_hook(residual_hook),
                mlp.register_forward_hook(mlp_hook),
                block.register_forward_hook(block_hook),
            )
        )
    try:
        for index in range(tokens.shape[0]):
            cast(Any, model)(
                input_ids=tokens[index : index + 1].to(device),
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()
    for kind in captured.values():
        if any(len(kind[block]) != tokens.shape[0] for block in blocks):
            raise ValueError("MLP context capture did not cover every requested sequence")
    return MlpContextCapture(
        **{
            name: {block: torch.cat(values[block]) for block in blocks}
            for name, values in captured.items()
        },
        post_feedforward_norm_kinds=norm_kinds,
        post_feedforward_norm_multipliers=norm_multipliers,
        post_feedforward_norm_epsilons=norm_epsilons,
    )


def _normalized_rmse(target: torch.Tensor, candidate: torch.Tensor) -> float:
    if target.shape != candidate.shape or target.ndim != 2:
        raise ValueError("hidden-state RMSE tensors must be aligned matrices")
    target32 = target.float()
    return math.sqrt(
        float((candidate.float() - target32).square().sum())
        / max(float(target32.square().sum()), 1e-12)
    )


def _paired_metric_payload(
    baseline: KlBudgetArmResult,
    candidate: KlBudgetArmResult,
    attribute: str,
    *,
    seed: int = 0,
    resamples: int = 10_000,
) -> dict[str, object]:
    before = baseline.sequences
    after = candidate.sequences
    if (
        len(before) != len(after)
        or not before
        or resamples <= 0
    ):
        raise ValueError("paired metric requires aligned sequence results")

    def weighted(indices: list[int], values: tuple[KlSequenceResult, ...]) -> float:
        tokens = math.fsum(float(values[index].token_count) for index in indices)
        return math.fsum(
            float(getattr(values[index], attribute))
            * float(values[index].token_count)
            for index in indices
        ) / tokens

    generator = random.Random(seed)
    deltas = []
    for _sample in range(resamples):
        indices = [generator.randrange(len(before)) for _ in before]
        deltas.append(weighted(indices, after) - weighted(indices, before))
    deltas.sort()
    point = float(getattr(candidate, attribute)) - float(getattr(baseline, attribute))
    lower = deltas[int(0.025 * resamples)]
    upper = deltas[int(0.975 * resamples) - 1]
    return {
        "point_delta": point,
        "relative_delta": point / float(getattr(baseline, attribute)),
        "lower_delta": lower,
        "upper_delta": upper,
        "confidence": 0.95,
        "resamples": resamples,
        "improved_with_confidence": point < 0 and upper < 0,
    }


def _state_targets(
    teacher: MlpContextCapture,
    student: MlpContextCapture,
) -> dict[int, torch.Tensor]:
    if teacher.block_outputs.keys() != student.post_attention_residuals.keys():
        raise ValueError("state-recovery capture inventories differ")
    result = {}
    for block in teacher.block_outputs:
        desired = (
            teacher.block_outputs[block].float()
            - student.post_attention_residuals[block].float()
        )
        kind = student.post_feedforward_norm_kinds[block]
        if kind == "identity":
            result[block] = desired.to(torch.bfloat16)
            continue
        if kind != "rms":
            raise ValueError("unsupported post-feed-forward normalization kind")
        multiplier = student.post_feedforward_norm_multipliers[block].float()
        inverse_direction = desired / multiplier.clamp_min(1e-12)
        source_rms = student.mlp_outputs[block].float().square().mean(
            dim=-1,
            keepdim=True,
        ).sqrt()
        target_rms = inverse_direction.square().mean(dim=-1, keepdim=True).sqrt()
        result[block] = (
            inverse_direction * source_rms / target_rms.clamp_min(1e-12)
        ).to(torch.bfloat16)
    return result


def _apply_post_feedforward_norm(
    value: torch.Tensor,
    capture: MlpContextCapture,
    block: int,
) -> torch.Tensor:
    kind = capture.post_feedforward_norm_kinds[block]
    if kind == "identity":
        return value.float()
    if kind != "rms":
        raise ValueError("unsupported post-feed-forward normalization kind")
    value32 = value.float()
    normalized = value32 * torch.rsqrt(
        value32.square().mean(dim=-1, keepdim=True)
        + capture.post_feedforward_norm_epsilons[block]
    )
    return normalized * capture.post_feedforward_norm_multipliers[block].float()


def _fit_context(
    context: str,
    teacher: torch.nn.Module,
    blocks: tuple[int, ...],
    fit_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    baseline: SpliceReconstructionSet,
    teacher_fit: MlpContextCapture,
    teacher_validation: MlpContextCapture,
    student_fit: MlpContextCapture,
    student_validation: MlpContextCapture,
    args: argparse.Namespace,
) -> tuple[dict[str, SpliceReconstructionSet], dict[str, object]]:
    if context not in CONTEXTS:
        raise ValueError(f"unknown composed-context arm: {context}")
    fit_capture = teacher_fit if context == "teacher_function" else student_fit
    validation_capture = (
        teacher_validation if context == "teacher_function" else student_validation
    )
    fit_targets = (
        _state_targets(teacher_fit, student_fit)
        if context == "student_state"
        else None
    )
    validation_targets = (
        _state_targets(teacher_validation, student_validation)
        if context == "student_state"
        else None
    )
    reconstruction_sets = {
        "free_words": baseline,
        "corrected_codebook": baseline,
    }
    reconstruction_sets, operator = _operator_refit_sets(
        teacher,
        blocks,
        fit_tokens,
        validation_tokens,
        reconstruction_sets,
        device=args.device,
        gate_grid_points=args.gate_grid_points,
        minimum_gate_multiplier=args.minimum_gate_multiplier,
        maximum_gate_multiplier=args.maximum_gate_multiplier,
        minimum_up_multiplier=args.minimum_up_multiplier,
        maximum_up_multiplier=args.maximum_up_multiplier,
        fit_inputs_by_block=fit_capture.mlp_inputs,
        validation_inputs_by_block=validation_capture.mlp_inputs,
    )
    reconstruction_sets, output = _downstream_refit_sets(
        teacher,
        blocks,
        fit_tokens,
        validation_tokens,
        reconstruction_sets,
        device=args.device,
        minimum_multiplier=args.minimum_down_multiplier,
        maximum_multiplier=args.maximum_down_multiplier,
        fit_inputs_by_block=fit_capture.mlp_inputs,
        validation_inputs_by_block=validation_capture.mlp_inputs,
        fit_targets_by_block=fit_targets,
        validation_targets_by_block=validation_targets,
    )
    reconstruction_sets, input_refit = _downstream_input_refit_sets(
        teacher,
        blocks,
        fit_tokens,
        validation_tokens,
        reconstruction_sets,
        device=args.device,
        minimum_multiplier=args.minimum_down_multiplier,
        maximum_multiplier=args.maximum_down_multiplier,
        iterations=args.input_iterations,
        learning_rate=args.input_learning_rate,
        fit_inputs_by_block=fit_capture.mlp_inputs,
        validation_inputs_by_block=validation_capture.mlp_inputs,
        fit_targets_by_block=fit_targets,
        validation_targets_by_block=validation_targets,
    )
    return reconstruction_sets, {
        "operator": operator["free_words"],
        "output": output["free_words_operator_refit"],
        "input_and_joint": input_refit["free_words_operator_refit"],
    }


def _policy_source(choice: str) -> str:
    return {
        "operator": "free_words_operator_refit",
        "output": "free_words_operator_downstream_refit",
        "input": "free_words_operator_downstream_input_refit",
        "joint": "free_words_operator_downstream_joint_refit",
    }[choice]


def _single_block_candidate(
    baseline: SpliceReconstructionSet,
    fitted: dict[str, SpliceReconstructionSet],
    block: int,
    choice: str,
) -> SpliceReconstructionSet:
    source = {
        item.layer: item.weight
        for item in fitted[_policy_source(choice)].layers
        if item.layer.block.index == block and item.layer.path in MLP_PATHS
    }
    if len(source) != 3:
        raise ValueError("single-block candidate does not contain three MLP layers")
    return _replace_weights(baseline, source)


def _selected_mlp_only(
    candidate: SpliceReconstructionSet,
    block: int,
) -> SpliceReconstructionSet:
    layers = tuple(
        item
        for item in candidate.layers
        if item.layer.block.index == block and item.layer.path in MLP_PATHS
    )
    if len(layers) != 3:
        raise ValueError("MLP overlay export inventory is incomplete")
    return SpliceReconstructionSet(layers, (), ())


def _block_state_diagnostic(
    candidate: SpliceReconstructionSet,
    block: int,
    student: MlpContextCapture,
    teacher: MlpContextCapture,
    *,
    device: str,
) -> dict[str, float]:
    by_layer = {item.layer: item for item in candidate.layers}
    weights = []
    for path in (PROJECTION_PATHS["gate"], PROJECTION_PATHS["up"], PROJECTION_PATHS["down"]):
        item = by_layer.get(LayerId(BlockId(block), path))
        if item is None:
            raise ValueError("block-state diagnostic reconstruction is incomplete")
        weights.append(item.weight)
    mlp_output = _gated_down_outputs(
        student.mlp_inputs[block],
        weights[0],
        weights[1],
        weights[2],
        device=device,
    ).cpu()
    predicted_block = student.post_attention_residuals[block].float() + _apply_post_feedforward_norm(
        mlp_output,
        student,
        block,
    )
    teacher_block = teacher.block_outputs[block]
    result = {
        "mlp_input_teacher_student_normalized_rmse": _normalized_rmse(
            teacher.mlp_inputs[block],
            student.mlp_inputs[block],
        ),
        "post_attention_residual_teacher_student_normalized_rmse": _normalized_rmse(
            teacher.post_attention_residuals[block],
            student.post_attention_residuals[block],
        ),
        "predicted_block_output_teacher_normalized_rmse": _normalized_rmse(
            teacher_block,
            predicted_block,
        ),
    }
    del mlp_output, predicted_block
    return result


def run(args: argparse.Namespace) -> int:
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    _validate_args(args, adapter.decoder_block_count_from_config(config))
    blocks = tuple(block for block, _choice in args.policy)
    policies = dict(args.policy)
    required_samples = max(
        args.fit_offset + args.fit_samples,
        args.validation_offset + args.validation_samples,
        args.evaluation_offset + args.evaluation_samples,
    )
    all_tokens, dataset_fingerprint, bos_token_id = _wikitext_tokens(
        args.snapshot,
        samples=required_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    fit_tokens = _select_token_window(all_tokens, offset=args.fit_offset, samples=args.fit_samples)
    validation_tokens = _select_token_window(
        all_tokens,
        offset=args.validation_offset,
        samples=args.validation_samples,
    )
    evaluation_tokens = _select_token_window(
        all_tokens,
        offset=args.evaluation_offset,
        samples=args.evaluation_samples,
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
            use_global_tuning=args.use_global_tuning,
        )
        loaded.model.eval()
        baseline = collect_splice_reconstructions(loaded)
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        student_fit = _capture_mlp_context(
            loaded.model,
            blocks,
            fit_tokens,
            device=args.device,
        )
        student_validation = _capture_mlp_context(
            loaded.model,
            blocks,
            validation_tokens,
            device=args.device,
        )
        del loaded
        gc.collect()
        torch.cuda.empty_cache()

        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        teacher_fit = _capture_mlp_context(
            teacher,
            blocks,
            fit_tokens,
            device=args.device,
        )
        teacher_validation = _capture_mlp_context(
            teacher,
            blocks,
            validation_tokens,
            device=args.device,
        )

        fitted_by_context: dict[str, dict[str, SpliceReconstructionSet]] = {}
        fit_metrics: dict[str, object] = {}
        candidates: dict[str, SpliceReconstructionSet] = {}
        state_diagnostics: dict[str, dict[str, dict[str, float]]] = {
            context: {} for context in CONTEXTS
        }
        baseline_state_diagnostics = {
            str(block): _block_state_diagnostic(
                baseline,
                block,
                student_validation,
                teacher_validation,
                device=args.device,
            )
            for block in blocks
        }
        for context in CONTEXTS:
            fitted, metrics = _fit_context(
                context,
                teacher,
                blocks,
                fit_tokens,
                validation_tokens,
                baseline,
                teacher_fit,
                teacher_validation,
                student_fit,
                student_validation,
                args,
            )
            fitted_by_context[context] = fitted
            fit_metrics[context] = metrics
            for block in blocks:
                key = f"{context}:block{block}:{policies[block]}"
                candidate = _single_block_candidate(
                    baseline,
                    fitted,
                    block,
                    policies[block],
                )
                candidates[key] = candidate
                state_diagnostics[context][str(block)] = _block_state_diagnostic(
                    candidate,
                    block,
                    student_validation,
                    teacher_validation,
                    device=args.device,
                )

        selected_student_context = {
            block: min(
                ("student_function", "student_state"),
                key=lambda context: float(
                    state_diagnostics[context][str(block)][
                        "predicted_block_output_teacher_normalized_rmse"
                    ]
                ),
            )
            for block in blocks
        }
        evaluated_candidates = {
            key: candidate
            for key, candidate in candidates.items()
            if key.startswith("teacher_function:")
            or any(
                key.startswith(f"{selected_student_context[block]}:block{block}:")
                for block in blocks
            )
        }
        results: dict[str, KlBudgetArmResult] = {}
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        for key, reconstruction_set in {"baseline": baseline, **evaluated_candidates}.items():
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_set,
                evaluation_tokens,
                device=args.device,
                batch_size=1,
                token_chunk_size=128,
                teacher_cache_mode="cpu",
            )
            if not teacher_cache:
                teacher_nll, teacher_cache = evaluator.teacher_cache_state()
            else:
                evaluator.install_teacher_cache(teacher_nll, teacher_cache)
            results[key] = evaluator("full")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del teacher

    baseline_result = results["baseline"]
    kl_comparisons = {
        key: _paired_payload(baseline_result, result, seed=0)
        for key, result in results.items()
        if key != "baseline"
    }
    nll_comparisons = {
        key: _paired_metric_payload(
            baseline_result,
            result,
            "negative_log_likelihood",
        )
        for key, result in results.items()
        if key != "baseline"
    }
    context_comparisons = {}
    for block in blocks:
        student_key = (
            f"{selected_student_context[block]}:block{block}:{policies[block]}"
        )
        teacher_key = f"teacher_function:block{block}:{policies[block]}"
        context_comparisons[str(block)] = {
            "student_context": selected_student_context[block],
            "student_minus_teacher_nll": _paired_metric_payload(
                results[teacher_key],
                results[student_key],
                "negative_log_likelihood",
            ),
            "student_minus_teacher_kl": _paired_payload(
                results[teacher_key],
                results[student_key],
                seed=0,
            ),
        }
    exports = {}
    if args.export_directory is not None:
        for key, nll_comparison in nll_comparisons.items():
            kl_comparison = kl_comparisons[key]
            if (
                not bool(nll_comparison["improved_with_confidence"])
                or float(kl_comparison["lower_delta"]) > 0
            ):
                continue
            block = int(key.split(":")[1].removeprefix("block"))
            destination = args.export_directory / key.replace(":", "-")
            exports[key] = _export_reconstruction_set(
                destination,
                key,
                _selected_mlp_only(evaluated_candidates[key], block),
            )
    atomic_write_json(
        args.output,
        {
            "schema_version": 2,
            "status": "completed",
            "role": "analysis-only composed-context MLP refit ablation",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "policy": {str(block): choice for block, choice in args.policy},
            "protocol": {
                "sequence_length": args.sequence_length,
                "fit_offset": args.fit_offset,
                "fit_samples": args.fit_samples,
                "validation_offset": args.validation_offset,
                "validation_samples": args.validation_samples,
                "evaluation_offset": args.evaluation_offset,
                "evaluation_samples": args.evaluation_samples,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "fit_token_hash": _token_hash(fit_tokens),
                "validation_token_hash": _token_hash(validation_tokens),
                "evaluation_token_hash": _token_hash(evaluation_tokens),
            },
            "settings": {
                "gate_grid_points": args.gate_grid_points,
                "minimum_gate_multiplier": args.minimum_gate_multiplier,
                "maximum_gate_multiplier": args.maximum_gate_multiplier,
                "minimum_up_multiplier": args.minimum_up_multiplier,
                "maximum_up_multiplier": args.maximum_up_multiplier,
                "minimum_down_multiplier": args.minimum_down_multiplier,
                "maximum_down_multiplier": args.maximum_down_multiplier,
                "input_iterations": args.input_iterations,
                "input_learning_rate": args.input_learning_rate,
            },
            "fit_metrics": fit_metrics,
            "baseline_state_diagnostics": baseline_state_diagnostics,
            "state_diagnostics": state_diagnostics,
            "selected_student_context": {
                str(block): context for block, context in selected_student_context.items()
            },
            "functional": {key: to_dict(value) for key, value in results.items()},
            "paired_candidate_minus_baseline_nll": nll_comparisons,
            "paired_candidate_minus_baseline_kl": kl_comparisons,
            "paired_selected_student_minus_teacher": context_comparisons,
            "exports": exports,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
