"""Coordinate-refit an accepted dense MLP overlay in composed-student context."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import (
    MLP_PATHS,
    _capture_mlp_context,
    _fit_context,
    _single_block_candidate,
)
from probe_corrected_codebook_splice import (
    _dtype,
    _export_reconstruction_set,
    _replace_weights,
    _select_token_window,
)
from probe_mlp_policy_frozen_transfer import (
    _evaluate_per_sequence,
    _install_dense_linear,
    _load_overlay,
)
from probe_tuned_mlp_scale_refit import MODEL_SOURCE, PINNED_MODEL_REVISION

from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    SpliceReconstructionSet,
    collect_splice_reconstructions,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

DEFAULT_POLICY = (
    (0, "output", "student_function"),
    (17, "joint", "student_function"),
    (18, "joint", "teacher_function"),
    (23, "joint", "student_function"),
    (24, "joint", "teacher_function"),
)


def _parse_policy(value: str) -> tuple[tuple[int, str, str], ...]:
    choices = {"operator", "output", "input", "joint"}
    contexts = {"teacher_function", "student_function"}
    result = []
    for item in value.split(","):
        if not item.strip():
            continue
        block_text, choice, context = item.split(":", maxsplit=2)
        block = int(block_text)
        if block < 0 or choice not in choices or context not in contexts:
            raise argparse.ArgumentTypeError("coordinate policy contains an invalid placement")
        result.append((block, choice, context))
    if not result or len({block for block, _choice, _context in result}) != len(result):
        raise argparse.ArgumentTypeError("coordinate policy blocks must be non-empty and unique")
    return tuple(result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--base-overlay", type=Path, required=True)
    parser.add_argument("--output-overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--policy", type=_parse_policy, default=DEFAULT_POLICY)
    parser.add_argument("--fit-offset", type=int, default=460)
    parser.add_argument("--fit-samples", type=int, default=8)
    parser.add_argument("--validation-offset", type=int, default=468)
    parser.add_argument("--validation-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--maximum-sweeps", type=int, default=2)
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
    fit = set(range(args.fit_offset, args.fit_offset + args.fit_samples))
    validation = set(
        range(
            args.validation_offset,
            args.validation_offset + args.validation_samples,
        )
    )
    if (
        min(
            args.fit_samples,
            args.validation_samples,
            args.sequence_length - 1,
            args.maximum_sweeps,
        )
        <= 0
        or min(args.fit_offset, args.validation_offset) < 0
        or fit & validation
        or any(block >= block_count for block, _choice, _context in args.policy)
    ):
        raise ValueError("coordinate-refit protocol is invalid")


def _overlay_replacements(
    tensors: dict[str, torch.Tensor],
) -> dict[LayerId, torch.Tensor]:
    replacements = {}
    prefix = "model.layers."
    suffix = ".weight"
    for name, weight in tensors.items():
        if not name.startswith(prefix) or not name.endswith(suffix):
            raise ValueError("coordinate base overlay tensor name is invalid")
        logical = name.removeprefix(prefix).removesuffix(suffix)
        block_text, path = logical.split(".", maxsplit=1)
        if path not in MLP_PATHS:
            raise ValueError("coordinate base overlay contains a non-MLP tensor")
        replacements[LayerId(BlockId(int(block_text)), path)] = weight
    return replacements


def _decoder(model: torch.nn.Module) -> torch.nn.ModuleList:
    base = getattr(model, "model", None)
    decoder = getattr(base, "layers", None)
    if not isinstance(decoder, torch.nn.ModuleList):
        raise TypeError("model does not expose decoder blocks")
    return decoder


def _install_candidate_block(
    model: torch.nn.Module,
    candidate: SpliceReconstructionSet,
    block: int,
    *,
    device: str,
) -> None:
    decoder = _decoder(model)
    by_layer = {item.layer: item for item in candidate.layers}
    for path in MLP_PATHS:
        item = by_layer.get(LayerId(BlockId(block), path))
        if item is None:
            raise ValueError("coordinate candidate block is incomplete")
        current = decoder[block].get_submodule(path)
        if isinstance(current, torch.nn.Linear):
            with torch.no_grad():
                current.weight.copy_(
                    item.weight.to(device=device, dtype=current.weight.dtype)
                )
            continue
        _install_dense_linear(
            decoder[block],
            path,
            item.weight,
            device=device,
        )


def _selected_overlay(
    candidate: SpliceReconstructionSet,
    blocks: tuple[int, ...],
) -> SpliceReconstructionSet:
    layers = tuple(
        item
        for item in candidate.layers
        if item.layer.block.index in blocks and item.layer.path in MLP_PATHS
    )
    if len(layers) != 3 * len(blocks):
        raise ValueError("coordinate overlay export inventory is incomplete")
    return SpliceReconstructionSet(layers, (), ())


def run(args: argparse.Namespace) -> int:
    started = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[{time.perf_counter() - started:8.1f}s] {message}", flush=True)

    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    _validate_args(args, adapter.decoder_block_count_from_config(config))
    blocks = tuple(block for block, _choice, _context in args.policy)
    required_samples = max(
        args.fit_offset + args.fit_samples,
        args.validation_offset + args.validation_samples,
    )
    all_tokens, dataset_fingerprint, bos_token_id = _wikitext_tokens(
        args.snapshot,
        samples=required_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    fit_tokens = _select_token_window(
        all_tokens,
        offset=args.fit_offset,
        samples=args.fit_samples,
    )
    validation_tokens = _select_token_window(
        all_tokens,
        offset=args.validation_offset,
        samples=args.validation_samples,
    )
    overlay_tensors, overlay_manifest = _load_overlay(args.base_overlay)
    replacements = _overlay_replacements(overlay_tensors)
    if {layer.block.index for layer in replacements} != set(blocks):
        raise ValueError("coordinate policy must exactly cover the base overlay blocks")

    with acquire_device_lease(args.device):
        progress("loading frozen student")
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
        current = _replace_weights(collect_splice_reconstructions(loaded), replacements)
        progress("frozen student and incumbent overlay loaded")
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        loaded.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        progress("loading dense teacher")
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        progress("capturing dense-teacher fit context")
        teacher_fit = _capture_mlp_context(
            teacher,
            blocks,
            fit_tokens,
            device=args.device,
        )
        progress("capturing dense-teacher validation context")
        teacher_validation = _capture_mlp_context(
            teacher,
            blocks,
            validation_tokens,
            device=args.device,
        )
        teacher.cpu()
        teacher_decoder = _decoder(teacher)
        for block in blocks:
            for path in MLP_PATHS:
                teacher_decoder[block].get_submodule(path).to(args.device)
        progress("teacher offloaded with selected MLPs pinned")
        gc.collect()
        torch.cuda.empty_cache()

        student = loaded.model.to(args.device)
        student.eval()
        for block in blocks:
            _install_candidate_block(
                student,
                current,
                block,
                device=args.device,
            )
        progress("incumbent overlay installed in student")
        initial_validation = _evaluate_per_sequence(
            student,
            validation_tokens,
            args.device,
        )
        progress("initial validation evaluated")
        current_nll = float(
            cast(Any, initial_validation["mean_negative_log_likelihood"])
        )
        sweeps = []
        for sweep_index in range(args.maximum_sweeps):
            sweep_start_nll = current_nll
            coordinates = []
            for block, choice, context in args.policy:
                progress(
                    f"sweep {sweep_index + 1} fitting block {block} ({context})"
                )
                if context == "student_function":
                    student_fit = _capture_mlp_context(
                        student,
                        (block,),
                        fit_tokens,
                        device=args.device,
                    )
                    student_validation = _capture_mlp_context(
                        student,
                        (block,),
                        validation_tokens,
                        device=args.device,
                    )
                else:
                    student_fit = teacher_fit
                    student_validation = teacher_validation
                fitted, fit_metrics = _fit_context(
                    context,
                    teacher,
                    (block,),
                    fit_tokens,
                    validation_tokens,
                    current,
                    teacher_fit,
                    teacher_validation,
                    student_fit,
                    student_validation,
                    args,
                )
                progress(f"sweep {sweep_index + 1} fitted block {block}")
                candidate = _single_block_candidate(
                    current,
                    fitted,
                    block,
                    choice,
                )
                _install_candidate_block(
                    student,
                    candidate,
                    block,
                    device=args.device,
                )
                candidate_validation = _evaluate_per_sequence(
                    student,
                    validation_tokens,
                    args.device,
                )
                progress(f"sweep {sweep_index + 1} evaluated block {block}")
                candidate_nll = float(
                    cast(
                        Any,
                        candidate_validation["mean_negative_log_likelihood"],
                    )
                )
                accepted = candidate_nll < current_nll
                if accepted:
                    current = candidate
                    previous_nll = current_nll
                    current_nll = candidate_nll
                else:
                    previous_nll = current_nll
                    _install_candidate_block(
                        student,
                        current,
                        block,
                        device=args.device,
                    )
                coordinates.append(
                    {
                        "block": block,
                        "choice": choice,
                        "context": context,
                        "accepted": accepted,
                        "before_validation_nll": previous_nll,
                        "candidate_validation_nll": candidate_nll,
                        "delta": candidate_nll - previous_nll,
                        "fit_metrics": fit_metrics,
                    }
                )
                gc.collect()
                torch.cuda.empty_cache()
            sweep = {
                "index": sweep_index,
                "start_validation_nll": sweep_start_nll,
                "end_validation_nll": current_nll,
                "delta": current_nll - sweep_start_nll,
                "coordinates": coordinates,
            }
            sweeps.append(sweep)
            if current_nll >= sweep_start_nll:
                break
        final_validation = _evaluate_per_sequence(
            student,
            validation_tokens,
            args.device,
        )
        progress("final validation evaluated")
        del student, teacher, loaded

    overlay_export = _export_reconstruction_set(
        args.output_overlay,
        "composed-context-coordinate-sweep",
        _selected_overlay(current, blocks),
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only composed-context MLP coordinate sweep",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "base_overlay": {"directory": str(args.base_overlay), **overlay_manifest},
            "output_overlay": overlay_export,
            "policy": [
                {"block": block, "choice": choice, "context": context}
                for block, choice, context in args.policy
            ],
            "protocol": {
                "sequence_length": args.sequence_length,
                "fit_offset": args.fit_offset,
                "fit_samples": args.fit_samples,
                "validation_offset": args.validation_offset,
                "validation_samples": args.validation_samples,
                "maximum_sweeps": args.maximum_sweeps,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "fit_token_hash": _token_hash(fit_tokens),
                "validation_token_hash": _token_hash(validation_tokens),
            },
            "initial_validation": initial_validation,
            "sweeps": sweeps,
            "final_validation": final_validation,
            "mean_negative_log_likelihood_delta": (
                float(cast(Any, final_validation["mean_negative_log_likelihood"]))
                - float(cast(Any, initial_validation["mean_negative_log_likelihood"]))
            ),
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
