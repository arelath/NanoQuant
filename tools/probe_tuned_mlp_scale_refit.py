"""Screen zero-bit MLP scale refits on a retained tuned frozen run."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_corrected_codebook_splice import (
    _downstream_input_refit_sets,
    _downstream_policy_sets,
    _downstream_refit_sets,
    _dtype,
    _export_reconstruction_set,
    _module_at_path,
    _operator_refit_sets,
    _paired_payload,
    _parse_block_policy,
    _select_token_window,
)

from nanoquant.application.layers import FrozenReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.models import BlockId, LayerId
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
    collect_splice_reconstructions,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import _wikitext_tokens

MODEL_SOURCE = "google/gemma-3-1b-it"
PINNED_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
MLP_PATHS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _collect_selected_mlp_reconstructions(
    model: torch.nn.Module,
    blocks: tuple[int, ...],
) -> SpliceReconstructionSet:
    base = getattr(model, "model", None)
    decoder = getattr(base, "layers", None)
    if not isinstance(decoder, torch.nn.ModuleList):
        raise TypeError("frozen model does not expose decoder blocks")
    layers = []
    for block_index in blocks:
        for path in MLP_PATHS:
            module = _module_at_path(decoder[block_index], path)
            if not isinstance(module, FrozenReferenceLinear):
                raise TypeError("selected frozen MLP layer is not reconstructable")
            layer = LayerId(BlockId(block_index), path)
            layers.append(
                SpliceReconstruction(
                    layer,
                    module.dense_weight().detach().cpu().clone(),
                    None if module.bias is None else module.bias.detach().cpu().clone(),
                    0.0,
                )
            )
    return SpliceReconstructionSet(tuple(layers), (), ())


def _parse_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)) or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("blocks must be unique non-negative integers")
    return result


def _parse_arms(value: str) -> tuple[str, ...]:
    choices = {"baseline", "operator", "output", "input", "joint", "policy"}
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if "baseline" not in result or len(result) != len(set(result)) or any(item not in choices for item in result):
        raise argparse.ArgumentTypeError("arms must be unique known choices including baseline")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export-reconstruction-set", type=Path)
    parser.add_argument("--export-arm")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--blocks", type=_parse_ints, default=tuple(range(26)))
    parser.add_argument("--policy", type=_parse_block_policy)
    parser.add_argument(
        "--arms",
        type=_parse_arms,
        default=("baseline", "operator", "output", "input", "joint"),
    )
    parser.add_argument("--fit-offset", type=int, default=48)
    parser.add_argument("--fit-samples", type=int, default=4)
    parser.add_argument("--validation-offset", type=int, default=52)
    parser.add_argument("--validation-samples", type=int, default=4)
    parser.add_argument("--evaluation-offset", type=int, default=272)
    parser.add_argument("--evaluation-samples", type=int, default=12)
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


def _validate_windows(args: argparse.Namespace) -> None:
    inventories = (
        set(range(args.fit_offset, args.fit_offset + args.fit_samples)),
        set(
            range(
                args.validation_offset,
                args.validation_offset + args.validation_samples,
            )
        ),
        set(
            range(
                args.evaluation_offset,
                args.evaluation_offset + args.evaluation_samples,
            )
        ),
    )
    if (
        min(
            args.fit_samples,
            args.validation_samples,
            args.evaluation_samples,
            args.sequence_length - 1,
        )
        <= 0
        or inventories[0] & inventories[1]
        or inventories[0] & inventories[2]
        or inventories[1] & inventories[2]
    ):
        raise ValueError("tuned MLP refit windows must be positive and disjoint")


def run(args: argparse.Namespace) -> int:
    _validate_windows(args)
    if (args.export_reconstruction_set is None) != (args.export_arm is None):
        raise ValueError("reconstruction export requires destination and arm")
    if args.export_only and args.export_reconstruction_set is None:
        raise ValueError("export-only requires a reconstruction export")
    if args.policy is not None and {block for block, _choice in args.policy} != set(args.blocks):
        raise ValueError("tuned MLP policy must choose every requested block")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    if any(block >= adapter.decoder_block_count_from_config(config) for block in args.blocks):
        raise ValueError("requested block is outside the model")
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
            device="cpu",
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=args.use_global_tuning,
        )
        baseline = (
            _collect_selected_mlp_reconstructions(loaded.model, args.blocks)
            if args.export_only
            else collect_splice_reconstructions(loaded)
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        del loaded
        gc.collect()
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        reconstruction_sets = {
            "free_words": baseline,
            "corrected_codebook": baseline,
        }
        reconstruction_sets, operator_metrics = _operator_refit_sets(
            teacher,
            args.blocks,
            fit_tokens,
            validation_tokens,
            reconstruction_sets,
            device=args.device,
            gate_grid_points=args.gate_grid_points,
            minimum_gate_multiplier=args.minimum_gate_multiplier,
            maximum_gate_multiplier=args.maximum_gate_multiplier,
            minimum_up_multiplier=args.minimum_up_multiplier,
            maximum_up_multiplier=args.maximum_up_multiplier,
        )
        reconstruction_sets, output_metrics = _downstream_refit_sets(
            teacher,
            args.blocks,
            fit_tokens,
            validation_tokens,
            reconstruction_sets,
            device=args.device,
            minimum_multiplier=args.minimum_down_multiplier,
            maximum_multiplier=args.maximum_down_multiplier,
        )
        reconstruction_sets, input_metrics = _downstream_input_refit_sets(
            teacher,
            args.blocks,
            fit_tokens,
            validation_tokens,
            reconstruction_sets,
            device=args.device,
            minimum_multiplier=args.minimum_down_multiplier,
            maximum_multiplier=args.maximum_down_multiplier,
            iterations=args.input_iterations,
            learning_rate=args.input_learning_rate,
        )
        if args.policy is not None:
            reconstruction_sets = _downstream_policy_sets(
                reconstruction_sets,
                args.policy,
            )
        available_arms = {
            "baseline": "free_words",
            "operator": "free_words_operator_refit",
            "output": "free_words_operator_downstream_refit",
            "input": "free_words_operator_downstream_input_refit",
            "joint": "free_words_operator_downstream_joint_refit",
            "policy": "free_words_operator_policy_refit",
        }
        arms = {name: available_arms[name] for name in args.arms}
        reconstruction_export = None
        if args.export_reconstruction_set is not None:
            if args.export_arm not in available_arms:
                raise ValueError(f"reconstruction export arm is unavailable: {args.export_arm}")
            selected = tuple(
                item
                for item in reconstruction_sets[available_arms[args.export_arm]].layers
                if item.layer.block.index in args.blocks
                and item.layer.path in {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}
            )
            reconstruction_export = _export_reconstruction_set(
                args.export_reconstruction_set,
                args.export_arm,
                SpliceReconstructionSet(selected, (), ()),
            )
        results = {}
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        for name, source in () if args.export_only else arms.items():
            evaluator = DenseKlSpliceEvaluator(
                teacher,
                reconstruction_sets[source],
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
            results[name] = evaluator("full")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del teacher
    comparisons = (
        {}
        if args.export_only
        else {
            name: _paired_payload(results["baseline"], result, seed=0)
            for name, result in results.items()
            if name != "baseline"
        }
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only tuned frozen MLP scale-refit screen",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "use_global_tuning": args.use_global_tuning,
            "reconstruction_export": reconstruction_export,
            "blocks": list(args.blocks),
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
            "kl": {name: to_dict(result) for name, result in results.items()},
            "paired_refit_minus_baseline": comparisons,
            "operator_refit": operator_metrics["free_words"],
            "downstream_output_refit": output_metrics["free_words_operator_refit"],
            "downstream_input_refit": input_metrics["free_words_operator_refit"],
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
