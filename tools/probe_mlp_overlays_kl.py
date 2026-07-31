"""Compare saved dense MLP overlays with paired held-out NLL and teacher KL."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_composed_context_coordinate_sweep import _overlay_replacements
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import (
    _dtype,
    _export_reconstruction_set,
    _paired_payload,
    _replace_weights,
    _select_token_window,
)
from probe_mlp_policy_frozen_transfer import _load_overlay
from probe_tuned_mlp_scale_refit import MODEL_SOURCE, PINNED_MODEL_REVISION
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_task_evaluation import load_pinned_dataset_split
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstruction,
    SpliceReconstructionSet,
    collect_splice_reconstructions,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.quality_evaluation import (
    WIKITEXT_CONFIG,
    WIKITEXT_DATASET,
    WIKITEXT_REVISION,
    _wikitext_tokens,
)


def _parse_overlay(value: str) -> tuple[str, tuple[Path, ...]]:
    name, path = value.split("=", maxsplit=1)
    paths = tuple(Path(item) for item in path.split(",") if item.strip())
    if (
        not name.strip()
        or not paths
        or any(character.isspace() for character in name)
    ):
        raise argparse.ArgumentTypeError("overlay must use a non-empty name=path form")
    return name, paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overlay",
        type=_parse_overlay,
        action="append",
        required=True,
    )
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--export-arm")
    parser.add_argument("--export-overlay", type=Path)
    parser.add_argument(
        "--wikitext-split",
        choices=("test", "validation"),
        default="test",
    )
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-global-tuning", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _split_tokens(
    snapshot: Path,
    *,
    split: str,
    samples: int,
    sequence_length: int,
    local_files_only: bool,
) -> tuple[torch.Tensor, str, int | None]:
    if split == "test":
        return _wikitext_tokens(
            snapshot,
            samples=samples,
            sequence_length=sequence_length,
            local_files_only=local_files_only,
        )
    dataset = load_pinned_dataset_split(
        WIKITEXT_DATASET,
        WIKITEXT_CONFIG,
        WIKITEXT_REVISION,
        split,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=local_files_only,
    )
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        raise ValueError("MLP overlay validation split requires a tokenizer BOS token")
    payload = sequence_length - 1
    required = samples * payload
    encoded = tokenizer(
        "\n\n".join(dataset["text"]),
        return_tensors="pt",
        truncation=True,
        max_length=required,
    ).input_ids
    if encoded.shape[1] < required:
        raise ValueError(
            f"WikiText {split} stream has {encoded.shape[1]} tokens; protocol "
            f"requires {required}"
        )
    rows = tuple(
        torch.cat(
            (
                torch.tensor([[bos_id]], dtype=encoded.dtype),
                encoded[:, index * payload : (index + 1) * payload],
            ),
            dim=1,
        )
        for index in range(samples)
    )
    return torch.cat(rows), str(getattr(dataset, "_fingerprint", "unknown")), int(bos_id)


def run(args: argparse.Namespace) -> int:
    if (
        args.samples <= 0
        or args.sequence_length <= 1
        or args.wikitext_offset < 0
        or len({name for name, _arm_paths in args.overlay}) != len(args.overlay)
        or (args.export_arm is None) != (args.export_overlay is None)
    ):
        raise ValueError("MLP overlay KL protocol is invalid")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, dataset_fingerprint, bos_token_id = _split_tokens(
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
    loaded_overlays = {}
    manifests = {}
    for name, paths in args.overlay:
        replacements = {}
        arm_manifests = []
        for path in paths:
            tensors, manifest = _load_overlay(path)
            arm_replacements = _overlay_replacements(tensors)
            overlap = replacements.keys() & arm_replacements.keys()
            if overlap:
                raise ValueError(
                    f"overlay arm {name} repeats layers: {sorted(map(str, overlap))}"
                )
            replacements.update(arm_replacements)
            arm_manifests.append({"directory": str(path), **manifest})
        loaded_overlays[name] = replacements
        manifests[name] = arm_manifests
    if args.export_arm is not None and args.export_arm not in loaded_overlays:
        raise ValueError("requested export arm is not an overlay arm")
    exported_overlay = None
    if args.export_arm is not None:
        assert args.export_overlay is not None
        exported_overlay = _export_reconstruction_set(
            args.export_overlay,
            args.export_arm,
            SpliceReconstructionSet(
                tuple(
                    SpliceReconstruction(layer, weight, None, 0.0)
                    for layer, weight in sorted(
                        loaded_overlays[args.export_arm].items(),
                        key=lambda item: (
                            item[0].block.index,
                            item[0].path,
                        ),
                    )
                ),
                (),
                (),
            ),
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
        baseline = collect_splice_reconstructions(loaded)
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        global_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
        candidates = {
            name: _replace_weights(baseline, replacements)
            for name, replacements in loaded_overlays.items()
        }
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
        results = {}
        teacher_nll = 0.0
        teacher_cache: tuple[torch.Tensor, ...] = ()
        for name, reconstructions in {"baseline": baseline, **candidates}.items():
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
                evaluator.install_teacher_cache(teacher_nll, teacher_cache)
            results[name] = evaluator("full")
            del evaluator
            gc.collect()
            torch.cuda.empty_cache()
        del teacher

    baseline_result = results["baseline"]
    paired_baseline_nll = {
        name: _paired_metric_payload(
            baseline_result,
            result,
            "negative_log_likelihood",
        )
        for name, result in results.items()
        if name != "baseline"
    }
    paired_baseline_kl = {
        name: _paired_payload(baseline_result, result, seed=0)
        for name, result in results.items()
        if name != "baseline"
    }
    names = tuple(name for name, _arm_paths in args.overlay)
    paired_arms = {}
    for before_name, after_name in zip(names, names[1:], strict=False):
        key = f"{after_name}_minus_{before_name}"
        paired_arms[key] = {
            "nll": _paired_metric_payload(
                results[before_name],
                results[after_name],
                "negative_log_likelihood",
            ),
            "kl": _paired_payload(
                results[before_name],
                results[after_name],
                seed=0,
            ),
        }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only dense MLP overlay KL comparison",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "overlays": manifests,
            "exported_overlay": exported_overlay,
            "protocol": {
                "wikitext_offset": args.wikitext_offset,
                "wikitext_split": args.wikitext_split,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_candidate_minus_baseline_nll": paired_baseline_nll,
            "paired_candidate_minus_baseline_kl": paired_baseline_kl,
            "paired_adjacent_overlays": paired_arms,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
