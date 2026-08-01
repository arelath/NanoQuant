"""Compare factorized component overlays with paired held-out NLL and teacher KL."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload, _select_token_window
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from torch import nn

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlSequenceResult,
    causal_kl_nll_per_sequence_from_logits,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_model_protocol import HuggingFaceModel
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash


def _parse_arm(value: str) -> tuple[str, Path | None]:
    name, separator, path = value.partition("=")
    if not name.strip() or (separator and not path.strip()):
        raise argparse.ArgumentTypeError("component arm must use name or name=path")
    return name.strip(), None if not separator else Path(path.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _arm_result(name: str, sequences: tuple[KlSequenceResult, ...]) -> KlBudgetArmResult:
    if not sequences:
        raise ValueError("factorized component arm produced no sequence metrics")
    tokens = math.fsum(item.token_count for item in sequences)
    return KlBudgetArmResult(
        name,
        math.fsum(item.negative_log_likelihood * item.token_count for item in sequences) / tokens,
        math.fsum(item.kl_nats_per_token * item.token_count for item in sequences) / tokens,
        int(tokens),
        sequences=sequences,
    )


@torch.no_grad()
def _evaluate_arm(
    name: str,
    teacher: nn.Module,
    student: nn.Module,
    tokens: torch.Tensor,
    *,
    device: str,
    token_chunk_size: int,
) -> KlBudgetArmResult:
    teacher.eval()
    student.eval()
    sequences: list[KlSequenceResult] = []
    for index in range(tokens.shape[0]):
        batch = tokens[index : index + 1].to(device)
        teacher_logits = cast(HuggingFaceModel, teacher)(input_ids=batch, use_cache=False).logits
        student_logits = cast(HuggingFaceModel, student)(input_ids=batch, use_cache=False).logits
        sequences.extend(
            causal_kl_nll_per_sequence_from_logits(
                teacher_logits,
                student_logits,
                batch,
                token_chunk_size=token_chunk_size,
            )
        )
        del teacher_logits, student_logits, batch
    return _arm_result(name, tuple(sequences))


def run(args: argparse.Namespace) -> int:
    if (
        args.samples <= 0
        or args.sequence_length <= 1
        or args.wikitext_offset < 0
        or args.token_chunk_size <= 0
        or len(args.arm) < 2
        or len({name for name, _path in args.arm}) != len(args.arm)
    ):
        raise ValueError("factorized component KL protocol is invalid")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = _select_token_window(all_tokens, offset=args.wikitext_offset, samples=args.samples)
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        results: dict[str, KlBudgetArmResult] = {}
        manifests = {}
        identity = None
        global_tuning = None
        for name, path in args.arm:
            loaded = load_frozen_run(
                args.run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=args.device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=True,
                component_overlay=path,
            )
            observed_identity = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            if identity is not None and observed_identity != identity:
                raise ValueError("factorized component arms have different frozen identities")
            identity = observed_identity
            observed_tuning = None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
            if global_tuning is not None and observed_tuning != global_tuning:
                raise ValueError("factorized component arms have different global tuning identities")
            global_tuning = observed_tuning
            manifests[name] = (
                None
                if path is None
                else {
                    "directory": str(path),
                    **json.loads((path / "manifest.json").read_text(encoding="utf-8")),
                }
            )
            results[name] = _evaluate_arm(
                name,
                teacher,
                loaded.model,
                tokens,
                device=args.device,
                token_chunk_size=args.token_chunk_size,
            )
            del loaded
            gc.collect()
            torch.cuda.empty_cache()
        del teacher

    names = tuple(name for name, _path in args.arm)
    paired = {}
    for before, after in zip(names, names[1:], strict=False):
        paired[f"{after}_minus_{before}"] = {
            "nll": _paired_metric_payload(results[before], results[after], "negative_log_likelihood"),
            "kl": _paired_payload(results[before], results[after], seed=0),
        }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only factorized component overlay KL comparison",
            "model_revision": args.model_revision,
            "run_output": str(args.run_output),
            "frozen_identity": identity,
            "global_tuning": global_tuning,
            "component_overlays": manifests,
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
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_adjacent_arms": paired,
        },
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
