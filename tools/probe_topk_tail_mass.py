"""Measure the probability-mass blind spot of conditional top-k distillation."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_corrected_codebook_splice import _dtype
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from transformers.modeling_utils import PreTrainedModel

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash


@dataclass(frozen=True, slots=True)
class TailMassSums:
    negative_log_likelihood: float
    full_kl: float
    conditional_topk_kl: float
    topk_plus_tail_kl: float
    teacher_topk_mass: float
    student_teacher_topk_mass: float
    tail_mass_absolute_error: float
    token_count: int

    def __add__(self, other: TailMassSums) -> TailMassSums:
        return TailMassSums(
            self.negative_log_likelihood + other.negative_log_likelihood,
            self.full_kl + other.full_kl,
            self.conditional_topk_kl + other.conditional_topk_kl,
            self.topk_plus_tail_kl + other.topk_plus_tail_kl,
            self.teacher_topk_mass + other.teacher_topk_mass,
            self.student_teacher_topk_mass + other.student_teacher_topk_mass,
            self.tail_mass_absolute_error + other.tail_mass_absolute_error,
            self.token_count + other.token_count,
        )


def _parse_arm(value: str) -> tuple[str, str, Path | None]:
    name, separator, specification = value.partition("=")
    state, overlay_separator, overlay = specification.partition(":")
    if (
        not separator
        or not name.strip()
        or state not in {"prekd", "postkd", "tuning"}
        or (state == "tuning" and (not overlay_separator or not overlay.strip()))
        or (overlay_separator and (state == "prekd" or not overlay.strip()))
    ):
        raise argparse.ArgumentTypeError(
            "arm must use name=prekd, name=postkd, name=postkd:component-overlay, "
            "or name=tuning:artifact-reference-json"
        )
    return name.strip(), state, None if not overlay_separator else Path(overlay)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _tail_mass_sums(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    top_k: int,
) -> TailMassSums:
    if (
        teacher_logits.shape != student_logits.shape
        or teacher_logits.ndim != 3
        or labels.shape != teacher_logits.shape[:2]
        or top_k <= 0
        or top_k >= teacher_logits.shape[-1]
    ):
        raise ValueError("top-k tail-mass metric geometry is invalid")
    teacher_log_probabilities = torch.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probabilities = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_probabilities = teacher_log_probabilities.exp()
    indices = torch.topk(teacher_logits, top_k, dim=-1).indices
    teacher_top_probabilities = teacher_probabilities.gather(-1, indices)
    student_top_log_probabilities = student_log_probabilities.gather(-1, indices)
    student_top_probabilities = student_top_log_probabilities.exp()
    teacher_mass = teacher_top_probabilities.sum(dim=-1)
    student_mass = student_top_probabilities.sum(dim=-1)
    teacher_tail = (1 - teacher_mass).clamp_min(1e-12)
    student_tail = (1 - student_mass).clamp_min(1e-12)
    conditional_teacher = teacher_top_probabilities / teacher_mass.unsqueeze(-1)
    conditional_student_log = torch.log_softmax(student_logits.float().gather(-1, indices), dim=-1)
    conditional_topk_kl = (
        conditional_teacher
        * (conditional_teacher.clamp_min(1e-30).log() - conditional_student_log)
    ).sum()
    topk_plus_tail_kl = (
        teacher_top_probabilities
        * (teacher_log_probabilities.gather(-1, indices) - student_top_log_probabilities)
    ).sum() + (teacher_tail * (teacher_tail.log() - student_tail.log())).sum()
    return TailMassSums(
        negative_log_likelihood=float(
            -student_log_probabilities.gather(-1, labels.unsqueeze(-1)).sum()
        ),
        full_kl=float(
            (teacher_probabilities * (teacher_log_probabilities - student_log_probabilities)).sum()
        ),
        conditional_topk_kl=float(conditional_topk_kl),
        topk_plus_tail_kl=float(topk_plus_tail_kl),
        teacher_topk_mass=float(teacher_mass.sum()),
        student_teacher_topk_mass=float(student_mass.sum()),
        tail_mass_absolute_error=float((student_tail - teacher_tail).abs().sum()),
        token_count=labels.numel(),
    )


def _means(sums: TailMassSums) -> dict[str, float | int]:
    if sums.token_count <= 0:
        raise ValueError("top-k tail-mass result contains no tokens")
    denominator = float(sums.token_count)
    return {
        "negative_log_likelihood": sums.negative_log_likelihood / denominator,
        "perplexity": math.exp(sums.negative_log_likelihood / denominator),
        "full_kl": sums.full_kl / denominator,
        "conditional_topk_kl": sums.conditional_topk_kl / denominator,
        "topk_plus_tail_kl": sums.topk_plus_tail_kl / denominator,
        "teacher_topk_mass": sums.teacher_topk_mass / denominator,
        "student_teacher_topk_mass": sums.student_teacher_topk_mass / denominator,
        "tail_mass_absolute_error": sums.tail_mass_absolute_error / denominator,
        "token_count": sums.token_count,
    }


def _evaluate(
    teacher: PreTrainedModel,
    student: PreTrainedModel,
    tokens: torch.Tensor,
    *,
    top_k: int,
    device: str,
) -> dict[str, float | int]:
    total = TailMassSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for row in tokens:
            batch = row.unsqueeze(0).to(device)
            teacher_logits = cast(Any, teacher)(input_ids=batch, use_cache=False).logits[:, :-1]
            student_logits = cast(Any, student)(input_ids=batch, use_cache=False).logits[:, :-1]
            total += _tail_mass_sums(
                teacher_logits,
                student_logits,
                batch[:, 1:],
                top_k=top_k,
            )
            del batch, teacher_logits, student_logits
    return _means(total)


def run(args: argparse.Namespace) -> int:
    if (
        len(args.arm) < 2
        or len({name for name, _state, _overlay in args.arm}) != len(args.arm)
        or min(args.samples, args.sequence_length - 1, args.top_k) <= 0
        or args.wikitext_offset < 0
    ):
        raise ValueError("top-k tail-mass probe protocol is invalid")
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, dataset_fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = all_tokens[args.wikitext_offset : args.wikitext_offset + args.samples]
    results: dict[str, dict[str, float | int]] = {}
    manifests: dict[str, dict[str, object]] = {}
    frozen_identity = None
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        for name, state, overlay in args.arm:
            global_tuning_override = None
            component_overlay = overlay
            if state == "tuning":
                assert overlay is not None
                global_tuning_override = from_dict(
                    ArtifactRef,
                    json.loads(overlay.read_text(encoding="utf-8")),
                    path=f"arm[{name}].global_tuning",
                )
                component_overlay = None
            loaded = load_frozen_run(
                args.run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=args.device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=state != "prekd",
                global_tuning_override=global_tuning_override,
                component_overlay=component_overlay,
            )
            observed_identity = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            if frozen_identity is not None and frozen_identity != observed_identity:
                raise ValueError("top-k tail-mass arms have different frozen identities")
            frozen_identity = observed_identity
            manifests[name] = {
                "state": state,
                "global_tuning": None if loaded.global_tuning is None else to_dict(loaded.global_tuning),
                "global_tuning_pointer": (
                    str(overlay.resolve()) if state == "tuning" and overlay is not None else None
                ),
                "component_overlay": (
                    None if component_overlay is None else str(component_overlay.resolve())
                ),
            }
            results[name] = _evaluate(
                cast(PreTrainedModel, teacher),
                cast(PreTrainedModel, loaded.model),
                tokens,
                top_k=args.top_k,
                device=args.device,
            )
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher
    baseline_name = args.arm[0][0]
    baseline = results[baseline_name]
    comparisons = {
        f"{name}_minus_{baseline_name}": {
            key: float(value) - float(baseline[key])
            for key, value in result.items()
            if key != "token_count"
        }
        for name, result in results.items()
        if name != baseline_name
    }
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only conditional top-k probability-mass blind-spot audit",
            "run_output": str(args.run_output),
            "frozen_identity": frozen_identity,
            "arms": manifests,
            "protocol": {
                "wikitext_split": args.wikitext_split,
                "wikitext_offset": args.wikitext_offset,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "top_k": args.top_k,
                "dataset_fingerprint": dataset_fingerprint,
                "bos_token_id": bos_token_id,
                "token_hash": _token_hash(tokens),
            },
            "results": results,
            "comparisons": comparisons,
        },
    )
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
