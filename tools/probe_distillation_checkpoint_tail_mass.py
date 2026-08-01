"""Screen durable KD checkpoints on held-out NLL, KL, and teacher-top-k mass."""

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
from probe_topk_tail_mass import TailMassSums, _means, _tail_mass_sums

from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.global_distillation import _selected_parameters, _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_checkpoint import (
    DistillationCheckpointIdentity,
    load_distillation_checkpoint,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.tensor_store import LocalTensorStore


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    epoch: int
    steps: int
    reference: ArtifactRef
    identity: DistillationCheckpointIdentity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument(
        "--frozen-run-output",
        type=Path,
        help="Frozen resident run to load when checkpoints live in a separate analysis run.",
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, action="append", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--wikitext-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--wikitext-offset", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument(
        "--student-logit-scale",
        type=float,
        action="append",
        help="Repeat to screen foldable student-logit calibration values (default: 1.0).",
    )
    parser.add_argument(
        "--fold-final-norm-scale",
        type=float,
        action="append",
        help="Repeat to screen the same scale folded into Gemma's final RMSNorm.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def discover_checkpoints(
    run_output: Path,
    epochs: set[int],
) -> tuple[CheckpointCandidate, ...]:
    active = from_dict(
        ArtifactRef,
        json.loads((run_output / "global-distillation-training.json").read_text(encoding="utf-8")),
        path="active_distillation_checkpoint",
    )
    artifacts = LocalArtifactStore(run_output / "artifacts")
    active_manifest = json.loads(
        (artifacts.path_for(active.artifact_id) / "checkpoint.json").read_text(encoding="utf-8")
    )
    protocol_hash = str(active_manifest["identity"]["protocol_hash"])
    candidates = []
    for manifest_path in sorted((run_output / "artifacts").glob("*/*/checkpoint.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        epoch = int(manifest["completed_epochs"])
        if epoch not in epochs or str(manifest["identity"]["protocol_hash"]) != protocol_hash:
            continue
        identity = from_dict(
            DistillationCheckpointIdentity,
            manifest["identity"],
            path=f"checkpoint[{epoch}].identity",
        )
        candidates.append(
            CheckpointCandidate(
                epoch,
                int(manifest["steps_completed"]),
                ArtifactRef("distillation-checkpoint", manifest_path.parent.name, 1),
                identity,
            )
        )
    by_epoch = {candidate.epoch: candidate for candidate in candidates}
    if set(by_epoch) != epochs or len(by_epoch) != len(candidates):
        raise ValueError("requested KD checkpoint inventory is missing or ambiguous")
    return tuple(by_epoch[epoch] for epoch in sorted(by_epoch))


def _zero_sums() -> TailMassSums:
    return TailMassSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)


def _scaled_result_name(state: str, scale: float, *, multiple: bool) -> str:
    return f"{state}@scale={scale:g}" if multiple else state


def _apply_gemma_final_norm_scale(
    parameter: torch.Tensor,
    source: torch.Tensor,
    scale: float,
) -> None:
    if parameter.shape != source.shape:
        raise ValueError("Gemma final RMSNorm source shape differs")
    with torch.no_grad():
        parameter.copy_(((1.0 + source.float()) * scale - 1.0).to(parameter))


def run(args: argparse.Namespace) -> int:
    requested_epochs = set(args.epoch)
    if args.student_logit_scale and args.fold_final_norm_scale:
        raise ValueError("choose post-logit scaling or folded final-RMSNorm scaling, not both")
    fold_final_norm = bool(args.fold_final_norm_scale)
    logit_scales = tuple(
        args.fold_final_norm_scale or args.student_logit_scale or (1.0,)
    )
    if (
        not requested_epochs
        or min(requested_epochs) <= 0
        or min(args.samples, args.sequence_length - 1, args.top_k) <= 0
        or args.wikitext_offset < 0
        or len(set(logit_scales)) != len(logit_scales)
        or any(not math.isfinite(scale) or scale <= 0.0 for scale in logit_scales)
    ):
        raise ValueError("checkpoint tail-mass protocol is invalid")
    candidates = discover_checkpoints(args.run_output, requested_epochs)
    frozen_run_output = args.frozen_run_output or args.run_output
    model_config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(model_config)
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.wikitext_split,
        samples=args.wikitext_offset + args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    tokens = all_tokens[args.wikitext_offset : args.wikitext_offset + args.samples]
    artifacts = LocalArtifactStore(args.run_output / "artifacts")
    frozen_artifacts = LocalArtifactStore(frozen_run_output / "artifacts")
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            frozen_run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device="cpu",
            verify_hashes=False,
            backend="factorized",
            use_global_tuning=False,
        )
        trainable = _thaw_frozen_layers(loaded, LocalTensorStore(frozen_artifacts))
        selected_ids, _auxiliary = _selected_parameters(loaded.model, trainable)
        selected = {
            name: parameter
            for name, parameter in loaded.model.named_parameters()
            if id(parameter) in selected_ids
        }
        states: dict[str, dict[str, torch.Tensor]] = {
            "prekd": {name: parameter.detach().cpu().clone() for name, parameter in selected.items()}
        }
        checkpoint_receipts = []
        for candidate in candidates:
            checkpoint = load_distillation_checkpoint(candidate.reference, candidate.identity, artifacts)
            values = dict(checkpoint.state.parameter_values)
            if set(values) != set(selected):
                raise ValueError(f"checkpoint epoch {candidate.epoch} differs from the KD selector")
            name = f"epoch_{candidate.epoch}"
            states[name] = values
            checkpoint_receipts.append(
                {
                    "epoch": candidate.epoch,
                    "steps": candidate.steps,
                    "reference": to_dict(candidate.reference),
                    "epoch_loss": checkpoint.state.epoch_losses[-1],
                }
            )
            del checkpoint
        student = loaded.model.to(args.device)
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(model_config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        student.eval()
        multiple_scales = len(logit_scales) > 1 or logit_scales != (1.0,)
        totals = {
            _scaled_result_name(name, scale, multiple=multiple_scales): _zero_sums()
            for name in states
            for scale in logit_scales
        }
        with torch.no_grad():
            for row_index, row in enumerate(tokens):
                batch = row.unsqueeze(0).to(args.device)
                teacher_logits = cast(Any, teacher)(input_ids=batch, use_cache=False).logits[:, :-1]
                labels = batch[:, 1:]
                for name, values in states.items():
                    for parameter_name, parameter in selected.items():
                        parameter.copy_(
                            values[parameter_name].to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )
                    if fold_final_norm:
                        final_norm = selected.get("model.norm.weight")
                        final_norm_source = values.get("model.norm.weight")
                        if final_norm is None or final_norm_source is None:
                            raise ValueError("KD selector does not contain Gemma's final RMSNorm")
                        for scale in logit_scales:
                            _apply_gemma_final_norm_scale(final_norm, final_norm_source, scale)
                            student_logits = cast(Any, student)(
                                input_ids=batch,
                                use_cache=False,
                            ).logits[:, :-1]
                            result_name = _scaled_result_name(
                                name,
                                scale,
                                multiple=multiple_scales,
                            )
                            totals[result_name] += _tail_mass_sums(
                                teacher_logits,
                                student_logits,
                                labels,
                                top_k=args.top_k,
                            )
                            del student_logits
                    else:
                        student_logits = cast(Any, student)(
                            input_ids=batch,
                            use_cache=False,
                        ).logits[:, :-1]
                        for scale in logit_scales:
                            result_name = _scaled_result_name(
                                name,
                                scale,
                                multiple=multiple_scales,
                            )
                            totals[result_name] += _tail_mass_sums(
                                teacher_logits,
                                student_logits if scale == 1.0 else student_logits.float() * scale,
                                labels,
                                top_k=args.top_k,
                            )
                        del student_logits
                del batch, teacher_logits, labels
                print(f"checkpoint tail-mass row {row_index + 1}/{len(tokens)}", flush=True)
        report = {
            "schema_version": 1,
            "role": "durable production KD checkpoint selection on held-out tail-mass metrics",
            "protocol": {
                "run_output": str(args.run_output.resolve()),
                "frozen_run_output": str(frozen_run_output.resolve()),
                "model_revision": args.model_revision,
                "wikitext_split": args.wikitext_split,
                "wikitext_offset": args.wikitext_offset,
                "samples": args.samples,
                "sequence_length": args.sequence_length,
                "top_k": args.top_k,
                "student_logit_scales": logit_scales,
                "student_logit_scale_mode": (
                    "folded-final-rmsnorm" if fold_final_norm else "post-logits"
                ),
                "dataset_fingerprint": fingerprint,
                "bos_token_id": bos_token_id,
            },
            "checkpoints": checkpoint_receipts,
            "results": {name: _means(total) for name, total in totals.items()},
        }
        atomic_write_json(args.output, report)
        teacher.cpu()
        student.cpu()
        del teacher, student, loaded, trainable
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
