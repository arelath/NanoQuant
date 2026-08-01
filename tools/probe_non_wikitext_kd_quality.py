"""Compare retained KD checkpoints on a pinned non-WikiText C4 slice."""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from datasets import Dataset, load_dataset  # type: ignore[import-untyped]
from probe_composed_context_mlp_refit import _paired_metric_payload
from probe_corrected_codebook_splice import _paired_payload
from probe_factorized_component_overlays_kl import _dtype
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.kl_budget import (
    KlBudgetArmResult,
    KlSequenceResult,
    causal_kl_nll_per_sequence_from_logits,
)
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_model_protocol import HuggingFaceModel
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.kl_budget_workflow import _token_hash

C4_DATASET = "allenai/c4"
C4_REVISION = "f998d2cd8b92435980789e3ecb2f89b4c68bfe1e"
C4_VALIDATION_FILE = "en/c4-validation.00000-of-00008.json.gz"
C4_VALIDATION_FILE_SHA256 = "bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0"


def _parse_arm(value: str) -> tuple[str, str, Path, Path | None]:
    name, separator, specification = value.partition("=")
    parts = specification.split(";")
    mode = parts[0] if parts else ""
    if (
        not separator
        or not name.strip()
        or mode not in {"prekd", "postkd", "tuning"}
        or (mode in {"prekd", "postkd"} and len(parts) != 2)
        or (mode == "tuning" and len(parts) != 3)
        or not parts[1].strip()
        or (mode == "tuning" and not parts[2].strip())
    ):
        raise argparse.ArgumentTypeError(
            "arm must use name=prekd;run-output, name=postkd;run-output, or "
            "name=tuning;run-output;artifact-reference-json"
        )
    return (
        name.strip(),
        mode,
        Path(parts[1].strip()),
        None if mode != "tuning" else Path(parts[2].strip()),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", type=_parse_arm, action="append", required=True)
    parser.add_argument("--primary-baseline", required=True)
    parser.add_argument("--primary-candidate", required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--c4-revision", default=C4_REVISION)
    parser.add_argument("--c4-file", default=C4_VALIDATION_FILE)
    parser.add_argument("--c4-documents", type=int, default=1_100)
    parser.add_argument("--offset", type=int, default=104)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _contiguous_token_windows(
    texts: Sequence[str],
    tokenizer: Any,
    *,
    offset: int,
    samples: int,
    sequence_length: int,
) -> tuple[torch.Tensor, int | None]:
    if offset < 0 or samples <= 0 or sequence_length <= 1 or not texts:
        raise ValueError("C4 token window protocol is invalid")
    encoded = tokenizer(
        " ".join(texts),
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("C4 tokenizer output must contain one contiguous token stream")
    required = (offset + samples) * sequence_length
    if input_ids.shape[1] < required:
        raise ValueError("C4 token stream is shorter than the requested held-out window")
    windows = input_ids[:, :required].reshape(offset + samples, sequence_length)
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    return windows[offset : offset + samples].long().contiguous(), bos_token_id


def _load_c4_tokens(
    snapshot: Path,
    *,
    revision: str,
    data_file: str,
    documents: int,
    offset: int,
    samples: int,
    sequence_length: int,
    local_files_only: bool,
) -> tuple[torch.Tensor, str, int | None]:
    local_data_file = Path(data_file)
    if local_data_file.is_file() and local_data_file.suffix == ".arrow":
        dataset = Dataset.from_file(str(local_data_file.resolve()))
    elif local_data_file.is_file():
        dataset = load_dataset(
            "json",
            data_files={"validation": str(local_data_file.resolve())},
            split="validation",
        )
    else:
        dataset = load_dataset(
            C4_DATASET,
            data_files={"validation": data_file},
            split="validation",
            revision=revision,
        )
    if documents <= 0 or len(dataset) < documents:
        raise ValueError("C4 document inventory is shorter than the pinned protocol")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=local_files_only,
    )
    texts = dataset[:documents]["text"]
    if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
        raise ValueError("C4 validation shard has an invalid text column")
    tokens, bos_token_id = _contiguous_token_windows(
        texts,
        tokenizer,
        offset=offset,
        samples=samples,
        sequence_length=sequence_length,
    )
    return tokens, str(dataset._fingerprint), bos_token_id


def _arm_result(name: str, sequences: tuple[KlSequenceResult, ...]) -> KlBudgetArmResult:
    if not sequences:
        raise ValueError("non-WikiText arm produced no sequence metrics")
    tokens = math.fsum(item.token_count for item in sequences)
    return KlBudgetArmResult(
        name,
        math.fsum(item.negative_log_likelihood * item.token_count for item in sequences)
        / tokens,
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
    completed: tuple[KlSequenceResult, ...] = (),
    progress: Callable[[tuple[KlSequenceResult, ...]], None] | None = None,
) -> KlBudgetArmResult:
    if len(completed) > tokens.shape[0]:
        raise ValueError("non-WikiText checkpoint exceeds the token inventory")
    teacher.eval()
    student.eval()
    sequences = list(completed)
    for index in range(len(completed), tokens.shape[0]):
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
        if progress is not None:
            progress(tuple(sequences))
        print(f"{name}: {index + 1}/{tokens.shape[0]} sequences", flush=True)
        del batch, teacher_logits, student_logits
    return _arm_result(name, tuple(sequences))


def _comparison(
    baseline: KlBudgetArmResult,
    candidate: KlBudgetArmResult,
) -> dict[str, object]:
    nll = _paired_metric_payload(baseline, candidate, "negative_log_likelihood")
    kl = _paired_payload(baseline, candidate, seed=0)
    return {
        "nll": nll,
        "kl": kl,
        "passes": bool(nll["improved_with_confidence"])
        and float(kl["upper_delta"]) <= 0.0,
    }


def run(args: argparse.Namespace) -> int:
    arms = tuple(args.arm)
    names = tuple(name for name, _mode, _path, _pointer in arms)
    if (
        len(arms) < 2
        or len(set(names)) != len(names)
        or args.primary_baseline not in names
        or args.primary_candidate not in names
        or args.primary_baseline == args.primary_candidate
        or min(args.c4_documents, args.samples, args.sequence_length - 1, args.token_chunk_size)
        <= 0
        or args.offset < 0
    ):
        raise ValueError("non-WikiText KD protocol is invalid")
    tokens, fingerprint, bos_token_id = _load_c4_tokens(
        args.snapshot,
        revision=args.c4_revision,
        data_file=args.c4_file,
        documents=args.c4_documents,
        offset=args.offset,
        samples=args.samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    protocol = {
        "dataset": C4_DATASET,
        "dataset_revision": args.c4_revision,
        "data_file": args.c4_file,
        "data_file_sha256": (
            C4_VALIDATION_FILE_SHA256
            if args.c4_revision == C4_REVISION and args.c4_file == C4_VALIDATION_FILE
            else None
        ),
        "dataset_fingerprint": fingerprint,
        "documents": args.c4_documents,
        "document_join": "single-space",
        "tokenizer_policy": "source-model-add-special-tokens",
        "bos_token_id": bos_token_id,
        "offset": args.offset,
        "samples": args.samples,
        "sequence_length": args.sequence_length,
        "token_chunk_size": args.token_chunk_size,
        "token_hash": _token_hash(tokens),
        "model_revision": args.model_revision,
        "arms": [
            {
                "name": name,
                "mode": mode,
                "run_output": str(path.resolve()),
                "global_tuning_pointer": None if pointer is None else str(pointer.resolve()),
            }
            for name, mode, path, pointer in arms
        ],
        "primary_baseline": args.primary_baseline,
        "primary_candidate": args.primary_candidate,
    }
    checkpoint_path = args.output.with_name(args.output.stem + ".checkpoint.json")
    if args.output.is_file():
        completed = json.loads(args.output.read_text(encoding="utf-8"))
        if completed.get("status") != "completed" or completed.get("protocol") != protocol:
            raise ValueError("existing non-WikiText output protocol differs")
        return 0
    checkpoint: dict[str, Any] = {
        "schema_version": 1,
        "status": "in_progress",
        "protocol": protocol,
        "sequences": {},
    }
    if checkpoint_path.is_file():
        restored_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            restored_checkpoint.get("schema_version") != 1
            or restored_checkpoint.get("protocol") != protocol
            or not isinstance(restored_checkpoint.get("sequences"), dict)
        ):
            raise ValueError("non-WikiText checkpoint protocol differs")
        checkpoint = restored_checkpoint

    def restored(name: str) -> tuple[KlSequenceResult, ...]:
        payload = cast(dict[str, object], checkpoint["sequences"]).get(name, [])
        if not isinstance(payload, list):
            raise ValueError("non-WikiText checkpoint arm is invalid")
        return tuple(
            KlSequenceResult(
                float(item["negative_log_likelihood"]),
                float(item["kl_nats_per_token"]),
                int(item["token_count"]),
            )
            for item in payload
        )

    def save_progress(name: str, sequences: tuple[KlSequenceResult, ...]) -> None:
        cast(dict[str, object], checkpoint["sequences"])[name] = [
            to_dict(item) for item in sequences
        ]
        atomic_write_json(checkpoint_path, checkpoint)

    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    results: dict[str, KlBudgetArmResult] = {}
    manifests: dict[str, dict[str, object]] = {}
    frozen_identity: dict[str, object] | None = None
    with acquire_device_lease(args.device):
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=_dtype(config),
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        for name, mode, run_output, tuning_pointer in arms:
            global_tuning_override = None
            if tuning_pointer is not None:
                global_tuning_override = from_dict(
                    ArtifactRef,
                    json.loads(tuning_pointer.read_text(encoding="utf-8")),
                    path=f"arm[{name}].global_tuning",
                )
            loaded = load_frozen_run(
                run_output,
                args.snapshot,
                source_name=MODEL_SOURCE,
                revision=args.model_revision,
                device=args.device,
                verify_hashes=False,
                backend="factorized",
                use_global_tuning=mode != "prekd",
                global_tuning_override=global_tuning_override,
            )
            observed_identity: dict[str, object] = {
                "model_hash": loaded.identity.model_hash,
                "config_hash": loaded.identity.config_hash,
                "plan_hash": loaded.identity.plan_hash,
            }
            if frozen_identity is not None and observed_identity != frozen_identity:
                raise ValueError("non-WikiText arms have different frozen identities")
            frozen_identity = observed_identity
            manifests[name] = {
                "mode": mode,
                "run_output": str(run_output.resolve()),
                "global_tuning": (
                    None if loaded.global_tuning is None else to_dict(loaded.global_tuning)
                ),
                "global_tuning_pointer": (
                    None if tuning_pointer is None else str(tuning_pointer.resolve())
                ),
            }

            def save_arm_progress(
                sequences: tuple[KlSequenceResult, ...],
                arm: str = name,
            ) -> None:
                save_progress(arm, sequences)

            results[name] = _evaluate_arm(
                name,
                teacher,
                loaded.model,
                tokens,
                device=args.device,
                token_chunk_size=args.token_chunk_size,
                completed=restored(name),
                progress=save_arm_progress,
            )
            del loaded
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        del teacher

    adjacent = {
        f"{after}_minus_{before}": _comparison(results[before], results[after])
        for before, after in zip(names, names[1:], strict=False)
    }
    primary = _comparison(
        results[args.primary_baseline],
        results[args.primary_candidate],
    )
    atomic_write_json(
        args.output,
        {
            "schema_version": 1,
            "status": "completed",
            "role": "analysis-only pinned non-WikiText KD quality gate",
            "protocol": protocol,
            "frozen_identity": frozen_identity,
            "arms": manifests,
            "results": {name: to_dict(result) for name, result in results.items()},
            "paired_adjacent_arms": adjacent,
            "primary_comparison": {
                "candidate_minus_baseline": (
                    f"{args.primary_candidate}_minus_{args.primary_baseline}"
                ),
                **primary,
            },
        },
    )
    checkpoint["status"] = "completed"
    atomic_write_json(checkpoint_path, checkpoint)
    return 0


def main(arguments: list[str] | None = None) -> int:
    return run(_parser().parse_args(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
