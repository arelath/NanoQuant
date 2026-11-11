"""Build reusable full-vocabulary KL teacher targets without a compression run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.hf_task_evaluation import hash_hf_tokenizer_snapshot
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import DenseKlSpliceEvaluator, SpliceReconstructionSet
from nanoquant.infrastructure.kl_teacher_cache import (
    commit_active_kl_teacher_cache,
    load_active_kl_teacher_cache,
)
from nanoquant.infrastructure.kl_teacher_dataset import (
    KlTeacherDataset,
    KlTeacherProtocol,
    load_kl_teacher_dataset,
    publish_kl_teacher_dataset,
    require_compatible_teacher_dataset,
    token_hash,
)
from nanoquant.infrastructure.memory_cleanup import release_memory
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.runs import RunLease
from nanoquant.kl_budget_workflow import _dtype, _model_config_hash
from nanoquant.quality_evaluation import _wikitext_tokens


def build_kl_teacher_dataset(
    *,
    snapshot: Path,
    source: str,
    revision: str,
    output: Path,
    samples: int = 48,
    sequence_length: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    local_files_only: bool = False,
    reuse_cache_root: Path | None = None,
) -> KlTeacherDataset:
    if samples <= 0 or sequence_length < 2 or batch_size <= 0:
        raise ValueError("KL teacher dataset dimensions must be positive, with at least two tokens")
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    model_dtype = _dtype(config)
    attention = adapter_for_config(config).attention_implementation
    tokenizer_hash = hash_hf_tokenizer_snapshot(snapshot)
    output.mkdir(parents=True, exist_ok=True)
    with RunLease(output / ".active-lease.json"):
        if (output / "dataset.json").exists():
            dataset = load_kl_teacher_dataset(output)
            require_compatible_teacher_dataset(
                dataset,
                source=source,
                revision=revision,
                model_hash=_model_config_hash(config),
                tokenizer_hash=tokenizer_hash,
                model_dtype=model_dtype,
                attention_implementation=attention,
                device=device,
                batch_size=batch_size,
                samples=samples,
                sequence_length=sequence_length,
                vocabulary_size=int(config["vocab_size"]),
            )
            print(f"Reusing completed KL teacher dataset {output}", flush=True)
            return dataset

        print("Preparing the pinned WikiText inputs", flush=True)
        tokens, fingerprint, bos = _wikitext_tokens(
            snapshot,
            samples=samples,
            sequence_length=sequence_length,
            local_files_only=local_files_only,
        )
        protocol = KlTeacherProtocol(
            source=source,
            revision=revision,
            model_hash=_model_config_hash(config),
            tokenizer_hash=tokenizer_hash,
            model_dtype=str(model_dtype),
            attention_implementation=attention,
            device=device,
            batch_size=batch_size,
            samples=samples,
            sequence_length=sequence_length,
            vocabulary_size=int(config["vocab_size"]),
            dataset_fingerprint=fingerprint,
            bos_token_id=bos,
            token_hash=token_hash(tokens),
        )
        cache = load_active_kl_teacher_cache(output, protocol.cache_key)
        if cache is None and reuse_cache_root is not None:
            cache = load_active_kl_teacher_cache(reuse_cache_root, protocol.cache_key)
            if cache is None:
                raise ValueError("requested source KL teacher cache is incomplete")
            print("Importing validated teacher targets", flush=True)
            destination = LocalArtifactStore(output / "artifacts")
            destination.import_validated(
                LocalArtifactStore(reuse_cache_root / "artifacts"),
                cache.reference.artifact_id,
            )
            atomic_write_json(
                output / "active.json",
                {
                    "schema_version": 1,
                    "cache_key": cache.cache_key,
                    "artifact": to_dict(cache.reference),
                },
            )
        if cache is None:
            expected_bytes = samples * sequence_length * int(config["vocab_size"]) * 2
            print(f"Computing teacher targets ({expected_bytes / 2**30:.2f} GiB of FP16 log probabilities)", flush=True)
            with acquire_device_lease(device):
                teacher = load_causal_language_model(
                    snapshot,
                    torch_dtype=model_dtype,
                    attention_implementation=attention,
                    local_files_only=local_files_only,
                ).to(device)
                evaluator = DenseKlSpliceEvaluator(
                    teacher,
                    SpliceReconstructionSet((), (), ()),
                    tokens,
                    device=device,
                    batch_size=batch_size,
                )
                baseline_nll, batches = evaluator.teacher_cache_state()
                cache = commit_active_kl_teacher_cache(output, protocol.cache_key, baseline_nll, batches)
                del evaluator, teacher
                release_memory(device)
        dataset = publish_kl_teacher_dataset(output, protocol, tokens, cache)
        print(f"KL teacher dataset complete at {output}", flush=True)
        return dataset


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Hugging Face teacher model ID")
    parser.add_argument("--revision", required=True, help="pinned model commit")
    parser.add_argument("--snapshot", type=Path, help="existing local snapshot; otherwise download the pinned model")
    parser.add_argument("--output", type=Path, help="defaults to Datasets/<model>-kl-<samples>x<length>")
    parser.add_argument("--wikitext-samples", type=int, default=48)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--reuse-cache-root",
        type=Path,
        help="extract existing validated targets without teacher inference",
    )
    args = parser.parse_args(arguments)
    if len(args.revision) != 40 or any(character not in "0123456789abcdef" for character in args.revision):
        parser.error("--revision must be a pinned 40-character commit hash")
    output = args.output or (
        Path(__file__).resolve().parents[2]
        / "Datasets"
        / f"{args.source.split('/')[-1]}-kl-{args.wikitext_samples}x{args.sequence_length}"
    )
    snapshot = args.snapshot or Path(
        snapshot_download(
            repo_id=args.source,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken"],
        )
    )
    build_kl_teacher_dataset(
        snapshot=snapshot,
        source=args.source,
        revision=args.revision,
        output=output,
        samples=args.wikitext_samples,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=args.device,
        local_files_only=args.local_files_only,
        reuse_cache_root=args.reuse_cache_root,
    )
    return 0
