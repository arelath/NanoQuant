"""Build or resume a KL splice sensitivity profile from a completed resident run."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

from nanoquant.application.kl_budget import (
    KL_BUDGET_EVALUATOR_VERSION,
    KlBudgetProfile,
    KlBudgetProvenance,
    KlBudgetRequest,
    KlBudgetWorkflow,
    load_kl_budget_profile,
    persist_kl_budget_profile,
)
from nanoquant.config.codec import to_dict
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.kl_splice import (
    DenseKlSpliceEvaluator,
    SpliceReconstructionSet,
    load_splice_reconstructions_from_run,
)
from nanoquant.infrastructure.kl_teacher_cache import (
    commit_active_kl_teacher_cache,
    load_active_kl_teacher_cache,
)
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.quality_evaluation import _wikitext_tokens


def _dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _token_hash(tokens: torch.Tensor) -> str:
    payload = tokens.contiguous().view(torch.uint8).numpy().tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _expected_blocks(config: dict[str, object]) -> int:
    value = config.get("num_hidden_layers")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("model config must contain a positive integer num_hidden_layers")
    return value


def _model_config_hash(config: dict[str, object]) -> str:
    encoded = json.dumps(config, sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _profile_recipe_hash(base_recipe_hash: str, global_tuning: str | None) -> str:
    payload = f"{base_recipe_hash}|kl-budget-evaluator-v{KL_BUDGET_EVALUATOR_VERSION}"
    if global_tuning is not None:
        payload += f"|{global_tuning}"
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _teacher_cache_key(
    *,
    source: str,
    revision: str,
    model_hash: str,
    token_hash: str,
    model_dtype: torch.dtype,
    attention_implementation: str | None,
    device: str,
    batch_size: int,
) -> str:
    payload = json.dumps(
        {
            "schema_version": 1,
            "source": source,
            "revision": revision,
            "model_hash": model_hash,
            "token_hash": token_hash,
            "model_dtype": str(model_dtype),
            "teacher_cache_dtype": str(torch.float16),
            "attention_implementation": attention_implementation,
            "device": device,
            "batch_size": batch_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def default_kl_budget_arms(reconstructions: SpliceReconstructionSet) -> tuple[str, ...]:
    units = tuple(unit for unit, _members in reconstructions.unit_members)
    blocks = tuple(
        sorted({member.block.index for _unit, members in reconstructions.unit_members for member in members})
    )
    types = tuple(sorted({unit.split(":", 1)[1] for unit in units}))
    return (
        "full",
        *(f"type:{name}" for name in types),
        *(f"block:{block}" for block in blocks),
        *(f"unit:{unit}" for unit in units),
    )


def add_kl_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--wikitext-samples", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--token-chunk-size", type=int, default=128)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--teacher-cache-mode", choices=("cpu", "on_the_fly"), default="cpu")
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--use-global-tuning", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")


def execute_kl_budget(args: argparse.Namespace) -> int:
    if args.wikitext_samples <= 0 or args.sequence_length < 2:
        raise ValueError("KL budget dataset dimensions must be positive")
    if args.batch_size <= 0 or args.token_chunk_size <= 0:
        raise ValueError("KL budget batch and token chunk sizes must be positive")
    args.profile_output.mkdir(parents=True, exist_ok=True)
    profile_path = args.profile_output / "kl-budget-profile.json"
    checkpoint = load_kl_budget_profile(profile_path) if profile_path.exists() else None
    config_payload = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise ValueError("model config must be a JSON object")
    config = {str(key): value for key, value in config_payload.items()}
    adapter = adapter_for_config(config)
    tokens, dataset_fingerprint, _bos = _wikitext_tokens(
        args.snapshot,
        samples=args.wikitext_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    token_hash = _token_hash(tokens)
    with acquire_device_lease(args.device):
        loaded = load_splice_reconstructions_from_run(
            args.run_output,
            _expected_blocks(config),
            device=args.device,
            source=args.source,
            revision=args.revision,
            model_config_hash=_model_config_hash(config),
            use_global_tuning=args.use_global_tuning,
        )
        reconstructions = loaded.reconstructions
        identity = loaded.identity
        model_hash = identity.model_hash
        global_tuning = loaded.global_tuning
        del loaded
        gc.collect()
        model_dtype = _dtype(config)
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=model_dtype,
            attention_implementation=adapter.attention_implementation,
            local_files_only=args.local_files_only,
        ).to(args.device)
        teacher.eval()
        evaluator = DenseKlSpliceEvaluator(
            teacher,
            reconstructions,
            tokens,
            device=args.device,
            batch_size=args.batch_size,
            token_chunk_size=args.token_chunk_size,
            teacher_cache_mode=args.teacher_cache_mode,
        )
        teacher_cache = None
        teacher_cache_reused = False
        if args.teacher_cache_root is not None:
            if args.teacher_cache_mode != "cpu":
                raise ValueError("persistent KL teacher cache requires --teacher-cache-mode cpu")
            cache_key = _teacher_cache_key(
                source=args.source,
                revision=args.revision,
                model_hash=model_hash,
                token_hash=token_hash,
                model_dtype=model_dtype,
                attention_implementation=adapter.attention_implementation,
                device=args.device,
                batch_size=args.batch_size,
            )
            teacher_cache = load_active_kl_teacher_cache(args.teacher_cache_root, cache_key)
            if teacher_cache is None:
                baseline_nll, cache_batches = evaluator.teacher_cache_state()
                teacher_cache = commit_active_kl_teacher_cache(
                    args.teacher_cache_root,
                    cache_key,
                    baseline_nll,
                    cache_batches,
                )
            else:
                evaluator.install_teacher_cache(
                    teacher_cache.baseline_negative_log_likelihood,
                    teacher_cache.batches,
                )
                teacher_cache_reused = True
        base_recipe_hash = identity.config_hash
        base_source_identity = f"{identity.config_hash}|{identity.model_hash}|{identity.plan_hash}"
        recipe_hash = _profile_recipe_hash(
            base_recipe_hash,
            None if global_tuning is None else global_tuning.artifact_id,
        )
        source_identity = (
            base_source_identity
            if global_tuning is None
            else f"{base_source_identity}|{global_tuning.artifact_id}"
        )
        provenance = KlBudgetProvenance(
            args.source,
            args.revision,
            recipe_hash,
            dataset_fingerprint,
            token_hash,
            source_identity,
        )
        arms = tuple(args.arm) if args.arm else default_kl_budget_arms(reconstructions)

        def save_checkpoint(profile: KlBudgetProfile) -> None:
            atomic_write_json(profile_path, to_dict(profile))

        profile = KlBudgetWorkflow().run(
            KlBudgetRequest(provenance, arms),
            evaluator,
            baseline_negative_log_likelihood=evaluator.baseline_negative_log_likelihood,
            resume=checkpoint,
            checkpoint=save_checkpoint,
        )
        save_checkpoint(profile)
        persisted = persist_kl_budget_profile(profile, LocalArtifactStore(args.profile_output / "artifacts"))
        atomic_write_json(
            args.profile_output / "artifact.json",
            {
                "profile_key": profile.profile_key,
                "evaluator_version": KL_BUDGET_EVALUATOR_VERSION,
                "artifact": to_dict(persisted.reference),
                "arm_count": len(profile.arms),
                "complete": profile.complete,
                "teacher_cache_mode": args.teacher_cache_mode,
                "teacher_cache": None if teacher_cache is None else to_dict(teacher_cache.reference),
                "teacher_cache_key": None if teacher_cache is None else teacher_cache.cache_key,
                "teacher_cache_reused": teacher_cache_reused,
                "teacher_cache_bytes": 0 if teacher_cache is None else teacher_cache.tensor_bytes,
                "global_tuning": None if global_tuning is None else to_dict(global_tuning),
            },
        )
    return 0


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_kl_budget_arguments(parser)
    return execute_kl_budget(parser.parse_args(arguments))


__all__ = [
    "add_kl_budget_arguments",
    "default_kl_budget_arms",
    "execute_kl_budget",
    "main",
]
