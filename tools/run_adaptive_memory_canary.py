"""Run a complete exact-snapshot compression through adaptive memory planning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import _paths  # noqa: F401
from recipes.base_compression import (
    BASE_COMPRESSION_TEMPLATE,
    GEMMA_3_4B_COMPRESSION_TEMPLATE,
    LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
)
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.config.codec import config_hash
from nanoquant.config.schema import (
    ActivationGpuCacheMode,
    CalibrationMethod,
    ExecutorKind,
    MemoryPolicyConfig,
    MemoryPolicyMode,
    MemoryPolicyProfile,
    RunConfig,
)
from nanoquant.infrastructure.hf_calibration_dataset import (
    load_or_prepare_calibration,
    materialize_pinned_calibration,
)
from nanoquant.infrastructure.preprocessing_materialization import materialize_resident_preprocessing
from nanoquant.infrastructure.retained_recipe import load_retained_run_recipe
from nanoquant.resident_workflow import (
    ResidentExecutionOptions,
    ResolvedResidentInputs,
    execute_resident_workflow,
)


@dataclass(frozen=True, slots=True)
class CanaryCase:
    source: str
    revision: str
    cache_directory: str
    template: RunConfig


CASES = {
    "gemma-3-270m-it": CanaryCase(
        "google/gemma-3-270m-it",
        "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
        "models--google--gemma-3-270m-it",
        BASE_COMPRESSION_TEMPLATE,
    ),
    "gemma-3-1b-it": CanaryCase(
        "google/gemma-3-1b-it",
        "dcc83ea841ab6100d6b47a070329e1ba4cf78752",
        "models--google--gemma-3-1b-it",
        BASE_COMPRESSION_TEMPLATE,
    ),
    "llama-3.2-1b-instruct": CanaryCase(
        "meta-llama/Llama-3.2-1B-Instruct",
        "9213176726f574b556790deb65791e0c5aa438b6",
        "models--meta-llama--Llama-3.2-1B-Instruct",
        LLAMA_3_2_1B_INSTRUCT_COMPRESSION_TEMPLATE,
    ),
    "gemma-3-4b-it": CanaryCase(
        "google/gemma-3-4b-it",
        "093f9f388b31de276ce2de164bdc2081324b9767",
        "models--google--gemma-3-4b-it",
        GEMMA_3_4B_COMPRESSION_TEMPLATE,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(CASES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "hub",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in MemoryPolicyProfile),
        default=MemoryPolicyProfile.BALANCED.value,
    )
    parser.add_argument("--samples", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--recipe-run",
        type=Path,
        help=(
            "Completed retained run whose canonical RunConfig should be used as the semantic recipe; "
            "only intent, exact model identity, adaptive execution controls, and disabled post-compression "
            "distillation are changed."
        ),
    )
    parser.add_argument(
        "--allow-interrupted-recipe-run",
        action="store_true",
        help="Allow --recipe-run to use an interrupted retained run after its artifacts were separately validated.",
    )
    parser.add_argument(
        "--preprocessing-run",
        type=Path,
        help="Validated retained run whose transitive calibration/objective/plan graph should be materialized.",
    )
    parser.add_argument("--interrupt-after-block-commits", type=int)
    parser.add_argument("--interrupt-after-layer-commits", type=int)
    parser.add_argument("--replan-memory", action="store_true")
    parser.add_argument("--skip-source-hash-verification", action="store_true")
    parser.add_argument(
        "--calibration-input-run",
        type=Path,
        help="Validated run whose deterministic calibration tokens should be materialized locally.",
    )
    parser.add_argument(
        "--calibration-input-snapshot",
        type=Path,
        help="Snapshot that generated --calibration-input-run; tokenizer behavior must match the target.",
    )
    parser.add_argument(
        "--forward-only-calibration",
        action="store_true",
        help="Explicit semantic fallback for CPU-offloaded runs without reusable Fisher preprocessing.",
    )
    return parser


def _snapshot(cache_root: Path, case: CanaryCase) -> Path:
    snapshot = cache_root / case.cache_directory / "snapshots" / case.revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"requested local snapshot is unavailable: {snapshot}")
    return snapshot.resolve()


def _tokenizer_identity(snapshot: Path) -> str:
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    digest = hashlib.sha256()
    for token, index in sorted(tokenizer.get_vocab().items()):
        digest.update(str(index).encode())
        digest.update(b"\0")
        digest.update(token.encode())
        digest.update(b"\0")
    digest.update(json.dumps(tokenizer.special_tokens_map, sort_keys=True).encode())
    digest.update(str(tokenizer.chat_template).encode())
    return "sha256:" + digest.hexdigest()


def _config(case: CanaryCase, args: argparse.Namespace, template: RunConfig) -> RunConfig:
    sample_count = template.calibration.sample_count if args.samples is None else args.samples
    if sample_count <= 0:
        raise ValueError("calibration sample count must be positive")
    calibration = replace(
        template.calibration,
        sample_count=sample_count,
        method=(CalibrationMethod.FORWARD_ONLY if args.forward_only_calibration else template.calibration.method),
    )
    return replace(
        template,
        intent=replace(template.intent, experiment_number=None, name=args.output.name),
        model=replace(
            template.model,
            source=case.source,
            revision=case.revision,
            tokenizer_source=case.source,
            tokenizer_revision=case.revision,
        ),
        calibration=calibration,
        reproducibility=replace(template.reproducibility, seed=args.seed),
        distillation=replace(template.distillation, enabled=False),
        runtime=replace(
            template.runtime,
            executor=ExecutorKind.AUTO,
            memory_policy=MemoryPolicyConfig(
                mode=MemoryPolicyMode.ADAPTIVE,
                profile=MemoryPolicyProfile(args.profile),
            ),
            activations=replace(template.runtime.activations, gpu_cache=ActivationGpuCacheMode.AUTO),
            source_streaming=replace(
                template.runtime.source_streaming,
                verify_tensor_hashes=not args.skip_source_hash_verification,
            ),
            on_cuda_oom=("reduce_batch_size", "move_activations_down_one_tier", "fail"),
        ),
    )


def main() -> int:
    args = _parser().parse_args()
    case = CASES[args.model]
    template = case.template
    maximum_wddm_shared_bytes = None
    if args.recipe_run is not None:
        retained = load_retained_run_recipe(
            args.recipe_run,
            expected_source=case.source,
            expected_revision=case.revision,
            allowed_statuses=(
                ("completed", "interrupted")
                if args.allow_interrupted_recipe_run
                else ("completed",)
            ),
        )
        template = retained.config
        maximum_wddm_shared_bytes = retained.maximum_wddm_shared_bytes
    snapshot = _snapshot(args.cache_root.resolve(), case)
    output = args.output.resolve()
    kernel_cache = output / "state" / "kernel-cache"
    triton_cache = kernel_cache / "triton"
    torchinductor_cache = kernel_cache / "torchinductor"
    triton_cache.mkdir(parents=True, exist_ok=True)
    torchinductor_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(triton_cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(torchinductor_cache)
    config = _config(case, args, template)
    preparation_id = config_hash(config)
    if (args.calibration_input_run is None) != (args.calibration_input_snapshot is None):
        raise ValueError("calibration input run and snapshot must be supplied together")
    if args.calibration_input_run is not None and args.calibration_input_snapshot is not None:
        target_tokenizer = _tokenizer_identity(snapshot)
        source_tokenizer = _tokenizer_identity(args.calibration_input_snapshot.resolve())
        if source_tokenizer != target_tokenizer:
            raise ValueError("source and target calibration tokenizers are not behaviorally identical")
        calibration = materialize_pinned_calibration(
            args.calibration_input_run.resolve(),
            output,
            sample_count=config.calibration.sample_count,
            sequence_length=config.model.sequence_length,
            seed=config.reproducibility.seed,
            preparation_id=preparation_id,
            tokenizer_identity=target_tokenizer,
        )
    else:
        calibration = load_or_prepare_calibration(
            snapshot,
            output,
            sample_count=config.calibration.sample_count,
            sequence_length=config.model.sequence_length,
            seed=config.reproducibility.seed,
            preparation_id=preparation_id,
        )
    preprocessing = (
        None
        if args.preprocessing_run is None
        else materialize_resident_preprocessing(args.preprocessing_run, output)
    )
    inputs = ResolvedResidentInputs(
        snapshot=snapshot,
        output=output,
        registry_root=args.run_root.resolve(),
        token_ids=calibration.input_ids,
        quality_token_ids=(calibration.input_ids[:1, :8] if config.evaluation.inline_quality else None),
        launcher_path=Path(__file__).resolve(),
        precomputed_calibration=(None if preprocessing is None else preprocessing.calibration),
        precomputed_objectives=(None if preprocessing is None else preprocessing.objectives),
        precomputed_plan=(None if preprocessing is None else preprocessing.plan),
    )
    options = ResidentExecutionOptions(
        interrupt_after_layer_commits=args.interrupt_after_layer_commits,
        interrupt_after_block_commits=args.interrupt_after_block_commits,
        replan_memory=args.replan_memory,
        maximum_wddm_shared_bytes=maximum_wddm_shared_bytes,
    )
    try:
        workflow = execute_resident_workflow(config, inputs, options)
    except InterruptedError as exc:
        print(json.dumps({"status": "interrupted", "reason": str(exc), "output": str(output)}, indent=2))
        return 0
    result = workflow.quantization
    print(
        json.dumps(
            {
                "status": "completed",
                "model": case.source,
                "revision": case.revision,
                "blocks": len(result.blocks),
                "layers": sum(len(block.layers) for block in result.blocks),
                "reused": result.reused_commit_count,
                "bpw": result.frozen_model.effective_bpw,
                "reference_nll": result.reference_nll,
                "compressed_nll": result.compressed_nll,
                "logit_mse": result.logit_mse,
                "peak_device_bytes": result.peak_device_bytes,
                "peak_host_bytes": result.peak_host_bytes,
                "artifact_bytes": result.artifact_bytes,
                "elapsed_seconds": result.elapsed_seconds,
                "report": result.report.artifact_id,
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
