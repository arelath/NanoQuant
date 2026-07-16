"""Retained recipe fixture for the migrated legacy short-decode benchmark."""

from pathlib import Path

from nanoquant.short_decode_benchmark import LegacyShortDecodeCase, ShortDecodeBenchmarkRequest
from nanoquant.short_decode_workflow import ShortDecodeBenchmarkExperiment

from .._delta import config_delta, run_config_defaults

MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"

_SCHEMA_DEFAULTS = run_config_defaults("google/gemma-3-1b-it")

LEGACY_SHORT_DECODE_CONFIG = config_delta(
    _SCHEMA_DEFAULTS,
    model=config_delta(
        _SCHEMA_DEFAULTS.model,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        sequence_length=128,
    ),
    intent=config_delta(
        _SCHEMA_DEFAULTS.intent,
        experiment_number=2,
        name="002-benchmark-gemma-3-1b-it",
        purpose="Compare source, logical frozen, and production packed short-decode behavior.",
        hypothesis="The immutable packed runtime replaces the legacy mutable GEMV case.",
        baseline_run="legacy-experiment-002",
        tags=("gemma-3-1b-it", "runtime", "decode", "paired", "memory"),
    ),
    evaluation=config_delta(
        _SCHEMA_DEFAULTS.evaluation,
        suites=("runtime-short-decode-v1",),
    ),
)

LEGACY_SHORT_DECODE_BENCHMARK = ShortDecodeBenchmarkExperiment(
    ShortDecodeBenchmarkRequest(
        snapshot=Path("google/gemma-3-1b-it"),
        run_output=Path("evidence/m4/gemma-pageable-v28-four-block-canary"),
        runtime_bundle=Path("evidence/m6/gemma-pageable-v28-runtime-bundle"),
        source="google/gemma-3-1b-it",
        revision=MODEL_REVISION,
        device="cuda:0",
        dtype="bfloat16",
        backend="factorized",
        prompt="Explain why compact language models are useful for local inference.",
        prompt_tokens=32,
        max_new_tokens=32,
        warmups=1,
        repetitions=3,
        seed=0,
        top_k=32,
        temperature=0.8,
        legacy_cases=(
            LegacyShortDecodeCase("fp_original", 8.094968, 2_081_724_928, 2_099_249_152),
            LegacyShortDecodeCase("nq_eager", 8.297656, 1_999_090_176, 2_040_528_896),
            LegacyShortDecodeCase("nq_gemv_kernel", 7.127174, 719_535_616, 734_003_200),
        ),
        legacy_summary_sha256="fb54cfd9f8244b8a6dec30dbd8450b8a8cda729c728ab4959ddc9112954dfaa8",
    ),
    Path("evidence/m9/002-gemma-3-1b-it-short-decode.json"),
    resolve_model_from_config=True,
)

__all__ = ["LEGACY_SHORT_DECODE_BENCHMARK", "LEGACY_SHORT_DECODE_CONFIG"]
