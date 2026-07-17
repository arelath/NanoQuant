from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from recipes._delta import config_delta, run_config_defaults

import nanoquant.short_decode_workflow as workflow
from nanoquant.short_decode_benchmark import (
    LegacyShortDecodeCase,
    ShortDecodeBenchmarkRequest,
    force_prompt_tokens,
)
from nanoquant.short_decode_workflow import (
    ShortDecodeBenchmarkExperiment,
    execute_short_decode_experiment,
    resolve_short_decode_experiment,
)

_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
_DEFAULTS = run_config_defaults("google/gemma-3-1b-it")
_CONFIG = config_delta(
    _DEFAULTS,
    model=config_delta(
        _DEFAULTS.model,
        revision=_MODEL_REVISION,
        tokenizer_revision=_MODEL_REVISION,
        sequence_length=128,
    ),
    intent=config_delta(
        _DEFAULTS.intent,
        experiment_number=2,
        name="benchmark-gemma-3-1b-it",
        purpose="Exercise the paired short-decode workflow contract.",
        hypothesis="The packed runtime completes the configured decode workload.",
        tags=("runtime", "decode", "paired", "memory"),
    ),
    evaluation=config_delta(_DEFAULTS.evaluation, suites=("runtime-short-decode-v1",)),
)
_BENCHMARK = ShortDecodeBenchmarkExperiment(
    ShortDecodeBenchmarkRequest(
        snapshot=Path("google/gemma-3-1b-it"),
        run_output=Path("evidence/m4/gemma-pageable-v28-four-block-canary"),
        runtime_bundle=Path("evidence/m6/gemma-pageable-v28-runtime-bundle"),
        source="google/gemma-3-1b-it",
        revision=_MODEL_REVISION,
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
        legacy_summary_sha256=(
            "fb54cfd9f8244b8a6dec30dbd8450b8a8cda729c728ab4959ddc9112954dfaa8"
        ),
    ),
    Path("evidence/m9/002-gemma-3-1b-it-short-decode.json"),
    resolve_model_from_config=True,
)
class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, _text: str, *, return_tensors: str, padding: bool) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding
        return {
            "input_ids": torch.tensor(((1, 10, 11),)),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [7] if text == " hello" else [2]


def test_force_prompt_tokens_matches_legacy_truncate_and_fill_policy() -> None:
    tokenizer = _FakeTokenizer()

    assert force_prompt_tokens(tokenizer, "prompt", 2).tolist() == [[1, 10]]
    assert force_prompt_tokens(tokenizer, "prompt", 5).tolist() == [[1, 10, 11, 7, 7]]


def test_short_decode_request_rejects_invalid_protocols(tmp_path: Path) -> None:
    base = ShortDecodeBenchmarkRequest(
        tmp_path / "snapshot",
        tmp_path / "run",
        tmp_path / "bundle",
        "source",
        "revision",
    )

    with pytest.raises(ValueError, match="warmups"):
        replace(base, warmups=-1)
    with pytest.raises(ValueError, match="lengths"):
        replace(base, max_new_tokens=1)
    with pytest.raises(ValueError, match="hash"):
        replace(base, legacy_summary_sha256="INVALID")


def test_short_decode_fixture_exercises_all_comparison_cases() -> None:
    request = _BENCHMARK.request

    assert request.dtype == "bfloat16"
    assert request.prompt_tokens == 32
    assert request.max_new_tokens == 32
    assert request.warmups == 1
    assert request.repetitions == 3
    assert request.seed == 0
    assert request.top_k == 32
    assert request.temperature == 0.8
    assert request.backend == "factorized"
    assert request.prompt == "Explain why compact language models are useful for local inference."
    assert tuple(case.name for case in request.legacy_cases) == (
        "fp_original",
        "nq_eager",
        "nq_gemv_kernel",
    )


def test_short_decode_experiment_resolution_is_repository_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = tmp_path / "repo" / "experiments" / "002-example.py"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("# fixture\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    monkeypatch.setattr(
        workflow,
        "snapshot_download",
        lambda *, repo_id, revision: str(snapshot),
    )

    resolved = resolve_short_decode_experiment(
        _CONFIG,
        _BENCHMARK,
        launcher_path=launcher,
    )

    assert resolved.request.snapshot == snapshot.resolve()
    assert resolved.request.run_output == (
        tmp_path / "repo" / "evidence/m4/gemma-pageable-v28-four-block-canary"
    )
    assert resolved.request.runtime_bundle == (
        tmp_path / "repo" / "evidence/m6/gemma-pageable-v28-runtime-bundle"
    )
    assert resolved.result_path == (
        tmp_path / "repo" / "evidence/m9/002-gemma-3-1b-it-short-decode.json"
    )


def test_short_decode_workflow_records_config_and_launcher_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "result.json"
    request = replace(
        _BENCHMARK.request,
        snapshot=tmp_path / "snapshot",
        run_output=tmp_path / "run",
        runtime_bundle=tmp_path / "bundle",
    )
    experiment = ShortDecodeBenchmarkExperiment(request, output)
    observed: list[ShortDecodeBenchmarkRequest] = []
    launcher = tmp_path / "002-benchmark-gemma-3-1b-it.py"
    launcher.write_text("# provenance fixture\n", encoding="utf-8")

    def benchmark(resolved: ShortDecodeBenchmarkRequest) -> dict[str, Any]:
        observed.append(resolved)
        return {"schema_version": 1, "passed": True, "cases": []}

    monkeypatch.setattr(workflow, "execute_short_decode_benchmark", benchmark)
    payload = execute_short_decode_experiment(
        _CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert observed == [request]
    assert payload["experiment"]["launcher"]["experiment_number"] == 2
    assert payload["experiment"]["resolved_config"]["intent"]["name"] == (
        "benchmark-gemma-3-1b-it"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
