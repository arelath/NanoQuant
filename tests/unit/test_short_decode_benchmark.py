from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

import nanoquant.short_decode_workflow as workflow
from nanoquant.recipes.legacy_short_decode import (
    LEGACY_SHORT_DECODE_BENCHMARK,
    LEGACY_SHORT_DECODE_CONFIG,
)
from nanoquant.short_decode_benchmark import (
    ShortDecodeBenchmarkRequest,
    force_prompt_tokens,
)
from nanoquant.short_decode_workflow import (
    ShortDecodeBenchmarkExperiment,
    execute_short_decode_experiment,
    resolve_short_decode_experiment,
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


def test_retained_recipe_preserves_legacy_short_decode_workload() -> None:
    request = LEGACY_SHORT_DECODE_BENCHMARK.request

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
        LEGACY_SHORT_DECODE_CONFIG,
        LEGACY_SHORT_DECODE_BENCHMARK,
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
        LEGACY_SHORT_DECODE_BENCHMARK.request,
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
        LEGACY_SHORT_DECODE_CONFIG,
        experiment,
        launcher_path=launcher,
    )

    assert observed == [request]
    assert payload["experiment"]["launcher"]["experiment_number"] == 2
    assert payload["experiment"]["resolved_config"]["intent"]["name"] == (
        "002-benchmark-gemma-3-1b-it"
    )
    assert json.loads(output.read_text(encoding="utf-8")) == payload
