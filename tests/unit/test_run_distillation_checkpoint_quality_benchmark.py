from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoquant.quality_evaluation import QualityEvaluationRequest
from tools.run_distillation_checkpoint_quality_benchmark import _matched_base_result


def _request(tmp_path: Path) -> QualityEvaluationRequest:
    return QualityEvaluationRequest(
        snapshot=tmp_path,
        source="model",
        revision="revision",
        run_output=tmp_path,
        wikitext_samples=64,
        wikitext_sequence_length=128,
        task_names=("piqa",),
        task_limit=200,
    )


def test_matched_base_result_rejects_protocol_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "model": {"source": "model", "revision": "revision"},
                "protocol": {
                    "wikitext_samples": 64,
                    "wikitext_sequence_length": 128,
                    "task_names": ["piqa"],
                    "task_limit": 25,
                },
                "results": {"base": {"wikitext": {}, "tasks": []}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        _matched_base_result(path, _request(tmp_path))


def test_matched_base_result_returns_protocol_and_result(tmp_path: Path) -> None:
    path = tmp_path / "quality.json"
    protocol = {
        "wikitext_samples": 64,
        "wikitext_sequence_length": 128,
        "task_names": ["piqa"],
        "task_limit": 200,
    }
    base = {"wikitext": {"perplexity": 1.0}, "tasks": []}
    path.write_text(
        json.dumps(
            {
                "model": {"source": "model", "revision": "revision"},
                "protocol": protocol,
                "results": {"base": base},
            }
        ),
        encoding="utf-8",
    )

    assert _matched_base_result(path, _request(tmp_path)) == (base, protocol)
