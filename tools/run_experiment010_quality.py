"""Run only the retained Experiment 010 quality benchmark."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation
from nanoquant.quality_evaluation_workflow import render_quality_evaluation_markdown

ROOT = Path(__file__).resolve().parent.parent
RUN_OUTPUT = ROOT / "evidence/010/010-compress-and-benchmark-gemma-3-270m-it"
PACKED_ARTIFACT = ROOT / "outputs/010/packed"
QUALITY_JSON = ROOT / "Results/010/010-compress-and-benchmark-gemma-3-270m-it-quality.json"
QUALITY_MARKDOWN = ROOT / "Results/010/010-compress-and-benchmark-gemma-3-270m-it-quality.md"
PROGRESS_LOG = ROOT / "outputs/010/quality-only.log"


def _progress(event: str, fields: Mapping[str, object]) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    print(line, flush=True)
    with PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> int:
    manifest = cast(
        dict[str, object],
        json.loads((RUN_OUTPUT / "manifest.json").read_text(encoding="utf-8")),
    )
    resolved = cast(dict[str, object], manifest["resolved_config"])
    canonical = cast(dict[str, object], resolved["canonical_run_config"])
    distillation = cast(dict[str, object], canonical["distillation"])
    request = QualityEvaluationRequest(
        snapshot=Path(str(resolved["snapshot"])),
        source=str(resolved["source"]),
        revision=str(resolved["revision"]),
        run_output=RUN_OUTPUT,
        device=str(resolved["device"]),
        backend="factorized",
        use_global_tuning=bool(distillation["enabled"]),
        wikitext_samples=64,
        wikitext_sequence_length=128,
        task_names=(
            "piqa",
            "arc_easy",
            "arc_challenge",
            "hellaswag",
            "winogrande",
            "boolq",
        ),
        task_limit=200,
        local_files_only=True,
        maximum_wddm_shared_bytes=int(str(resolved["maximum_wddm_shared_bytes"])),
        packed_artifact=PACKED_ARTIFACT,
    )
    _progress(
        "quality_only_started",
        {
            "wikitext_batch_size": request.wikitext_batch_size,
            "task_batch_size": request.task_batch_size,
            "quality_json": str(QUALITY_JSON),
        },
    )
    try:
        result = execute_quality_evaluation(request, progress=_progress)
        payload = {
            **result,
            "experiment": {
                "config_hash": manifest["config_hash"],
                "resolved_config": canonical,
                "launcher": manifest["launcher"],
            },
        }
        atomic_write_json(QUALITY_JSON, payload)
        atomic_write_text(QUALITY_MARKDOWN, render_quality_evaluation_markdown(payload))
    except BaseException as error:
        _progress(
            "quality_only_failed",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise
    _progress(
        "quality_only_completed",
        {"quality_json": str(QUALITY_JSON), "quality_markdown": str(QUALITY_MARKDOWN)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
