"""Numbered-experiment composition for shared quality evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from huggingface_hub import snapshot_download

from nanoquant.config.codec import config_hash, to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.io_utils import atomic_write_json, atomic_write_text
from nanoquant.infrastructure.runs import launcher_provenance, validate_launcher_number
from nanoquant.quality_evaluation import QualityEvaluationRequest, execute_quality_evaluation


@dataclass(frozen=True, slots=True)
class QualityEvaluationExperiment:
    request: QualityEvaluationRequest
    result_path: Path
    resolve_model_from_config: bool = False
    markdown_path: Path | None = None


def _resolve(path: Path, repository_root: Path) -> Path:
    return path if path.is_absolute() else repository_root / path


def resolve_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> QualityEvaluationExperiment:
    launcher = Path(launcher_path).resolve()
    repository_root = launcher.parent.parent
    request = experiment.request
    if experiment.resolve_model_from_config:
        configured = Path(config.model.source)
        snapshot = (
            configured.resolve()
            if configured.exists()
            else Path(
                snapshot_download(
                    repo_id=config.model.source,
                    revision=str(config.model.revision),
                )
            ).resolve()
        )
    else:
        snapshot = _resolve(request.snapshot, repository_root)
    return QualityEvaluationExperiment(
        replace(
            request,
            snapshot=snapshot,
            source=config.model.source,
            revision=str(config.model.revision),
            run_output=_resolve(request.run_output, repository_root),
        ),
        _resolve(experiment.result_path, repository_root),
        experiment.resolve_model_from_config,
        (
            None
            if experiment.markdown_path is None
            else _resolve(experiment.markdown_path, repository_root)
        ),
    )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"quality Markdown field is not numeric: {name}")
    return float(value)


def _markdown_cell(value: object) -> str:
    rendered = (
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        if isinstance(value, (dict, list, tuple))
        else str(value)
    )
    return rendered.replace("|", "\\|").replace("\n", " ")


def render_quality_evaluation_markdown(payload: dict[str, Any]) -> str:
    """Render a compact deterministic BF16-versus-NanoQuant benchmark report."""

    experiment = cast(dict[str, Any], payload["experiment"])
    config = cast(dict[str, Any], experiment["resolved_config"])
    intent = cast(dict[str, object], config["intent"])
    model = cast(dict[str, object], payload["model"])
    candidate = cast(dict[str, Any], payload["candidate"])
    protocol = cast(dict[str, Any], payload["protocol"])
    comparison = cast(dict[str, Any], payload["comparison"])
    wikitext = cast(dict[str, object], comparison["wikitext"])
    results = cast(dict[str, Any], payload["results"])
    base = cast(dict[str, object], results["base"])
    frozen = cast(dict[str, object], results["frozen"])
    base_ppl = _number(wikitext["base_perplexity"], "base_perplexity")
    frozen_ppl = _number(wikitext["frozen_perplexity"], "frozen_perplexity")
    relative = _number(wikitext["relative_change"], "relative_change")
    lines = [
        f"# Experiment {intent.get('experiment_number')}: {_markdown_cell(model['source'])} quality benchmark",
        "",
        f"- Status: `{'completed' if payload.get('passed') else 'invalid'}`",
        f"- Model: `{_markdown_cell(model['source'])}`",
        f"- Revision: `{_markdown_cell(model['revision'])}`",
        f"- Candidate run: `{_markdown_cell(candidate['run_output'])}`",
        f"- Backend: `{_markdown_cell(candidate['backend'])}`",
        f"- Wall time: {_number(payload['wall_seconds'], 'wall_seconds'):.2f} seconds",
        "",
        "`completed` means all evaluators returned finite metrics; it is not a BF16-quality acceptance gate.",
        "",
        "## Protocol",
        "",
        (
            f"- WikiText-2: {protocol['wikitext_samples']} samples × "
            f"{protocol['wikitext_sequence_length']} tokens, batch {protocol['wikitext_batch_size']}"
        ),
        f"- WikiText token hash: `{_markdown_cell(protocol['wikitext_token_hash'])}`",
        (
            f"- Tasks: {', '.join(str(item) for item in protocol['task_names'])}; "
            f"first {protocol['task_limit']} rows, batch {protocol['task_batch_size']}"
        ),
        f"- Tokenizer hash: `{_markdown_cell(protocol['tokenizer_hash'])}`",
        "",
        "## Quality results",
        "",
        "| Benchmark | Metric | BF16 | NanoQuant | Delta | Ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
        (
            f"| WikiText-2 | perplexity ↓ | {base_ppl:.6f} | {frozen_ppl:.6f} | "
            f"{frozen_ppl - base_ppl:+.6f} ({relative:+.2%}) | {frozen_ppl / base_ppl:.4f}x |"
        ),
    ]
    for item in cast(list[dict[str, object]], comparison["tasks"]):
        baseline = _number(item["base"], "task base")
        value = _number(item["frozen"], "task frozen")
        ratio = item.get("ratio")
        ratio_text = "n/a" if ratio is None else f"{_number(ratio, 'task ratio'):.4f}x"
        lines.append(
            f"| {_markdown_cell(item['task_name'])} | {_markdown_cell(item['metric'])} ↑ | "
            f"{baseline:.4f} | {value:.4f} | {value - baseline:+.4f} | {ratio_text} |"
        )
    lines.extend(
        (
            "",
            "## Runtime and memory",
            "",
            "| Model | Elapsed seconds | Peak CUDA bytes | Peak host bytes |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| BF16 | {_number(base['elapsed_seconds'], 'base elapsed'):.2f} | "
                f"{int(_number(base['peak_device_bytes'], 'base peak device')):,} | "
                f"{int(_number(base['peak_host_bytes'], 'base peak host')):,} |"
            ),
            (
                f"| NanoQuant | {_number(frozen['elapsed_seconds'], 'frozen elapsed'):.2f} | "
                f"{int(_number(frozen['peak_device_bytes'], 'frozen peak device')):,} | "
                f"{int(_number(frozen['peak_host_bytes'], 'frozen peak host')):,} |"
            ),
            "",
            "## Provenance",
            "",
            f"- Experiment config hash: `{_markdown_cell(experiment['config_hash'])}`",
            (
                "- Launcher: `"
                + _markdown_cell(
                    cast(dict[str, object], experiment["launcher"]).get("repository_relative_path")
                )
                + "`"
            ),
            f"- Candidate identity: `{_markdown_cell(candidate['commit_identity'])}`",
            f"- Global tuning: `{_markdown_cell(candidate.get('global_tuning'))}`",
            "",
        )
    )
    return "\n".join(lines)


def execute_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> dict[str, Any]:
    raise_for_issues(validate(config, ValidationPhase.RESOLVED))
    validate_launcher_number(config, launcher_path)
    resolved = resolve_quality_evaluation_experiment(
        config,
        experiment,
        launcher_path=launcher_path,
    )
    result = execute_quality_evaluation(resolved.request)
    payload = {
        **result,
        "experiment": {
            "config_hash": config_hash(config),
            "resolved_config": to_dict(config),
            "launcher": to_dict(
                launcher_provenance(launcher_path, config.intent.experiment_number)
            ),
        },
    }
    atomic_write_json(resolved.result_path, payload)
    if resolved.markdown_path is not None:
        atomic_write_text(
            resolved.markdown_path,
            render_quality_evaluation_markdown(payload),
        )
    return payload


def run_quality_evaluation_experiment(
    config: RunConfig,
    experiment: QualityEvaluationExperiment,
    *,
    launcher_path: str | Path,
) -> int:
    execute_quality_evaluation_experiment(config, experiment, launcher_path=launcher_path)
    return 0


__all__ = [
    "QualityEvaluationExperiment",
    "execute_quality_evaluation_experiment",
    "resolve_quality_evaluation_experiment",
    "render_quality_evaluation_markdown",
    "run_quality_evaluation_experiment",
]
