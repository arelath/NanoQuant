"""Find fitted sensitivity sweeps and publish attribution for completed GGUFs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from nanoquant.config.schema import RunConfig
from nanoquant.infrastructure.io_utils import hash_file
from nanoquant.infrastructure.publication import (
    PublishableArtifact,
    PublishableArtifactKind,
    publish_experiment_artifacts,
    publish_run_artifacts,
)


def sensitivity_weight_identity(snapshot: Path) -> str | dict[str, str]:
    """Use the same weight identity as the sensitivity sweep protocol."""
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        return hash_file(snapshot / "model.safetensors")
    shards = set(json.loads(index.read_bytes())["weight_map"].values())
    if any(Path(name).name != name or not name.endswith(".safetensors") for name in shards):
        raise ValueError("invalid checkpoint shard name in sensitivity weight identity")
    return {name: hash_file(snapshot / name) for name in sorted(shards)}


def matching_sensitivity_sweep(datasets: Path, snapshot: Path) -> Path | None:
    """Select a complete fitted sweep by checkpoint bytes, precision and trial count."""
    weights = None
    candidates: list[tuple[int, int, Path]] = []
    for sweep in sorted(datasets.glob("*/measurements/*/results.json")):
        fits = sweep.with_name("kl-response-fits.json")
        if not fits.is_file():
            continue
        measurement = json.loads(sweep.read_bytes())
        protocol = measurement["protocol"]
        if weights is None:
            weights = sensitivity_weight_identity(snapshot)
        if protocol["model_weights_sha256"] != weights:
            continue
        if len(measurement["trials"]) != measurement["expected_trials"]:
            continue
        fit = json.loads(fits.read_bytes())
        if set(fit["curves"]) != set(protocol["targets"]):
            continue
        precision = {"torch.float32": 2, "torch.float16": 1, "torch.bfloat16": 0}
        candidates.append((precision.get(protocol["teacher"]["model_dtype"], -1),
                           measurement["expected_trials"], sweep))
    return max(candidates)[2] if candidates else None


def publish_imatrix_attribution(
    config: RunConfig,
    repository_root: Path,
    snapshot: Path,
    gguf: Path,
    gguf_sha256: str,
    *,
    experiment_number: int | None,
    variant: bool = False,
) -> tuple[Path, Path] | None:
    """Analyze a validated GGUF on CPU when its calibrated checkpoint has fitted curves."""
    imatrix = config.calibration.imatrix
    if imatrix is None:
        return None
    sweep = matching_sensitivity_sweep(repository_root / "Datasets", snapshot)
    if sweep is None:
        return None
    prefix = "" if experiment_number is None else f"{experiment_number:03d}-"
    if variant:
        prefix += f"{gguf.stem}-"
    markdown = gguf.with_name(f"{prefix}imatrix-error-attribution.md")
    statistics = markdown.with_suffix(".json")
    command = [
        sys.executable, str(repository_root / "tools/predict_imatrix_weighted_kl.py"), str(sweep),
        "--predict-only", "--snapshot", str(snapshot), "--imatrix", str(repository_root / imatrix.path),
        "--imatrix-sha256", imatrix.sha256, "--gguf", str(gguf), "--gguf-sha256", gguf_sha256,
        "--output", str(markdown),
    ]
    for item in imatrix.tensor_map:
        command.extend(("--imatrix-tensor", item.layer, item.tensor))
    if imatrix.use_rows:
        command.append("--imatrix-use-rows")
    subprocess.run(command, cwd=repository_root, check=True)
    artifacts = (
        PublishableArtifact(markdown, PublishableArtifactKind.REPORT),
        PublishableArtifact(statistics, PublishableArtifactKind.STATISTICS),
    )
    if experiment_number is None:
        publish_run_artifacts(repository_root, config.intent.name, artifacts)
    else:
        publish_experiment_artifacts(repository_root, experiment_number, artifacts)
    return markdown, statistics
