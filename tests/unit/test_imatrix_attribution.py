from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from nanoquant.config.schema import ImatrixConfig
from nanoquant.imatrix_attribution import matching_sensitivity_sweep, publish_imatrix_attribution
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from tests.support.experiments import load_experiment


def _sweep(datasets: Path, name: str, weights: str, dtype: str, trials: int, *, complete: bool = True) -> Path:
    path = datasets / name / "measurements" / "test-sweep" / "results.json"
    atomic_write_json(path, {
        "protocol": {"model_weights_sha256": weights, "teacher": {"model_dtype": dtype}, "targets": ["projection"]},
        "trials": [{}] * (trials if complete else trials - 1), "expected_trials": trials,
    })
    atomic_write_json(path.with_name("kl-response-fits.json"), {"curves": {"projection": {}}})
    return path


def test_selects_complete_sensitivity_by_weight_hash_precision_and_trials(tmp_path: Path) -> None:
    snapshot = tmp_path / "renamed-hub-snapshot"
    snapshot.mkdir()
    weights_file = snapshot / "model.safetensors"
    weights_file.write_bytes(b"same weight bytes as the teacher")
    digest = hash_file(weights_file)
    datasets = tmp_path / "Datasets"
    _sweep(datasets, "unrelated", "different", "torch.float32", 10)
    assert matching_sensitivity_sweep(datasets, snapshot) is None
    _sweep(datasets, "bf16", digest, "torch.bfloat16", 8)
    _sweep(datasets, "fp32-small", digest, "torch.float32", 2)
    selected = _sweep(datasets, "fp32-large", digest, "torch.float32", 4)
    _sweep(datasets, "unfinished", digest, "torch.float32", 10, complete=False)
    assert matching_sensitivity_sweep(datasets, snapshot) == selected


def test_publishes_reports_for_the_matching_export(tmp_path: Path, monkeypatch) -> None:
    config = load_experiment(3).config
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights_file = snapshot / "model.safetensors"
    weights_file.write_bytes(b"source weights")
    gguf = tmp_path / "Results" / "003" / "model-pre-kd.gguf"
    assert publish_imatrix_attribution(config, tmp_path, snapshot, gguf, "hash", experiment_number=3) is None
    config = replace(config, calibration=replace(config.calibration, imatrix=ImatrixConfig("matrix.gguf", "imatrix")))
    assert publish_imatrix_attribution(config, tmp_path, snapshot, gguf, "hash", experiment_number=3) is None
    sweep = _sweep(tmp_path / "Datasets", "fp32", hash_file(weights_file), "torch.float32", 2)

    def predict(command, *, cwd, check) -> None:
        assert cwd == tmp_path and check
        assert str(sweep) in command
        assert command[command.index("--gguf") + 1] == str(gguf)
        assert command[command.index("--gguf-sha256") + 1] == "hash"
        output = Path(command[command.index("--output") + 1])
        atomic_write_json(output.with_suffix(".json"), {})
        output.write_text("report", encoding="utf-8")

    monkeypatch.setattr("nanoquant.imatrix_attribution.subprocess.run", predict)
    reports = publish_imatrix_attribution(config, tmp_path, snapshot, gguf, "hash", experiment_number=3, variant=True)
    assert reports is not None
    markdown = gguf.with_name("003-model-pre-kd-imatrix-error-attribution.md")
    assert reports == (markdown, markdown.with_suffix(".json"))
    manifest = json.loads((gguf.parent / "publication.json").read_bytes())
    assert {Path(item["published"]).name for item in manifest["artifacts"]} == {path.name for path in reports}
