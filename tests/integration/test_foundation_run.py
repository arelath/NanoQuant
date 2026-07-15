import json
from pathlib import Path

import pytest

from nanoquant.bootstrap import run_experiment
from nanoquant.config.schema import IntentConfig, ModelConfig, OutputConfig, RunConfig


def test_numbered_application_run_produces_self_contained_audit_envelope(tmp_path: Path) -> None:
    launcher = tmp_path / "019_foundation_smoke.py"
    launcher.write_text("# zero-argument launcher fixture\n", encoding="utf-8")
    config = RunConfig(
        ModelConfig("local/tiny", revision="source-revision", tokenizer_revision="tokenizer-revision"),
        intent=IntentConfig(experiment_number=19, name="foundation-smoke", purpose="verify audit envelope"),
        output=OutputConfig(
            run_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
        ),
    )

    def legacy_smoke(_config: RunConfig, context: object) -> tuple[str, ...]:
        with context.artifacts.begin_write("legacy-smoke") as writer:  # type: ignore[attr-defined]
            (writer.path / "result.json").write_text('{"ok":true}', encoding="utf-8")
            return (writer.commit().artifact_id,)

    assert run_experiment(config, launcher_path=str(launcher), runner=legacy_smoke, console=False) == 0
    directories = list((tmp_path / "runs").glob("run_*"))
    assert len(directories) == 1
    manifest = json.loads((directories[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["launcher"]["experiment_number"] == 19
    assert len(manifest["artifacts"]) == 2
    assert (directories[0] / "events.jsonl").is_file()
    assert (directories[0] / "run.log").is_file()
    assert "run.completed" in (directories[0] / "run.log").read_text(encoding="utf-8")
    report = (directories[0] / "reports" / "summary.md").read_text(encoding="utf-8")
    assert "foundation-smoke" in report and "Status: `completed`" in report
    assert "## Outcome" in report
    assert "Conclusion: Run completed successfully." in report
    assert "Recommended next action:" in report
    assert "## Launcher" in report
    assert "Zero-argument numbered runfile: **yes**" in report
    assert manifest["launcher"]["content_hash"] in report
    assert "## Environment" in report
    assert "## Cost" in report


def test_missing_pipeline_is_recorded_as_failure_not_success(tmp_path: Path) -> None:
    launcher = tmp_path / "020_new_experiment.py"
    launcher.write_text("# zero-argument launcher fixture\n", encoding="utf-8")
    config = RunConfig(
        ModelConfig(
            "local/tiny",
            revision="source-revision",
            tokenizer_revision="tokenizer-revision",
        ),
        intent=IntentConfig(experiment_number=20, name="missing-pipeline"),
        output=OutputConfig(
            run_root=str(tmp_path / "runs"),
            artifact_root=str(tmp_path / "artifacts"),
        ),
    )

    with pytest.raises(RuntimeError, match="RUN002.*run_quantization_experiment"):
        run_experiment(config, launcher_path=str(launcher), console=False)

    directories = list((tmp_path / "runs").glob("run_*"))
    assert len(directories) == 1
    manifest = json.loads((directories[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["failure"] == {
        "message": (
            "RUN002 no executable pipeline was supplied; new experiments must use "
            "run_quantization_experiment"
        ),
        "type": "RuntimeError",
    }
    events = (directories[0] / "events.jsonl").read_text(encoding="utf-8")
    assert "pipeline.not_configured" in events
    assert "run.failed" in events
