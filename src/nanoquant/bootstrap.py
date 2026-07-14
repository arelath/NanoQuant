"""Default local composition root."""

from __future__ import annotations

import json

from nanoquant.application.report import render_run_report
from nanoquant.application.service import ApplicationContext, LegacyRunner, QuantizeApplication
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig
from nanoquant.config.validation import raise_for_issues, validate
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.environment import capture_environment
from nanoquant.infrastructure.run_session import open_run_session
from nanoquant.infrastructure.runs import (
    RunDirectory,
    initial_manifest,
    launcher_provenance,
    transition,
    validate_launcher_number,
)


def run_experiment(
    config: RunConfig, *, launcher_path: str, runner: LegacyRunner | None = None, console: bool = True
) -> int:
    raise_for_issues(validate(config))
    validate_launcher_number(config, launcher_path)
    provenance = launcher_provenance(launcher_path, config.intent.experiment_number)
    manifest = initial_manifest(config, provenance, capture_environment())
    directory = RunDirectory(config.output.run_root, manifest.run_id)
    artifacts = LocalArtifactStore(config.output.artifact_root, config.output.temporary_root)
    with open_run_session(
        directory.root,
        manifest=manifest,
        observability=config.observability,
        registry_root=directory.root.parent,
        console=console,
    ) as session:
        manifest = session.manifest
        sink = session.events
        manifest = transition(manifest, RunStatus.RUNNING)
        directory.write_manifest(manifest)
        sink.emit("run", "info", "run.started", config_hash=manifest.config_hash)
        try:
            with artifacts.begin_write("resolved-config") as writer:
                (writer.path / "config.json").write_text(
                    json.dumps(to_dict(config), sort_keys=True, indent=2), encoding="utf-8"
                )
                config_artifact = writer.commit().artifact_id
            produced = QuantizeApplication().run(config, ApplicationContext(artifacts, sink), runner)
            committed = (config_artifact, *produced)
            sink.emit("run", "info", "run.completed", artifact_count=len(committed))
            manifest = transition(manifest, RunStatus.COMPLETED, artifacts=committed)
        except KeyboardInterrupt:
            sink.emit("run", "warning", "run.interrupted")
            manifest = transition(manifest, RunStatus.INTERRUPTED)
            directory.write_manifest(manifest)
            raise
        except BaseException as exc:
            sink.emit("run", "error", "run.failed", error_type=type(exc).__name__, error=str(exc))
            manifest = transition(manifest, RunStatus.FAILED, failure={"type": type(exc).__name__, "message": str(exc)})
            directory.write_manifest(manifest)
            report = render_run_report(directory.root)
            (directory.root / "reports" / "summary.md").write_text(report, encoding="utf-8")
            raise
        directory.write_manifest(manifest)
        report = render_run_report(directory.root)
        (directory.root / "reports" / "summary.md").write_text(report, encoding="utf-8")
    return 0
