"""Legacy process adapter that preserves the new audit envelope."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from nanoquant.application.service import ApplicationContext
from nanoquant.config.codec import to_dict
from nanoquant.config.schema import RunConfig


class LegacyProcessAdapter:
    def __init__(self, command: tuple[str, ...], working_directory: str | Path) -> None:
        self.command = command
        self.working_directory = Path(working_directory)

    def __call__(self, config: RunConfig, context: ApplicationContext) -> tuple[str, ...]:
        with context.artifacts.begin_write("legacy-invocation") as writer:
            (writer.path / "resolved-config.json").write_text(
                json.dumps(to_dict(config), sort_keys=True, indent=2), encoding="utf-8"
            )
            context.events.emit("legacy", "info", "process.started", command=list(self.command))
            result = subprocess.run(
                self.command, cwd=self.working_directory, text=True, capture_output=True, check=False
            )
            (writer.path / "stdout.txt").write_text(result.stdout, encoding="utf-8")
            (writer.path / "stderr.txt").write_text(result.stderr, encoding="utf-8")
            (writer.path / "exit-code.txt").write_text(str(result.returncode), encoding="ascii")
            descriptor = writer.commit()
        context.events.emit(
            "legacy",
            "info" if result.returncode == 0 else "error",
            "process.completed",
            exit_code=result.returncode,
            artifact_id=descriptor.artifact_id,
        )
        if result.returncode:
            raise RuntimeError(f"legacy process exited with {result.returncode}")
        return (descriptor.artifact_id,)
