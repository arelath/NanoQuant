"""Central stable diagnostics registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DiagnosticDefinition:
    code: str
    title: str
    remediation: str
    documentation: str


REGISTRY: dict[str, DiagnosticDefinition] = {}


def register(definition: DiagnosticDefinition) -> None:
    if definition.code in REGISTRY:
        raise ValueError(f"diagnostic code already registered: {definition.code}")
    REGISTRY[definition.code] = definition


def get(code: str) -> DiagnosticDefinition:
    try:
        return REGISTRY[code]
    except KeyError as exc:
        raise KeyError(f"unregistered diagnostic code: {code}") from exc


for _definition in (
    DiagnosticDefinition(
        "ART001", "Artifact corruption", "Restore or recompute the artifact.", "Docs/10-artifacts-and-compatibility.md"
    ),
    DiagnosticDefinition(
        "CFG001",
        "Unsupported schema",
        "Migrate the recipe to a supported schema.",
        "Docs/03-configuration-reference.md",
    ),
    DiagnosticDefinition(
        "CAL004",
        "Unsupported calibration mode",
        "Use a productized mode or install/select the versioned research component.",
        "Docs/adr/0007-calibration-and-objective-support.md",
    ),
    DiagnosticDefinition(
        "RUN001",
        "Active run lease",
        "Wait for the process or explicitly fork the run.",
        "Docs/03-configuration-and-runs.md",
    ),
    DiagnosticDefinition(
        "SRC001",
        "Unsupported model variant",
        "Select a registered adapter/checkpoint variant.",
        "Docs/02-architecture.md",
    ),
):
    register(_definition)
