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
    DiagnosticDefinition(
        "PERF001",
        "Insufficient profile coverage",
        "Instrument the largest unattributed parent phase until coverage reaches 90%.",
        "Docs/15-performance-profiling.md",
    ),
    DiagnosticDefinition(
        "PERF002",
        "Profiling overhead exceeded budget",
        "Disable span-event mirroring or reduce profiling granularity.",
        "Docs/15-performance-profiling.md",
    ),
    DiagnosticDefinition(
        "PERF003",
        "CUDA timing records were unresolved",
        "Inspect the device state and increase the deferred timing buffer.",
        "Docs/15-performance-profiling.md",
    ),
    DiagnosticDefinition(
        "NQ-CAL-001",
        "Non-finite calibration statistic",
        "Replay calibration with per-partition statistic and input-range capture.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-CAL-003",
        "Calibration partition instability",
        "Compare partition token diversity, clipping, and sample-order sensitivity.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-HES-001",
        "Poor Hessian conditioning",
        "Inspect eigenvalue range and objective fallback choices.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-FAC-001",
        "ADMM residual plateau",
        "Inspect primal/dual residual traces and rho/iteration sensitivity.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-FAC-002",
        "Latent-to-export error gap",
        "Compare latent, sign-export, scale-fit, and packed-reference errors.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-RNK-002",
        "Ineffective rank retry",
        "Compare added-bit utility with neighboring layers and the global budget.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-RNK-003",
        "Ineffective outlier allocation",
        "Compare block loss with outliers disabled and reallocate their bit cost.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-TUN-002",
        "Poor tuning recovery",
        "Inspect the tuning trajectory, best-state restore, and block targets.",
        "Docs/07-observability-and-reporting.md",
    ),
    DiagnosticDefinition(
        "NQ-INF-001",
        "Unexpected runtime fallback",
        "Inspect capability rejection codes and packed layout compatibility.",
        "Docs/07-observability-and-reporting.md",
    ),
):
    register(_definition)
