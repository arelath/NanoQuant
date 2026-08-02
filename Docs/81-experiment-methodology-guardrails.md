# Experiment Methodology Guardrails

## Purpose

The independent Experiment 042 review found a methodological transfer failure,
not a numerical implementation defect. A correction developed after 256 KD
steps was deployed after 2,048 steps, fixed correction and calibration values
were transferred across a much larger deficit, the binding margin was far
smaller than measured variation, and previously opened WikiText slices were
reused. These controls make those failure modes explicit and mechanically
checkable.

## Mandatory controls

### Deployment regime is part of the treatment

Every experiment must record both configured and observed optimizer steps,
batches per epoch, epochs, scheduler horizon, learning rate, objective, and
initializer identity. A production candidate may only inherit a policy from a
development experiment when these values match. If they do not match, the
policy is unvalidated in the new regime even if its checkpoint replay is
byte-exact.

Matched objective comparisons must enforce the same observed step count for
all objective arms. Horizon comparisons must name the differing horizons as
the sole treatment. The Experiment 043 evaluator requires an expected step
count for every arm and rejects a mismatched artifact.

### Point recipes do not transfer without a new calibration

Fixed correction budgets, coefficients, stopping epochs, and final-norm folds
are valid only for the trajectory on which they were selected. A new
factorization, initializer, training horizon, objective, or calibration
distribution invalidates that transfer unless a fresh held-out comparison
proves it.

If a selected-mass requirement remains, correction and calibration must be
adaptive to the observed deficit. Calibration chooses the smallest value that
clears a target plus a predeclared safety margin on calibration-only data. A
gate slice may not participate in that choice.

### Margins and uncertainty are mandatory for binding statistics

All sampled gate metrics, including selected top-k mass, require sequence-level
uncertainty intervals. A threshold gate passes only when the relevant interval
bound clears the threshold; a scalar point estimate is insufficient. Safety
margins must be sized from retained cross-slice or cross-run variation and
declared before the gate opens.

Selected mass is a calibration diagnostic, not a capability proxy, until a
downstream behavior it protects is identified and measured. NLL, full KL,
task quality, and the pinned non-WikiText gate retain priority. A future mass
floor must either be re-derived from a downstream requirement or treated as a
deployment calibration constraint.

### Evaluation slices never become fresh again

The machine-readable registry is
[`evaluation-slice-registry.json`](evaluation-slice-registry.json). Slices are
tracked by dataset, split, tokenizer-defined raw-token interval, sequence
length, offset, sample count, and token hash. Any interval used for fitting,
monitoring, selection, screening, or confirmation is permanently retired.
Rejection of the experiment does not release it.

New gates must be reserved in the registry before model evaluation. The gate
tool verifies the exact token hash and rejects overlap with every reserved or
retired interval. Once opened, the reservation is changed to `retired`, whether
the candidate passes, fails, or the run is interrupted after producing any
model-dependent metric.

### Distribution and role separation

Training constraints, calibration, development monitors, WikiText gates, C4
gates, and task benchmarks have distinct roles. A result on one distribution
does not establish the same constraint on another. Reusing a training batch as
a correction monitor is allowed only as an explicitly labeled optimization
diagnostic; it cannot select or accept a production policy.

Development monitors may diagnose trajectories but may not early-stop an arm
unless early stopping and its independent data are predeclared. Final gates
remain unopened until every coefficient, checkpoint, step horizon, and
calibration rule is frozen.

### Reproducibility evidence is not policy evidence

Resume identity, byte-exact checkpoint replay, scheduler restoration, and
artifact validation prove implementation reproducibility. They do not prove
that a recipe transfers to a new initializer or regime. Reports must state
which claim each piece of evidence supports.

## Campaign checklist

Before launch:

- freeze the source factorization and all arm identities;
- declare the sole treatment variables and expected observed step counts;
- reserve non-overlapping final slices and record their token hashes;
- declare all gates, minimum meaningful deltas, interval rules, and margins;
- state which quantities are capability gates versus calibration diagnostics;
- verify that development and intended deployment settings match, or make the
  mismatch the explicit experiment axis.

Before promotion:

- verify observed settings and immutable artifact references;
- report absolute intervals and paired intervals for every binding sampled
  statistic;
- run the fresh WikiText gate, pinned C4 gate, and full task guardrail in that
  order without retuning between them;
- integrate the surviving policy into a fresh complete compression run;
- validate resident artifacts, packed reload, GGUF export, and quality before
  changing a production default.

## Experiment 043 correction

The original Experiment 043 draft named validation offset 300 as untouched,
but that interval had already accepted Experiment 040 and was subsequently
opened by Experiments 042 and 043 exploratory probes. It is permanently
retired. Experiment 043 now reserves validation offset 348, 48x512, token hash
`sha256:a1ae1fe5d43b570e7472c6a12b891e162628083c10d292a730f5360bcd79a0e6`.
The arms and gates were not changed in response to results on the new slice;
the slice was reserved before it was evaluated.
