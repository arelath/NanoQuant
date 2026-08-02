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

## Enforced implementation safeguards

### Exact warm-start regime identity

An enabled mass-floor correction requires two explicit configuration fields:

- `expected_initializer_protocol_hash`, binding the normalized primary KD
  protocol, including objective, epoch count, batch cap, optimizer, sampling,
  and other semantic training fields;
- `expected_initializer_steps`, binding the completed optimizer-step count.

The canonical workflow compares the declared hash with the configured primary
KD protocol before primary training begins. Global distillation independently
compares both values with the persisted initializer before correction. Missing,
malformed, or mismatched values fail closed, and both expectations are part of
the correction protocol identity. This blocks accidental warm-start from the
wrong deployment artifact, but it does **not** by itself prove that the policy
was developed in that regime.

`tools/prepare_policy_initializer_transfer_receipt.py` closes that separate
methodology gap. It reads the completed development initializer's actual global-
tuning result, requires its protocol hash and observed steps to equal the
deployment launcher's primary protocol and explicit configured horizon, and
requires the correction declaration to bind those same values. It also rejects
an uncapped transferred primary, a cross-trajectory fixed final-norm value, or
reused selection/confirmation identities. The receipt explicitly limits its
claim to regime identity; held-out same-run selection and untouched confirmation
remain necessary policy evidence. This is the guard that detects the original
256-versus-2,048 development-to-deployment error.

For Experiment 048 this validator is not advisory. The immutable pre-selection
campaign receipt loads the manifest-authorized Experiment 044 primary endpoint,
runs the transfer check against the fresh Experiment 048 configuration, and
binds both the development evidence and validator bytes into its protocol hash.
Selection data cannot be opened unless that receipt succeeds.

### Deterministic numerical execution

The canonical resident and global-distillation paths seed execution inside a
scoped RNG context, require PyTorch deterministic algorithms, disable cuDNN
benchmark selection, require deterministic cuBLAS workspace configuration, and
fail if an unsupported nondeterministic kernel is encountered. Standalone
tail-aware/correction training uses the same guard and records its numerical
execution version in the report protocol.

CUDA Fisher calibration no longer uses the custom Triton linear-cross-entropy
backward path. It computes bounded 32-token vocabulary chunks through the model
head using operations covered by PyTorch deterministic enforcement. This trades
calibration time for reproducibility. The causal-calibration algorithm version
is 5 and the resident algorithm version is 52, preventing incompatible commit
adoption. A two-run tiny CUDA probe produced exact input- and output-importance
equality with zero maximum delta. A repeated pinned-Gemma preprocessing and
plan comparison remains required before the next fresh campaign.

### Permanent slice-ledger audit

`tools/validate_evaluation_slice_registry.py` audits the whole registry before
reservation or launch. It rejects duplicate identities, a `released` status,
malformed token-interval arithmetic, and every overlap among reserved and
retired intervals. The current ledger passes with 14 retired slices and two
Experiment 048 slices reserved under permanently distinct selection and
confirmation roles.

### Evidence-consistent adaptive selection

The Experiment 048 checkpoint selector consumes one calibration-only C4 curve.
It enforces exact arm and step inventories, requires aligned paired sequences,
reconstructs aggregate NLL and full KL from those sequences, rejects any
disagreement with reported means, and binds both input files by SHA-256.
Selected mass, WikiText, tasks, and later confirmation data cannot select the
checkpoint.

### Temperature-invariant capability diagnostic

The C4 evaluator reports per-token agreement with the teacher's argmax,
including aligned per-sequence values and a paired interval. A global logit
temperature cannot change this metric except at exact ties. It begins as a
reported diagnostic, not a post-hoc Experiment 048 selection gate. A binding
gate may be derived only after its behavior and uncertainty are measured on
data that will then be retired.

## Campaign checklist

Before launch:

- freeze the source factorization and all arm identities;
- declare the sole treatment variables and expected observed step counts;
- reserve non-overlapping final slices and record their token hashes;
- run the permanent slice-ledger validator;
- declare all gates, minimum meaningful deltas, interval rules, and margins;
- state which quantities are capability gates versus calibration diagnostics;
- verify that development and intended deployment settings match, or make the
  mismatch the explicit experiment axis;
- reproduce the pinned-Gemma calibration-statistics and allocation-plan hashes
  under the deterministic numerical path.

Before promotion:

- verify observed settings and immutable artifact references;
- report absolute values against same-run pre-KD, the uncorrected primary
  endpoint, and fixed retained references, not only stage marginals;
- report raw and temperature-fitted NLL/KL separately, applying comparable
  calibration to both arms before attributing a calibrated difference;
- report absolute intervals and paired intervals for every binding sampled
  statistic;
- use the six-task suite only as a 1,000-example-per-task guardrail; never tune
  or select on a 200-example mean, and use a separately powered protocol for
  any task-improvement claim;
- run the fresh WikiText gate, pinned C4 gate, and task guardrail in that order
  without retuning between them;
- integrate the surviving policy into a fresh complete compression run;
- validate resident artifacts, packed reload, GGUF export, BPW/bytes, resource
  use, and quality before changing a production default.

No failed fresh run may trigger a replacement coefficient, fold, checkpoint
rule, margin, or slice inside the same campaign.

## Deliberately deferred changes and blockers

The review's per-token mass hinge and adaptive mass-deficit budget are sensible
if a selected-mass constraint is retained. They are not the next implementation
priority because selected mass is no longer a capability gate and the always-on
tail-aware primary objective attacks the diagnosed drift directly. Revisit them
only if a concrete deployment requirement makes mass calibration load-bearing.

The allocation-statistics numerical path, pinned-Gemma repeatability gate, and
raw-versus-temperature-fitted reporting mechanism are now complete. A fresh
campaign remains blocked until its exact arms and non-overlapping slices are
declared immutably; no prior slice may be reopened for that purpose.

### Durable independent preprocessing replay

The resident workflow now has a non-semantic preprocessing boundary. When
requested, it durably commits calibration statistics, objective specifications,
the reconstruction-rank profile, and the final allocation plan, emits an
explicit interruption event, releases resident resources, and stops before the
first compression commit. Restarting the same run reuses that exact state.

`tools/replay_gemma_preprocessing_reproducibility.py` executes the two replay
arms in separate Python processes. `tools/compare_preprocessing_runs.py` then
performs a fresh hash validation of each root and every transitively referenced
artifact. The gate requires exact equality of the resident semantic-config
hash, preprocessing-state hash, calibration artifact, objective artifact,
allocation-plan artifact, and the complete reachable artifact graph. Matching
only summary statistics or plan totals is not sufficient.

The tiny CPU integration gate proves independent equality and byte-identical
resume reuse. A retained real-run self-check validates a 136-artifact graph.
The first two independent pinned-Gemma executions completed and correctly
failed the exact gate. Calibration and objective artifacts matched exactly, and
the two plan payloads differed only in their reconstruction-profile artifact
reference. All 130 rank-probe result artifacts differed solely because
`wall_seconds` was embedded in their content-addressed payloads; ranks, errors,
curves, allocation decisions, and every other plan field were identical. This
was an artifact-identity defect rather than numerical rank instability, but it
still violated the predeclared exact-hash gate.

Rank-probe schema 4 therefore keeps timing and peak workspace in observable
events while excluding both nondeterministic telemetry fields from immutable
semantic evidence. The resident algorithm version is 53, preventing reuse of
schema-3 commits. The failed receipt remains retained under
`evidence/048/pinned-gemma-preprocessing-replay`.

The clean version-53 replay at
`evidence/048/pinned-gemma-preprocessing-replay-v2` passes. Its two independent
Python processes have identical preprocessing-state hash
`sha256:fe81b786e27cc9d3994f913d8fe1f9e7420421572cf1f815e786c14752626190`,
calibration artifact `sha256-97a4d686...34de9`, objective artifact
`sha256-75cc889932...f73eb`, 130-unit reconstruction profile
`sha256-53a60bbf8...dab2`, and allocation plan
`sha256-625ebfa08...7385`. Fresh validation in both the orchestrator receipt and
a separately invoked comparison covers the same 136-artifact transitive graph.
This clears the pinned-Gemma allocation-repeatability blocker. The
raw-versus-temperature-fitted reporting path is now implemented under
`Docs/82-temperature-calibration-reporting-protocol.md`: fits are resumable,
bound to the immutable selector decision and exact retired selection slice,
and consumed only through per-arm receipts by the untouched final C4 report.
The existing raw C4 result remains the primary gate. The repository-wide gate
passes with 1,216 tests, 49 expected deselections, clean Ruff, and clean mypy.
The remaining launch preflight is the fresh campaign's immutable slice/arm
declaration.

## Experiment 043 correction

The original Experiment 043 draft named validation offset 300 as untouched,
but that interval had already accepted Experiment 040 and was subsequently
opened by Experiments 042 and 043 exploratory probes. It is permanently
retired. Experiment 043 now reserves validation offset 348, 48x512, token hash
`sha256:a1ae1fe5d43b570e7472c6a12b891e162628083c10d292a730f5360bcd79a0e6`.
The arms and gates were not changed in response to results on the new slice;
the slice was reserved before it was evaluated.
