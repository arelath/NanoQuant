# Experiment 048: Adaptive Capability-Correction Policy

## Status

Paused after implementation preflight. The selector and its retired-evidence
replay are implemented and tested, but no fresh slice is reserved and no model
evaluation is authorized. The independent Experiment 042 review identified
additional methodology gates in
[Document 81](../81-experiment-methodology-guardrails.md). Allocation
reproducibility, exact initializer-regime binding, calibration/capability
separation, and the revised reporting protocol must be complete before this
experiment can become predeclared.

## Motivation

Experiment 047 validates correction epoch 3 only on the retained Experiment
044 trajectory. Experiments 040 and 042 already showed why copying a fixed
checkpoint or fold to a new trajectory is unsafe. Experiment 048 converts the
useful mechanism into an adaptive checkpoint policy while preserving a
baseline escape path.

This is not adaptive hyperparameter search. The correction objective and
maximum work are one treatment under test; only the checkpoint is selected by
a frozen rule on calibration-only data.

## Candidate policy

Starting from a fresh, fully materialized 256-step tail-aware endpoint:

1. retain the uncorrected endpoint as checkpoint 0;
2. run one-sided correction with weight 2.0, teacher-mass ratio 0.8, learning
   rate `1e-5`, four epochs, at most 32 batches per epoch, and a 128-step cosine
   horizon;
3. durably retain checkpoints at 32, 64, 96, and 128 observed correction
   steps;
4. finish all four checkpoints before evaluating the calibration slice; no
   development metric may early-stop training;
5. evaluate checkpoint 0 and all four correction checkpoints together on one
   calibration-only C4 slice with aligned per-sequence NLL and full-vocabulary
   KL;
6. choose at most one checkpoint using the frozen rule below.

The coefficient, ratio, learning rate, epoch cap, and scheduler are explicit
transfer variables in the fresh campaign. A campaign pass validates their
combination only through the adaptive policy; it does not establish each as a
universal default.

## Frozen selection rule

For each correction checkpoint, compute candidate-minus-checkpoint-0 paired
95% bootstrap intervals using 10,000 task-independent sequence resamples and
seed 0.

1. A checkpoint is eligible only when NLL and full-KL point deltas are negative
   and both paired upper bounds are below zero.
2. Among eligible checkpoints, compute the eligible minima for NLL and full
   KL. Keep checkpoints no more than 0.01 nats above both minima.
3. Select the earliest remaining checkpoint.
4. If no checkpoint is eligible or no joint plateau exists, select checkpoint
   0 and reject correction for that trajectory.

The 0.01 tolerance is a future policy choice informed by the completed
Experiments 046-047 development history. It is not retroactively claimed as a
rule for those experiments. It must be frozen before any Experiment 048 slice
is opened.

Selected top-k mass, training loss, block NRMSE, the ordinary WikiText
benchmark, and tasks are reported diagnostics or later gates. None may choose
the checkpoint. In particular, mass is not a capability gate.

## Required implementation before reservation

- a selector that consumes one C4 curve and its sequence checkpoint, binds
  both inputs by SHA-256, enforces exact arm/step inventories, applies only the
  two metrics above, and emits an immutable decision receipt;
- unit tests for ineligible arms, no-survivor fallback, earliest joint plateau,
  step mismatch, protocol mismatch, and input-hash binding;
- a materializer that proves the selected durable checkpoint and normal
  global-tuning reload have exact equality for every selected tensor;
- a campaign receipt binding the fresh factorization identity, 256-step
  tail-aware initializer, correction protocol, calibration reservation, and
  selector version;
- repository-wide pytest, Ruff, and mypy success before launch.

The rejected post-hoc Experiment 046 selector that mixed C4 NLL/KL with
WikiText tail KL is not an implementation starting point. Metric roles and
distributions may not be recombined after results are visible.

## Validation sequence

Before a fresh complete compression run, replay the selector mechanically on
the already-retired Experiment 046 C4 curve. This is implementation evidence
only: it should choose epoch 3, whose separate Experiment 047 result is already
known. It cannot tune the tolerance or count as a new gate.

The fresh campaign then uses two new, non-overlapping C4 ranges:

1. calibration-only selection slice;
2. untouched final confirmation slice, opened only after the decision receipt
   is frozen.

The final C4 candidate must improve or non-regress on paired NLL and full KL
under predeclared bounds. No retuning follows. It then runs the ordinary
64x128 WikiText deployment benchmark and the 1,000-example six-task guardrail.

## Complete-run requirement

The policy is not accepted from retained checkpoints alone. Its final test is
one fresh complete `execute_complete_compression` campaign with:

- one fresh Gemma factorization and exact 256-step tail-aware initializer;
- baseline and correction branches sharing that frozen identity;
- strict resident validation before model-level evaluation;
- immutable selected global-tuning artifact and exact reload audit;
- unchanged rank, represented factor bytes, and effective BPW;
- logical and packed reload quality;
- GGUF export and GGUF quality;
- C4 confirmation, WikiText PPL, and 1,000-task comparisons;
- wall time, peak GPU/host memory, artifact bytes, packed bytes, and GGUF bytes.

A fresh-run failure keeps the uncorrected tail-aware baseline and rejects this
policy. It does not trigger another coefficient, tolerance, epoch, or slice
search within the same campaign.

## Implementation preflight result

`tools/select_c4_capability_correction_checkpoint.py`, frozen in commit
`e7558f2`, implements the two-metric rule. Its input contract binds the complete ordered arm inventory,
baseline and checkpoint step counts, checkpoint identities, common protocol,
aligned sequence inventories, input-file SHA-256 hashes, tolerance, bootstrap
count, and seed. It emits the selected immutable arm identity and falls back
to the uncorrected baseline when there is no eligible joint plateau.

Focused tests cover invalid arm syntax, exact step identities, protocol and
inventory mismatch, aggregate-versus-sequence disagreement, no-survivor
fallback, disjoint eligible minima, earliest joint-plateau selection, and both
input hashes. Eight tests pass; focused Ruff and the repository-standard mypy
target also pass.

The authorized mechanical replay on the retired Experiment 046 C4 curve used
the exact future policy values: tolerance `0.01`, 10,000 paired resamples, seed
0, baseline 256 steps, and correction checkpoints at 32, 64, 96, and 128
steps. It produced:

- eligible: correction epochs 2, 3, and 4;
- joint plateau: correction epochs 3 and 4;
- decision: correction epoch 3, the earliest joint-plateau arm;
- selected immutable checkpoint:
  `sha256-d885f823e0651314d78f1c4c7b98edd66470344bb1957f5e27aa959ac0624a8d`.

The decision receipt is
`evidence/048/experiment048-retired046-selector-replay-v2.json`. It was created
only after the selector implementation was committed. This result is
implementation evidence only. The data were already opened, the resulting
epoch-3 hypothesis is already known, and the replay consumes no new slice. The
receipt binds quality SHA-256
`e7f9c6813164449ec127a3a90ca41ce2a34ae9feee7164bc86705fd435d92f14`
and sequence-checkpoint SHA-256
`5bb9f5327b48c631eaab324034aa14d8ca2a271a2c0ce3b2fd5908ec50112263`.

## Superseding constraints from the Experiment 042 review

The following constraints supersede any earlier implication that selector
preflight alone authorizes a fresh campaign:

- the 256-step `top_k_tail` primary protocol must be bound by its exact
  protocol hash and completed-step count before correction can run;
- the corrected deterministic calibration path must reproduce identical
  calibration-statistics and plan hashes on a pinned-Gemma replay, or the
  fresh-factorization comparison remains confounded;
- selected mass is reported as calibration, not used as a capability gate;
- C4 reports teacher-top-1 agreement alongside NLL and full KL, but it does not
  retroactively enter the frozen checkpoint rule;
- raw and temperature-fitted results must be separated in the final report;
- task evaluation is a 1,000-example guardrail, never a 200-example selection
  signal;
- the permanent slice-registry validator must pass before reservation and
  launch;
- absolute results against pre-KD, uncorrected tail-aware, and fixed retained
  references accompany every marginal comparison.

The next action is therefore not slice reservation. It is the deterministic
calibration-statistics/plan replay and the calibration-versus-capability report
implementation required by Document 81.

## Preprocessing replay mechanism

The replay mechanism is now implemented and fixture-tested. A resident run can
stop immediately after durable preprocessing without changing its semantic
configuration identity. Two independent fixture runs are required to
match the preprocessing-state hash, calibration, objectives, allocation plan,
and every transitively referenced artifact before resume. The resumed fixture
reuses the same state byte-for-byte and reaches compression normally.

The pinned replay command is bound to the Experiment 044 launcher, which is the
fresh factorization recipe underlying this campaign:

```powershell
.\.venv\Scripts\python.exe tools\replay_gemma_preprocessing_reproducibility.py `
  --launcher experiments\044-tail-aware-256-d2-compress-and-benchmark-gemma-3-1b-it.py `
  --output-root evidence\048\pinned-gemma-preprocessing-replay-v2
```

The orchestrator launches `run-a` and `run-b` in distinct Python processes and
writes `preprocessing-reproducibility.json` only after complete transitive hash
validation. No evaluation slice is reserved or opened by this operation. Until
that real receipt passes, Experiment 048 remains paused.

The initial real replay at `evidence/048/pinned-gemma-preprocessing-replay`
failed as intended. Calibration input, calibration statistics, objectives, and
the semantic configuration were exact. The plans differed only by their
reconstruction-profile reference, and all 130 profile members differed only in
recorded wall time. Telemetry had been included in content-addressed semantic
evidence. Rank-probe schema 4 removes wall time and peak workspace from the
artifact payload while retaining both in events; resident algorithm version 53
prevents adoption of the old evidence. The failed receipt is retained, and the
version-53 replay uses the new `-v2` output rather than overwriting it.

The version-53 replay passes. Run A and run B independently produce exact
identity for the calibration input, Fisher statistics, objectives, all 130
rank-probe results, reconstruction profile, final allocation plan, resident
semantic configuration, preprocessing-state file, and the complete reachable
136-artifact graph. The authoritative receipt is
`evidence/048/pinned-gemma-preprocessing-replay-v2/preprocessing-reproducibility.json`;
`independent-preprocessing-comparison.json` records a second fresh validation.
No evaluation slice was opened. Allocation reproducibility is no longer an
Experiment 048 blocker.

The calibration-versus-capability reporting mechanism is also implemented.
After a fresh selector decision, `tools/fit_non_wikitext_temperature.py` fits
each primary arm independently on that exact retired selection slice and emits
an identity-bound resumable receipt. The final C4 evaluator accepts only the
baseline and selected-arm receipts from one shared fit protocol, preserves raw
NLL/KL as the promotion gate, and reports calibrated NLL/KL and top-k-mass
diagnostics separately. It rejects a receipt whose arm, checkpoint, frozen
model, selector evidence, or token role differs. No fresh Experiment 048 slice
has yet been reserved or opened.
