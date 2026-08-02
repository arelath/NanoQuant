# Experiment 048: Adaptive Capability-Correction Policy

## Status

Design only. No slice is reserved and no model evaluation is authorized. The
selector, tests, complete command line, slice hashes, and gate rule must be
committed before this experiment can become predeclared.

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
