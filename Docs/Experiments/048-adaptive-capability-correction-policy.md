# Experiment 048: Adaptive Capability-Correction Policy

## Status

Paused after immutable slice reservation. Allocation reproducibility, exact
initializer-regime binding, calibration/capability separation, selected-
checkpoint reload equality, and the campaign-receipt implementation now pass
their preflight gates. Two fresh slices are reserved, but neither has been
opened and no model evaluation has run. The numbered launcher deliberately
fails if invoked directly; the adaptive orchestrator is the only authorized
entry point because it enforces selection, fallback, confirmation, and final
quality ordering.

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

## Selected-checkpoint materialization preflight

The selected-checkpoint materializer now treats reload equality as a binding
gate rather than assuming that freezing a checkpoint produces the same model.
It reloads the newly committed normal global-tuning artifact through the
factorized model loader and requires exact name, shape, dtype, and value
equality for every selected parameter. Its schema-2 receipt records the full
per-parameter content-hash inventory and equal checkpoint/reload inventory
hashes. Any missing, unexpected, or changed tensor aborts the atomic derived
run.

The derived run also receives a distinct run identity, records the source run
as its parent, names `distillation-checkpoint-materialization` as its fork
boundary, and adds the selected global-tuning artifact to the completed
manifest. This makes the selected pointer eligible for the normal completed
workflow and export loaders instead of leaving a source manifest that only
authorized the superseded active checkpoint.

An analysis-only replay of the already-known Experiment 046 epoch-3 checkpoint
is retained at
`evidence/048/experiment048-retired046-epoch3-materialization-v3`. It proves
exact equality for 677 parameters and 2,069,760 elements; both inventories hash
to `sha256:f362da60f4aa087eceeb95c0339a089dfbf4fe7885f4d86be1ede51de47c6fea`.
Fresh resident validation independently passes all 26 blocks, 130 owners, and
713 transitive artifacts. This replay opens no data and is implementation
evidence only. The materializer requirement is now cleared; the adaptive
orchestrator and immutable arm/slice declaration remain before reservation.

## Frozen fresh-campaign declaration

The configuration-only launcher is
`experiments/048-adaptive-capability-correction-d2-gemma-3-1b-it.py`. It binds
one fresh factorization to the exact Experiment 044 primary regime:

- `top_k_tail`, tail weight 0.5;
- eight epochs capped at 32 batches, exactly 256 expected steps;
- primary protocol hash
  `sha256:0ed7993a02eb980403ebeb97ff2d2cbf738242e64e6a7d07ad9f2900ef611936`;
- one correction trajectory with weight 2.0, mass ratio 0.8, learning rate
  `1e-5`, four 32-step epochs, and a fixed 128-step scheduler horizon;
- no fixed final-norm fold and no foldable-MLP continuation.

The only configuration differences from Experiment 044 are enabling the
correction, binding its initializer protocol/step identity, extending it to
four complete epochs, and the numbered experiment metadata/output paths. The
launcher cannot accidentally execute the ordinary epoch-4 export path: its
entry point remains fail-closed until the adaptive campaign orchestrator is
available.

Canonical correction checkpoints use the
`global-distillation-mass-floor` state namespace. Checkpoint discovery,
WikiText/C4 evaluation, temperature fitting, and selected-checkpoint
materialization now carry that namespace explicitly; an Experiment 042 real
artifact check resolves its canonical epoch-1 checkpoint at exactly 32 steps.

`tools/prepare_experiment048_campaign_receipt.py` is the pre-selection
fail-closed receipt builder. After a fresh resident run, but before the C4
selection slice is opened, it requires:

- a completed Experiment 048 manifest whose launcher bytes and canonical
  configuration match the declaration above;
- fresh strict validation with all 26 blocks and 130 owners;
- the exact 256-step primary artifact and active 128-step correction endpoint;
- all four correction checkpoints at 32, 64, 96, and 128 steps, sharing one
  protocol, source-block identity, and the exact primary initializer;
- two disjoint 48x512 C4 reservations with permanently distinct selection and
  final-confirmation roles;
- the frozen selector rule, tolerance 0.01, 10,000 resamples, seed 0, and hashes
  of every evaluator, fitter, materializer, registry, launcher, and protocol
  document that will consume the run.

The receipt is immutable and may only reach
`ready_for_selection_evaluation` when every binding agrees. The remaining
preflight work is the adaptive orchestrator plus choosing, hashing, and
reserving the two declared C4 intervals. No data were opened by this
declaration work.

## Fresh-data and deployment safeguards

Fresh slices now have an irreversible, receipt-authorized opening step.
`tools/open_experiment048_c4_slice.py` holds an exclusive registry lock,
verifies the complete registry snapshot and campaign protocol, and atomically
changes exactly one authorized slice from `reserved` to `retired` before any
model evaluation. The transition records the campaign receipt and protocol
hashes, is idempotent for the same authority, and rejects unrelated registry
changes. The C4 evaluator accepts `retired` slices by default and no longer
evaluates a merely reserved slice. Thus an interruption after the first model
forward cannot return opened data to the unused pool.

Final reporting also distinguishes causal arms from historical references.
The uncorrected baseline and selected checkpoint must have the same model,
configuration, and allocation-plan hashes. Explicitly declared pre-KD or
retained reference arms may have different factorization identities, but they
are excluded from the primary promotion pair and are emitted only as absolute
candidate-versus-reference comparisons. This keeps the marginal correction
claim same-run while satisfying the requirement to show absolute context.

The numbered workflow now fixes the six-task guardrail at 1,000 examples. Its
complete-compression options can explicitly load a relocated, exactly audited
selected-checkpoint run, allowing the final logical, packed, GGUF, and quality
path to consume the materialized winner instead of silently exporting the
last correction epoch. These safeguards still do not authorize a fresh data
reservation or CUDA launch; orchestration and a complete dry run remain.

The materialization boundary also accepts the canonical resident endpoint as
its metadata authority; it no longer assumes a standalone analysis probe left
`report.json`. The campaign receipt binds the primary epoch-8 checkpoint at
exactly 256 steps as the uncorrected fallback, alongside correction epochs
1-4. Therefore both selector outcomes are materializable: a winning correction
uses its exact correction checkpoint, while a no-survivor decision deploys the
same-run primary endpoint rather than the active 128-step correction endpoint.

## Adaptive campaign orchestrator

`tools/run_experiment048_campaign.py` now implements the complete resumable
stage graph. Its dry-run plan fixes the order before data are opened:

1. fresh resident factorization, exact 256-step primary, and the complete
   32/64/96/128-step correction curve;
2. strict resident validation and immutable campaign receipt;
3. irreversible selection-slice retirement, one same-run five-arm C4 curve,
   and the frozen selector;
4. exact selected-checkpoint (or primary fallback) materialization and strict
   validation;
5. for a selected correction only, identity-bound baseline/candidate
   temperature fits on the retired selection slice, followed by irreversible
   confirmation-slice retirement and raw/fitted final C4 reporting;
6. mandatory `execute_complete_compression`, logical/packed/GGUF artifacts,
   WikiText quality, and the 1,000-example six-task guardrail on the deployed
   derived run.

The campaign receipt also freezes the two absolute historical references:
accepted Experiment 040 at 32 correction steps and tail-aware Experiment 044
at 256 primary steps. It records each manifest hash, active tuning artifact,
protocol hash, source-block inventory, and whether the historical manifest
explicitly authorized that tuning artifact. The latter is false for the old
Experiment 040 derived-manifest schema and is reported rather than hidden;
the reference is read-only and its artifact is freshly validated.

The baseline arm in selection is loaded through the explicit primary tuning
pointer, not the active correction pointer. The selector permits `tuning`
mode only for that baseline; all correction arms must remain exact checkpoint
loads. Final selected-arm evaluation likewise uses the checkpoint identity
that the decision and temperature-fit receipts bind. Exact materialization
then proves that the deployment artifact has the same tensors.

Failure paths are predeclared. A no-survivor selector result deploys the bound
primary fallback without opening confirmation. A selected correction that
fails untouched C4 confirmation is retained as failed evidence, after which
the orchestrator materializes and fully benchmarks the same-run primary
fallback. Neither path changes a coefficient, tolerance, checkpoint rule, or
slice. The orchestrator remains unlaunched. Its two fresh C4 ranges are now
reserved but have not been opened or evaluated:

- selection: offset 344, 48x512, token hash
  `sha256:2ca230bf679af1c2147744c1141a3eaf04f61616666c46c09b3ec687c55a70fd`;
- confirmation: offset 392, 48x512, token hash
  `sha256:6c62bd172303e33b6a6a0847dbaaf587df25852fa9f0bc56269bfd7d3f5e6f1d`.

The reservation intent is
`evidence/048/experiment048-c4-reservation-intent.json`, protocol hash
`sha256:ca19d3e885ad50753f58e1a1f77da867fda13af68b64a92d8bd314e084dcaedc`.
It binds C4 Arrow SHA-256
`b815c9d0f42d47d67710ccd6c56efe0876633ce265acda03b5fd96424b0c2556`,
dataset fingerprint `e2b0d3b96d5472e5`, BOS token 2, and the complete
pre-mutation registry snapshot. The ledger now validates with 14 retired and
two reserved slices.

The local C4 Arrow file is now a required campaign input rather than an
unbound loader default. Its absolute path and SHA-256 are frozen in the
campaign receipt and the same path is propagated to reservation, selection,
temperature fitting, and confirmation. `tools/reserve_experiment048_c4_slices.py`
tokenizes both proposed ranges through the evaluator's exact tokenizer/window
function, writes an immutable pre-mutation intent containing the original
registry snapshot and both token hashes, and then appends both reservations
under the same exclusive ledger lock used for retirement. It is resumable
across an interruption and rejects overlap, identity reuse, or any registry
change outside that intent. The implementation passes the full repository
gate. It has now been invoked exactly once to create the reservations above;
no model was loaded and neither reservation has transitioned to `retired`.
