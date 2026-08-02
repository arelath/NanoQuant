# Experiment 048 Paused Handoff

## Status

Experiment 048 was explicitly stopped at the user's request on 2026-08-02 at
approximately 07:24 America/Los_Angeles. Several already-queued block-worker
commands subsequently started, so each exact process chain was terminated
child-first. A durable `evidence/048/PAUSED` sentinel now makes every campaign
worker exit before loading the experiment; it must be removed explicitly to
resume. No Experiment 048 process remains and CUDA memory returned to
display-only use. No evidence was deleted, moved, or rewritten.

The manifest still says `running` because the process was externally stopped
between durable commits. That status is stale and is not evidence that a worker
still owns the run. Process inspection and the progress journal are authoritative
for this pause.

## Durable resume point

- Run output:
  `evidence/048/048-adaptive-capability-correction-d2-gemma-3-1b-it`
- Run ID: `run_20260802T124929148668_0495cd2c`
- Semantic identity:
  - config: `sha256:c286fca323010d8621f942e024216d38e260558fe8807a835148155c91b8a8e7`
  - model: `sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130`
  - plan: `sha256-625ebfa08b94725da35fbc050467c63c5437319a1cd1252026bbc5743f957385`
- Durable contiguous block prefix: blocks 0 through 7.
- Last durable block artifact:
  `sha256-476301a44fdb4eab926ce1537cd52b613520ede13b2b893c12b84c6f21e3e401`
- Journal: 50 active records, one identity, no inactive records.
- Durable logical layers: 40 across eight block records.
- Fresh partial validation: 320 transitive artifacts, 5,106,240,944
  artifact bytes, effective partial BPW 1.2200656840.
- Validation receipt:
  `evidence/048/campaign/paused-resident-validation.json`

One queued worker completed block 7 durably after the first stop. Block 8 was in
progress at the final stop: `mlp.gate_proj` and `mlp.up_proj` have layer records,
but block 8 has no block journal record and is not part of the durable prefix. A
future resume must rely on normal identity-checked orphan/checkpoint discovery;
it must not manually promote partial block-8 files into the journal.

## Data and campaign state

Neither Experiment 048 C4 range has been opened:

- selection `experiment048-c4-validation344-48x512`: `reserved`;
- confirmation `experiment048-c4-validation392-48x512`: `reserved`.

The permanent slice registry validates with 14 retired and two reserved ranges.
There is no campaign receipt, selector decision, selected-checkpoint
materialization, temperature fit, confirmation result, complete export, GGUF,
WikiText result, or task result for this fresh campaign.

## Numerical history worth retaining

The first canonical process failed at block 1 because a freshly computed target
weighted mean square was non-finite. The committed block-0 teacher and compressed
activation tensors were scanned in full and contained zero non-finite values. A
bounded resume then completed block 1 with finite boundary metrics. This supports
a process-local numerical failure, not persisted activation corruption.

Every completed block observed a non-finite post-block-refit epoch and used the
explicit finite-state rollback introduced in commit `2aa4298`. Persisted block
boundary metrics are finite and the partial artifact validator passes. Do not
remove that rollback or weaken the finite-loss commit boundary when resuming.

## Source state and safe resume procedure

The last committed methodology change is `20c52ed` (`Bind correction policy
development regimes`). At pause time the worktree also contained uncommitted
changes to `tools/run_experiment048_campaign.py` and its unit test, plus unrelated
model-card, publication, and analysis-document changes. Those changes belong to
the ongoing/user worktree and must be reviewed rather than discarded before a
future resume.

Before resuming:

1. inspect process ownership by full command line and confirm no CUDA worker;
2. inspect `nvidia-smi` and the device lease;
3. rerun the partial validator and require the same block-0-through-7 prefix;
4. review and validate the uncommitted block-bounded orchestrator changes;
5. explicitly remove `evidence/048/PAUSED` only after deciding to resume;
6. restart the canonical campaign orchestrator, not a second resident worker;
7. let identity-bound resume/checkpoint discovery handle block 8;
8. keep both C4 ranges unopened until resident completion, strict complete
   validation, and the immutable campaign receipt.

The experiment is paused, not rejected or accepted. No quality conclusion can
be drawn from this partial run.

## Resume event

The user subsequently directed the experiment to continue. The pause sentinel
was explicitly removed after rechecking process ownership, the device lease,
the journal, and the saved validation receipt. A bounded resident worker then
completed block 8 at journal sequence 54. This document remains the historical
handoff for the block-0-through-7 pause point; the live status and later
evidence are recorded in
`Docs/Experiments/048-adaptive-capability-correction-policy.md`.
