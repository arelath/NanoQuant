# Tail-Aware Global KD: Staged Plan for the Final Fresh Experiment

## Status

Planning and retained-state validation are complete through the independent
non-WikiText gate. Production integration has not started, and no fresh full
compression campaign is authorized yet.

Experiment 036 remains a paused historical control. Its transplanted
composed-context seed and block-25 compensator are rejected. The eventual
fresh campaign should receive a new experiment number because its hypothesis
is now a global-distillation objective change rather than portable MLP-scale
initialization.

## Evidence that fixes the candidate

The retained Experiment 035 pre-KD state supports the following conclusions:

1. Conditional teacher-top-64 KD is invariant to the selected logits moving
   together relative to the unobserved vocabulary tail. It collapses selected
   probability mass and makes block 25 act as a nonlinear compensator.
2. Adding one aggregated tail bucket removes that invariance. A fresh
   block-25 refit after tail-aware KD is significantly harmful, so it must not
   be included in the next campaign.
3. At 256 optimizer steps, broader 1x256 and 2x128 schedules do not improve the
   retained WikiText-plus-task tradeoff over 8 epochs x 32 batches.
4. A binary mass coefficient of 0.5 is the only tested value that improves
   both retained WikiText PPL and the six-task mean. A coefficient of 0.25
   improves language modeling further but loses the task gain; 1.0 preserves
   mass more strongly but gives back both retained quality improvements.

The selected retained checkpoint is therefore:

| Field | Value |
| --- | --- |
| Objective | teacher top 64 plus one aggregated tail bucket |
| Conditional-shape coefficient | 1.0 |
| Binary selected-mass coefficient | **0.5** |
| Epochs | 8 |
| Maximum batches per epoch | 32 |
| Total optimizer steps | 256 |
| Learning rate | 1e-5 with the existing cosine schedule |
| Selected tokens per batch | at most 512 |
| WikiText PPL | **178.363145** |
| Six-task 200-example mean | **0.462500** |
| Effective BPW | 1.024494712 |

This is a model-level candidate, not a universal default. The coefficient and
schedule must remain explicit protocol fields.

## Independent C4 gate

Review item 6 requires a non-WikiText NLL/KL gate. The retained comparison uses
the immutable `allenai/c4` commit
`f998d2cd8b92435980789e3ecb2f89b4c68bfe1e`, validation shard
`en/c4-validation.00000-of-00008.json.gz` (SHA-256
`bc35d7c1b1d14b90cd3a394cccbcbe191935edd04bf42ee965379c6e2987a5f0`).
It follows the legacy contiguous-document construction, then selects windows
104-151 at 512 tokens. The resulting 24,528 scored tokens have hash
`sha256:e34b788d48b021857df1130779e98d3936d4275bb17553e079a027e496bc2bef`.

| Arm | C4 NLL | C4 PPL | BF16-teacher KL |
| --- | ---: | ---: | ---: |
| Pre-KD compressed | 5.257306 | 191.963656 | 1.222270 |
| Exact 1.0 tail mass | 5.015427 | 150.720479 | 1.106283 |
| **0.5 tail mass** | **4.970962** | **144.165461** | **1.087800** |
| 0.25 tail mass | 4.895788 | 133.725278 | 1.080069 |

The predeclared 0.5-minus-1.0 comparison passes both paired gates:

- NLL delta `-0.044465`, 95% interval `[-0.050693, -0.038023]`;
- KL delta `-0.018483`, 95% interval `[-0.021759, -0.015215]`.

The 0.25 arm is again the pure-language-model optimum and again remains
rejected because its retained six-task mean falls from 0.4625 to 0.4550. The
0.5 selection is not an artifact of WikiText.

## Gate A: production objective contract

Before a fresh campaign, implement the retained behavior through the ordinary
global-distillation workflow:

1. Add an explicit `top_k_tail` distillation loss. Keep `top_k` as the default
   so existing parity recipes and old run identities do not change silently.
2. Add `tail_mass_weight` and `maximum_batches_per_epoch` as validated,
   semantically hashed configuration fields. The candidate recipe sets them to
   0.5 and 32; defaults preserve existing behavior.
3. Extend each teacher batch with the teacher full-vocabulary log-normalizer
   for every selected token. Compute it in bounded vocabulary/token chunks,
   transfer only the one-value-per-token normalizer to CPU, include its bytes,
   and persist it in a new teacher-epoch schema.
4. Bind the objective, coefficient, batch cap, normalizer algorithm, and target
   inventory into teacher-cache and optimizer-checkpoint identities. A
   conditional checkpoint must never resume as a tail-aware run.
5. Dispatch the existing conditional loss unchanged for `top_k`; dispatch the
   tested top-k-plus-tail math for `top_k_tail`.
6. Increment the resident algorithm version when this semantic execution path
   is integrated so shared-store orphan adoption cannot mix old and new
   commits.

## Gate B: production tests

The implementation is not ready for real-model execution until all of these
are green:

- exact 1.0 equivalence to a direct 65-category cross entropy;
- coefficient decomposition and finite gradients at tail-mass extremes;
- conditional-cache backward compatibility;
- tail-normalizer cache round trip, byte accounting, and schema rejection;
- protocol-hash changes for objective, coefficient, and batch cap;
- interrupted/resumed tiny global KD equals uninterrupted execution;
- a tiny end-to-end run proves the conditional and tail objectives diverge in
  the expected mass direction;
- architecture, full Ruff, mypy, and repository tests.

## Gate C: retained production-path replay

Run the production implementation from the retained Experiment 035 pre-KD
state before spending on fresh factorization. This is distinct from the
analysis checkpoint already evaluated.

The replay must:

1. use the ordinary `run_global_topk_distillation` orchestration, durable
   teacher epochs, checkpoints, global-tuning commit, and loader;
2. interrupt after at least one epoch, resume in a new process, and match an
   uninterrupted control;
3. reproduce the analysis arm's target-selection hash, 256 optimizer steps,
   and held-out NLL/KL direction within a declared numerical tolerance;
4. select the lowest held-out WikiText-validation NLL among checkpoints whose
   full KL and tail KL both improve over pre-KD and whose teacher-top-64 mass
   remains at least 0.75;
5. pass the complete artifact validator, factorized reload, WikiText 64x128,
   six-task 200-example, and pinned C4 48x512 gates;
6. confirm once more that a fresh block-25 refit has no positive marginal.

The C4 slice is a final gate, not a checkpoint-selection input.

## Gate D: fresh full experiment

Only after Gates A-C pass should a new numbered campaign start. It should
factorize one fresh Gemma D2 frozen state, then branch that exact state into a
matched legacy conditional control and the 0.5 tail-aware candidate. This
avoids confounding the objective comparison with allocation or ADMM variance.

The candidate is accepted only if it:

1. completes and freshly validates all 26 resident blocks and every transitive
   artifact;
2. resumes correctly across the compression and KD boundaries;
3. preserves the intended effective BPW and reports logical, packed, and
   physical bytes;
4. completes `execute_complete_compression`, logical/packed reload, GGUF
   export, and the standard export summary;
5. improves same-frozen-state held-out NLL/full KL over the conditional arm
   without selected-mass collapse;
6. improves WikiText PPL and does not regress the six-task mean relative to the
   same-run pre-KD state;
7. passes the pinned C4 paired NLL/KL gate;
8. shows no new block-25-class correctable defect;
9. reports ranks, BPW, stage wall time, peak GPU/host memory, artifact bytes,
   and quality against Experiments 022 and 035.

No acceptance claim may rest on training loss, a reduced fixture, or the
retained replay alone.

## Retained evidence

- `evidence/035/experiment035-topk-tail-mass0p5-bounded8x32-monitor16/report.json`
- `evidence/035/experiment035-topk-tail-mass0p5-bounded8x32-monitor16-quality.json`
- `evidence/035/experiment035-topk-tail-mass0p25-bounded8x32-monitor16-quality.json`
- `evidence/035/experiment035-tail-mass-c4-validation104-48x512.json`
- [74-block25-anomaly-and-topk-tail-mass-audit.md](74-block25-anomaly-and-topk-tail-mass-audit.md)
- [75-topk-tail-kd-objective-ablation.md](75-topk-tail-kd-objective-ablation.md)
