# Analysis: Why Experiment 042 Did Not Transfer — Methodology Review of the 035-042 Sequence

Review of Experiments 035-042, Documents 72-78, and the implementation
(`src/nanoquant/global_distillation.py`,
`src/nanoquant/application/distillation.py`,
`src/nanoquant/final_norm_calibration.py`,
`src/nanoquant/config/schema.py`, the 037 and 042 launchers).
Follows [72-experiment-035-036-review.md](72-experiment-035-036-review.md).
Written 2026-08-01.

## Summary

Experiment 042 completed on 2026-08-01. Its absolute result is fine: WikiText
perplexity 171.87 versus Experiment 040's 172.63, C4 NLL 4.82922, six-task mean
0.45850 at 1,000 examples versus 040's 0.45283. A paired comparison of the two
endpoints gives `+0.00567` with interval `[-0.00467, +0.01583]` — the fresh
campaign and the retained one are statistically the same model. **In absolute
terms the method transferred.**

What did not transfer is the *marginal*. Experiment 040 measured the correction
as `-0.0575` C4 NLL and `-0.226` C4 KL. Experiment 042 measured the same
correction as `-0.316` C4 NLL and `-0.474` C4 KL — 5.5× larger on NLL and 2.1×
on KL, while landing in the same place. A gain that grows when the method is
unchanged and the endpoint is unchanged is not a gain; it is a measurement of
how damaged the baseline was.

The cause is concrete and is a pipeline defect, not variance:

> **Experiment 042's primary conditional KD ran 2,048 optimizer steps. Every
> state on which the correction policy was designed, tuned, and validated
> (Experiments 038, 039, 040, 041 and the Document 78 production replay) was
> the Experiment 037 conditional arm, which ran 256 steps.**

This is not 042 tripping over a bad default. **2,048 steps is the canonical
campaign horizon** — Experiments 022, 035, and 042 all ran it, because
`DistillationConfig.maximum_batches_per_epoch` defaults to `None`
([schema.py:575](../src/nanoquant/config/schema.py#L575)) and 256 calibration
samples at batch size 1 give 256 steps per epoch. **Experiment 037 was the
exception**: its two arms passed `--maximum-batches-per-epoch 32` on the command
line, a reasonable cost-saving choice for a matched two-arm comparison.

The consequence is that the entire 038 → 039 → 040 → 041 correction line, plus
the Document 78 production replay, was designed, tuned, and validated on a state
production has never produced. The correction was calibrated against a baseline
with teacher-top-64 mass 0.486 and shipped against one with mass 0.382.

Three further problems sit underneath that one, and they are the reason the
whole 035-042 sequence has produced a 25% perplexity improvement and no
measurable downstream gain:

1. The 0.75 selected-mass floor is a self-imposed gate, satisfiable by a single
   scalar temperature, and it is what forced the fold that killed Experiment
   037's tail-aware candidate — the only change in this series with matched
   evidence that it repairs the diagnosed mechanism.
2. The diagnosed root cause (Documents 74/75: conditional top-64 KD is blind to
   tail mass and drives a final-block pathology) was **never fixed in
   production**. 042 shipped the unrepaired objective at eight times the
   horizon and recorded the largest block-25 blow-up in the series.
3. The acceptance gates cannot resolve the effect sizes being gated, and the
   metric being optimized does not predict the metric that matters.

`Docs/Experiments/042-low-pressure-correction-d2-compression.md` still reads
"Predeclared and not yet started" and should be updated; this document does not
edit it.

## 1. The KD horizon defect

### Evidence

Step counts read directly from committed `global-tuning-result.json` artifacts:

| Run | KD objective | Steps | Protocol hash prefix |
| --- | --- | ---: | --- |
| 035 campaign | conditional top-64 | **2,048** | — |
| 037 conditional arm | conditional top-64 | **256** | `bf24f470` |
| 037 tail-aware arm | top-64 + tail 0.5 | 256 | — |
| 042 campaign | conditional top-64 | **2,048** | `486d928d` |

The 042 correction stage itself is correct: 32 steps, one epoch, 128-step cosine
horizon, exactly as predeclared
(`global-distillation-mass-floor-result.json` → `steps_completed: 32`).
Only its input differs.

Document 75 had already measured what the extra horizon does. Under the
conditional objective the best monitored checkpoint is the **first 32-step
epoch**; later epochs improve the conditional loss while held-out NLL, full KL,
and selected mass all get worse. Document 75's own table lists conditional KD at
2,048 steps as 221.04 perplexity against 256-step arms at 187-188. Experiment
042's conditional arm is 236.48. It is in the long-horizon family.

### What this does to the comparison

All rows below are the **same** WikiText validation slice (offset 300, 48×512,
token hash `sha256:9ee31088130e314637ed3607ddf7903890438c0aa7e62b1cd261b971479a4aef`),
so they are directly comparable across campaigns:

| State | 037/040 campaign (KD 256 steps) | 042 campaign (KD 2,048 steps) | Δ |
| --- | ---: | ---: | ---: |
| pre-KD NLL | 4.81112 | 4.85900 | +0.048 |
| pre-KD full KL | 1.47537 | 1.48839 | +0.013 |
| pre-KD top-64 mass | 0.83221 | 0.83889 | +0.007 |
| conditional-KD NLL | 4.49452 | 4.67800 | **+0.184** |
| conditional-KD full KL | 1.58588 | 1.81285 | **+0.227** |
| conditional-KD mass | 0.48618 | 0.38236 | **-0.104** |
| final candidate NLL | 4.40061 | 4.41761 | +0.017 |
| final candidate full KL | 1.22031 | 1.21130 | -0.009 |
| final candidate mass | 0.76308 | 0.74032 | -0.023 |

Read this table top to bottom. **The two factorizations are nearly identical**
(0.048 nats apart pre-KD — the campaign-variance worry from Document 72 is real
but small). **The KD stage manufactures the divergence** (0.184 nats, 0.227 KL,
0.104 mass). **The correction erases it** (0.017 nats apart at the endpoint).

Measured against pre-KD, the complete pipeline delivers `-0.410` nats in the
037/040 line and `-0.441` nats in 042. Consistent. Measured against the
conditional baseline, it delivers `-0.094` and `-0.260`. Inconsistent by 2.8×.
The correction is not adding value on top of KD; it is undoing a variable amount
of damage that KD does, and the reported marginal is a measurement of the damage.

### Why the production replay did not catch it

Document 78's replay was rigorous about the wrong axis. It proved byte-exact
tensor equality, resumability across a process boundary, and even found a real
scheduler-horizon defect (32 executed steps ≠ 32 scheduled steps). But it was a
hard-link fork of **the audited Experiment 037 conditional-KD initializer** —
the 256-step state. It verified that the correction reproduces on the state it
was tuned on. It could not detect that the canonical campaign feeds it a
different state.

**The artifact-identity system is doing exactly what it was built to do and
still missed this**, because the correction's protocol hash binds its
initializer *reference* but nothing gates on the initializer's *training
horizon* being the one the policy was calibrated against.

## 2. The mass floor is a temperature statistic, and it cost the best candidate

### The fold is a logit temperature

`apply_gemma_final_norm_scale`
([final_norm_calibration.py:48](../src/nanoquant/final_norm_calibration.py#L48))
computes `new = (1 + old) * scale - 1` on `model.norm.weight`. The pinned
Gemma-3-1B config has `final_logit_softcapping = null`, so nothing sits between
the final RMSNorm and the tied head. Scaling the norm's effective weight by `s`
scales the hidden state entering the head by `s`, which scales every logit by
`s`. **The fold is exactly a softmax temperature of `1/s`, applied globally.**

Document 77 says as much ("The mass deficit is substantially a logit-temperature
error") and treats it as a virtue — it costs no tensor, no operation, no BPW.
That is true, and it is also the problem: a quantity that a one-parameter
temperature can move at will is not measuring compression quality.

"Student mass on the teacher's top 64" is precisely such a quantity. So the
0.75 floor is a calibration gate wearing the costume of a capability gate. Its
value appears first in Document 76 as a checkpoint-selection criterion, and
Experiment 040 calls it "the deployment mass floor" — but nothing in the docs
derives 0.75 from a deployment requirement (a sampler truncation threshold, a
speculative-decoding acceptance rate, a serving constraint). It should either be
tied to one or removed.

### What it cost

Experiment 037's rejection had two legs. Both need restating.

**The task leg was not significant.** The predeclared gate used 200 examples and
saw `-0.01750`. At 1,000 examples the paired task-stratified bootstrap gives
`-0.00667` with interval `[-0.01433, +0.00100]` — no established regression.
(Computed with `tools/compare_quality_task_reports.py` from the retained
037 evidence.)

**The C4 leg was significant, and it was caused by the fold, not the objective:**

| Arm | C4 NLL vs conditional | Paired 95% interval | Verdict |
| --- | ---: | --- | --- |
| Tail-aware 0.5, **uncalibrated** | **-0.03706** | `[-0.05299, -0.02158]` | passes decisively |
| Tail-aware 0.5 **+ 1.06 fold** | +0.02249 | `[+0.00300, +0.04188]` | fails |

The uncalibrated tail-aware arm dominated conditional KD on WikiText NLL, full
KL, tail KL, C4 NLL, and C4 KL, and had no established task regression. It was
rejected because it reached mass 0.708, below a floor of 0.750, and the fold
required to clear that floor gave back the C4 NLL.

**A self-imposed, temperature-satisfiable gate rejected the only candidate in
this series with matched evidence that it repairs the diagnosed mechanism.**

### Attributing the correction between its two parts

The 042 endpoint is `conditional KD (2,048) → 32-step one-sided correction →
1.015 fold`. From Experiment 040's own fold sweep, the four screened scales span
only `4.34992` to `4.36319` NLL (0.013 nats across 1.005-1.020), and applying
1.015 to a pre-KD state costs about 0.032 nats. The trained 32-step correction
is doing most of the NLL work; the fold is a small, cheap, final calibration.
That is the right division of labour — but it means the *fold* is not the
lever, and the *floor* that mandates the fold is buying almost nothing.

## 3. The diagnosed root cause was never fixed in production

Documents 74 and 75 did the best work in this sequence. They established, with
controls, that conditional top-64 KD is invariant to moving all selected logits
together against the unobserved tail; that this collapses selected mass; and
that block 25 becomes a nonlinear compensator for the resulting model-wide
output-distribution error.

The fix was implemented (`topk_tail_distillation_loss`,
[distillation.py:311](../src/nanoquant/application/distillation.py#L311)),
validated, and then **not shipped**. `top_k` remains the default and is what
042 ran.

Block-output hidden-state MSE change across KD, from the committed
`block_metrics` of each run:

| Run | Objective | Steps | Block 24 | **Block 25** |
| --- | --- | ---: | ---: | ---: |
| 022 campaign | conditional | 2,048 | +0.440 | +6.818 |
| 035 campaign | conditional | 2,048 | +0.480 | +6.746 |
| 037 conditional arm | conditional | 256 | — | +6.512 |
| **037 tail-aware arm** | **top-64 + tail 0.5** | **256** | — | **+3.371** |
| **042 campaign** | **conditional** | **2,048** | **+0.533** | **+7.318** |

Two readings, both matched and both clean:

- **The objective is the driver.** At an identical 256 steps from an identical
  frozen state, swapping conditional for tail-aware halves the final-block
  blow-up (+6.512 → +3.371). That is the mechanism being repaired, not masked.
- **042 is the worst on record** (+7.318; hidden-state MSE 12,647 → 105,194,
  an 8.3× blow-up at the final block). Experiment 042's predeclared gate 9 asked
  for a block-snapshot check for a new block-25-class defect. The snapshot shows
  the mechanism is fully active.

Experiment 041 separately showed that on the 040 endpoint a block-25 refit is
harmful, and that finding stands — but it answers "is the damage separably
repairable after the fact," not "is the damage still being created." It is.

What was productionized instead is a 32-step patch plus a scalar temperature,
applied *after* an unrepaired objective has run eight times longer than any
state the patch was calibrated on. That is treating the symptom, downstream of
a known and already-implemented cure.

## 4. The gates cannot see what they are gating

### The 200-example gate is noise

Same model, same protocol, two sample sizes:

| Arm | Six-task mean @ 200 | Six-task mean @ 1,000 | Swing |
| --- | ---: | ---: | ---: |
| 042 conditional | 0.48333 | 0.46433 | 0.019 |
| 042 final | 0.46417 | 0.45850 | 0.006 |

A 0.019 swing on an unchanged model, from sample size alone. Experiment 035's
gate 7, Experiment 037's no-regression gate, and Document 77's comparison table
are all 200-example. Experiment 037 was rejected in part on a 200-example delta
of 0.0175 — smaller than the sampling swing above.

### The 1,000-example gate is barely adequate

Every 1,000-example arm in this series, identical protocol:

| Arm | WikiText PPL | Six-task mean @ 1,000 |
| --- | ---: | ---: |
| BF16 teacher | 96.46 | **0.60567** |
| 037 conditional (256 steps) | 187.52 | 0.45317 |
| 037 tail-aware 0.5 uncalibrated | 174.08 | 0.44650 |
| 037 tail-aware 0.5 + 1.06 fold | 186.92 | 0.44633 |
| 040 final | 172.63 | 0.45283 |
| 042 conditional (2,048 steps) | 236.48 | 0.46433 |
| 042 final | **171.87** | 0.45850 |

Every compressed variant sits in `0.4463-0.4643` — a band of 0.018 — while the
gap to the teacher is **0.152**. Paired intervals are roughly ±0.010 half-width.
The gate can resolve about 7% of the remaining gap. Eight experiments have moved
task quality by, at most, noise.

### Perplexity has not been buying task quality

WikiText perplexity has fallen from Experiment 022's 228.55 to Experiment 042's
171.87 — a 25% improvement, and the sequence has been selecting on perplexity,
NLL, KL, and selected mass throughout. The task band above did not move with it.

(I checked whether the two axes correlate across these six arms and am not
reporting the number: it is driven entirely by one point, and that point is the
2,048-step arm Section 1 establishes as defective. Dropping it flips the sign.
With n = 6 and one high-leverage observation the statistic is not evidence in
either direction.)

The sharpest single instance is a direct comparison instead. 042's conditional
arm has 49 more perplexity points, 0.227 worse full KL, and 0.104 worse selected
mass than 037's — and a paired task mean that is *better* by `+0.01117`,
interval `[+0.00133, +0.02133]`, which excludes zero. This is confounded by
campaign (different factorizations) and by the horizon defect, so it is not
proof of anything. It is a warning that the axis being optimized and the axis
that matters are, at best, decoupled — and that eight experiments of selecting
on the first have produced nothing measurable on the second.

### Experiment 042's gate 8 was not computed in the run

The predeclared protocol requires a 1,000-example six-task confirmation. No
paired interval for candidate-versus-conditional exists in `evidence/042` — the
`comparison` blocks there are frozen-versus-BF16. Computed here with the repo's
own tool, it passes:

```
042 final vs 042 conditional, 1,000 examples, paired task-stratified bootstrap
  baseline  0.46433   candidate  0.45850   delta  -0.00583
  95% interval [-0.01233, +0.00067]     regression established: false
  piqa -0.001 | arc_easy -0.009 | arc_challenge -0.006
  hellaswag -0.016 | winogrande -0.005 | boolq +0.002
```

Five of six tasks move down and the interval barely contains zero. By the letter
of the gate, no regression is established. By any honest reading, the correction
is directionally harmful on tasks in this campaign, and considerably closer to a
supported regression than Experiment 040's `-0.00033` was.

## 5. Secondary findings

**Reproducibility defect (real, but not the cause).** Experiments 037 and 042
share config hash `08d2e590…`, model hash `32d5b5d0…`, and the same pinned KL
profile artifact and key. They produced **different allocation plans**
(`2eaebd95…` vs `14ae9d3f…`). The difference traces to the calibration-stats
artifact (`e14166a0…` → `5569d719…`) and its derived per-layer importance
vectors. Net effect on the plan: 2 of 130 ranks changed (block 8 `mlp.gate_proj`
736→768, block 11 `mlp.up_proj` 928→896) and `planned_cost` is byte-identical.
`ReproducibilityConfig` defaults to `deterministic=True`,
`allow_nondeterministic_kernels=False`, so this should not happen. **It cannot
explain 49 perplexity points** — the pre-KD states are 0.048 nats apart — but
until it is fixed, no cross-campaign comparison is clean, and every "fresh
campaign" gate is confounded by an unknown amount.

**The mass-floor term constrains a batch mean.** In
`topk_mass_floor_distillation_loss`
([distillation.py:471](../src/nanoquant/application/distillation.py#L471)):

```python
mass_deficit = torch.relu(torch.logit(target_mass) - torch.logit(student_mass_mean))
```

`student_mass_mean` is the mean over the microbatch. Two consequences, both
design weaknesses rather than demonstrated defects: the constraint is
satisfiable by raising mass on tokens that already have it rather than repairing
collapsed ones, and a hinge applied to a stochastic mean is a biased estimator
of the hinge on the population constraint (Jensen — `E[relu(f(mean))]` is not
`relu(f(E[mean]))`). A per-token hinge, or a dual variable tracking the
population constraint across steps, would target the actual failure.

**Selection-slice reuse.** Experiment 040 selected the fold scale on WikiText
validation offset 104, described as "now released by the rejected Experiment
038." That slice had already been used for selection in 038 and 039. Releasing a
slice because the candidate that used it was rejected does not restore its
independence — the rejection itself was informed by it. Mild, but it compounds
across a nine-experiment sequence in which the same handful of slices are
reused.

**Marginal-only reporting.** The pattern throughout 038-042 is to report
candidate-minus-previous-stage. When the previous stage is unstable, that number
is uninterpretable — which is exactly what happened. Absolute values against
fixed references (pre-KD and BF16) would have made the 042 anomaly visible
immediately.

## 6. What to do differently

Ranked; the cheap items are first because they de-risk the expensive ones.

1. **Decide which KD horizon is production, then re-derive the correction
   there.** This is a decision, not a diagnostic, because 2,048 steps is what
   campaigns actually run and 256 is what the correction was fitted to. Two
   coherent options; pick one:
   - *Change the canonical horizon.* Select the KD checkpoint on held-out
     NLL/full-KL instead of hard-coding the step count. Document 75 already
     places the conditional optimum at epoch 1 — the pipeline currently trains
     2,048 steps past a known optimum and then pays a 32-step patch to walk part
     of the way back. Then re-derive the correction on the selected state.
   - *Keep 2,048 and re-tune the correction for it.* Ratio 0.8, weight 2.0,
     horizon 128, and fold 1.015 were all fitted against a state with mass
     0.486. The state they now run on has mass 0.382.

   Either way, a capped rerun changes the teacher-cache identity: setting
   `maximum_batches_per_epoch=32` keeps the field in the protocol instead of
   popping it ([global_distillation.py:393](../src/nanoquant/global_distillation.py#L393)),
   so the 411 MB cache is rebuilt. Budget for it.

2. **Bind the horizon to the correction's identity.** The correction's protocol
   hash binds its initializer *reference* but nothing gates on that initializer
   having the training horizon the policy was calibrated against. That gap is
   what let a validated replay and a fresh campaign disagree silently.

3. **Ship the objective fix.** `top_k_tail` at coefficient 0.5, as the primary
   KD objective. At matched 256 steps from a matched frozen state it halves the
   block-25 blow-up, improves WikiText NLL, full KL, tail KL, C4 NLL, and C4 KL,
   and has no established task regression. It was rejected on a 200-example
   task delta below the sampling noise and on a C4 failure caused by the fold
   that the mass floor mandated. Re-run it as a matched arm without the floor.

4. **Delete or derive the 0.75 mass floor.** If it protects a real deployment
   behaviour, name that behaviour and gate on it directly — sampler truncation
   quality, speculative-decoding acceptance rate, a generation metric. If it
   does not, it is a temperature statistic that has already cost one good
   candidate.

5. **Separate calibration from capability in every report.** Report NLL and KL
   both raw and after fitting a per-arm optimal temperature. Any improvement a
   single scalar can reproduce is a calibration improvement and should never
   count toward accepting a compression change. Keep the fold — it is free and
   it helps — but apply it last, to both arms, and report it separately from
   the comparison.

6. **Add a metric that a temperature cannot fake, and power the gates.**
   Top-1 agreement rate with the teacher's argmax per token is cheap, is
   invariant to global temperature, and measures the thing compression actually
   destroys. On tasks, either raise to 5,000+ examples per task, or accept that
   the six-task mean is a guardrail rather than a selection signal and say so in
   the gate text. Stop gating on 200-example means entirely.

7. **Fix the calibration-statistics nondeterminism.** Same config, same model,
   same pinned profile must produce the same plan hash, or the artifact-identity
   system's central promise does not hold across campaigns.

8. **If the mass-floor term is kept, make it per-token.** Replace the hinge on
   the batch mean with a per-token hinge or a dual variable on the population
   constraint, so the pressure lands on the tokens that actually collapsed.

9. **Update `Docs/Experiments/042-low-pressure-correction-d2-compression.md`.**
   It records the run as not started. The run completed on 2026-08-01 and
   passed gates 1-7. Gate 8's paired confirmation was not computed in the run's
   evidence; computed here it passes by the letter (`-0.00583`,
   `[-0.01233, +0.00067]`) with five of six tasks down. Gate 9's block snapshot
   shows the largest final-block blow-up in the series.

## 7. What is working and should not change

The infrastructure discipline in this repository is genuinely unusual and it is
why this analysis was possible at all. Content-addressed artifacts, hash-bound
protocols, interrupted/resumed equivalence proofs, byte-exact fold replay,
same-run ablations, predeclared gates, and honest rejection reporting all
functioned correctly. Document 78 found a real scheduler defect through replay.
Documents 74 and 75 diagnosed a subtle objective pathology with proper controls
and reversed an earlier wrong conclusion in the process.

The failures here are not procedural. They are about **what gets measured
against what**: a marginal against an unstable baseline, a calibration statistic
treated as a capability gate, an acceptance threshold below the noise floor, and
a validated cure left switched off while a patch for its symptom was shipped.

## Evidence

Read for this analysis:

- `evidence/042/.../artifacts/9e/sha256-9e4f7d1d…/global-tuning-result.json`
  (042 primary KD: `steps_completed` 2,048, `block_metrics`)
- `evidence/037/037-matched-tail-aware-d2-conditional-gemma-3-1b-it/.../global-tuning-result.json`
  (037 conditional: `steps_completed` 256)
- `evidence/037/037-matched-tail-aware-d2-tail-aware-gemma-3-1b-it/.../global-tuning-result.json`
  (037 tail-aware: 256 steps, block-25 +3.371)
- `evidence/035/035-foldable-mlp-d2-.../global-tuning-result.json` (2,048 steps)
- `evidence/042/.../global-distillation-mass-floor-result.json` (32 steps)
- `evidence/042/.../final-norm-calibration.json` (scale 1.015)
- `evidence/042-analysis/experiment042-prekd-{conditional,candidate}-validation300-48x512-tail-mass.json`
- `evidence/040/experiment040-conditional-epoch8-validation300-48x512-kl.json`
- `evidence/040/experiment040-weight2-epoch1-fold1p015-validation300-48x512-kl.json`
- `evidence/042/experiment042-matched-c4-validation104-48x512.json`
- `evidence/042/experiment042-{final,conditional-epoch8}-tasklimit1000-quality.json`
- `evidence/037/experiment037-{conditional,tail-aware-uncalibrated}-tasklimit1000-quality.json`
- `evidence/040/experiment040-weight2-epoch1-fold1p015-tasklimit1000-quality.json`
- Allocation plans `sha256-2eaebd95…/plan.json` (037) and `sha256-14ae9d3f…/plan.json` (042)
- Pinned `google/gemma-3-1b-it` revision `dcc83ea8…` `config.json`
  (`final_logit_softcapping: null`)

Paired intervals computed with `tools/compare_quality_task_reports.py`
(`paired-task-stratified-bootstrap-v1`, 10,000 resamples, seed 0).
