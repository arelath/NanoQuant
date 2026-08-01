# Analysis: Experiments 035/036 — What Failed, What Actually Drove the Gains, and the Transferable Path

Review of [71-composed-context-mlp-scale-refit-plan.md](71-composed-context-mlp-scale-refit-plan.md)
(execution results through Phase C and the 035/036 sections),
[Experiments/035-foldable-mlp-d2-compression.md](Experiments/035-foldable-mlp-d2-compression.md),
the in-progress
[Experiments/036-composed-context-initialized-foldable-mlp-d2.md](Experiments/036-composed-context-initialized-foldable-mlp-d2.md),
and their implementation (`src/nanoquant/foldable_mlp_tuning.py`,
`src/nanoquant/application/foldable_mlp_multipliers.py`,
`src/nanoquant/infrastructure/foldable_mlp_initializer.py`,
`tools/export_foldable_mlp_initializer.py`, the 036 launcher, and the seed
asset). Follows [70-tuned-mlp-scale-placement-analysis.md](70-tuned-mlp-scale-placement-analysis.md).
Written 2026-07-31.

## Summary

**Experiment 035 failed for a reason its own inputs predicted.** It
productionized the polish (a ±0.2% multiplier continuation) without the
discovery (the composed-context refit state that carried ~92% of Phase C's
gain), and it trained that continuation on an objective — the retained top-k
teacher cache — that the evidence shows is nearly blind to the errors that
matter: 035's foldable stage improved its training loss by 10% while moving
untouched held-out NLL by +0.0002.

**A finding bigger than any of this sits in the Phase A evidence and deserves
to be pulled out of the footnotes: roughly 31% of the post-KD model's entire
KL to the teacher (0.65 of 2.04 nats) is one block's separable scale error
(block 25), and teacher-context fitting removes ~98% as much of it as the
student-context method does.** The composed-context innovation is real but
second-order (±0.01-0.017 NLL); the first-order phenomena are (a) a gross,
KD-invisible defect at the final block, and (b) re-screening post-KD with
adequate statistical power. Doc 70's root-cause claim — teacher-context
calibration as the primary defect — was tested by Phase A and holds only in a
weak form. This review revises it.

**Experiment 036 is a legitimate controlled experiment with strong gates, but
its hypothesis is likely too optimistic and its failure mode is expensive.**
It transplants 124,416 per-channel multiplier values fitted to Experiment
022's specific post-KD factor realization onto a fresh campaign with a
different allocation, different factors, and a different KD trajectory —
values whose extremes (0.0156× to a clamp-saturated 100×) encode exactly the
least transferable, most realization-specific corrections. Two cheap
analysis-only tests on the retained 035 artifacts would settle the transfer
question in hours, before the ~3.2-hour campaign completes or is repeated.

The durable recommendation is unchanged in direction from Doc 70 but sharper
in target: **productionize the refit procedure (run per-block scale refits on
each fresh campaign's own post-KD factors — teacher-context fitting
suffices for the dominant effect), and fix the KD objective blindness that
plants or preserves the block-25-class defect in the first place.**

## What went wrong in Experiment 035

### The stage tested was not the thing that produced the value

Phase C's accepted result decomposes into a six-block composed-context refit
state (worth ~−20% perplexity: 216.24 → 172.38) plus a conservative 64-step
continuation on top of it (worth −1.7%: 172.38 → 169.48). Experiment 035
shipped only the continuation, from identity, and hoped it would act as a
discovery mechanism.

It cannot. With learning rate 1e-4, cosine decay, and family identity penalty
100, the 035 run's multipliers finished in [0.9978, 1.0021] — a ±0.2%
perturbation. The refits that carry value contain per-channel corrections of
0.25× to 8×. Phase C's own accepted-run statistics (multipliers in
[0.9967, 1.0033]) already showed the continuation's movement capacity; the
identity-initialized outcome was predictable without running a campaign.

### The training objective cannot see the errors that matter

Three independent pieces of evidence converge:

1. **035's own ablation**: cached top-k loss improved 1.631 → 1.468 (−10%)
   across the 64 steps, while untouched held-out NLL moved +0.000173
   (interval [−0.000564, +0.000881]). Ten percent training-objective progress,
   zero deployed-metric progress.
2. **Phase C's checkpoint curve**: top-k KL improved monotonically for 2,048
   steps while held-out causal NLL peaked at step 64 and then steadily
   regressed. Docs/71 already names this an objective mismatch.
3. **The block-25 defect survived eight epochs of global top-k KD.** A
   separable, per-channel-scale-correctable error worth 0.65 nats of
   full-vocabulary KL — 31% of the model's total — sat at the final block
   through the entire distillation stage. Pre-KD, block 25's marginal was
   −0.009 NLL (inconclusive); post-KD it is −0.44. Global KD either created
   the defect or grew it ~40×, and its objective never saw it.

The mechanism is consistent with how top-k distillation works: the loss is
computed on the teacher's top-k token mass. Final-block distortions that
preserve top-k structure but corrupt the remainder of the distribution (tail
mass, normalization) pass through unpenalized, then dominate full-vocabulary
KL and NLL. Re-running the same objective post-KD with fewer degrees of
freedom (569,088 multipliers) from identity finds nothing, because global KD
just finished optimizing that exact objective — the stage starts at its own
optimum.

### Secondary observations

- The 8-byte packed increase came from the fresh campaign's allocation, not
  the foldable stage; it broke the literal byte gate and, more usefully, it
  is direct proof that fresh campaigns produce different factor inventories —
  which matters for 036 (below).
- Campaign-to-campaign variance is large: the 035 same-recipe rerun's post-KD
  baseline is 221.04 ppl vs 022's 228.55 (−3.3%) — bigger than the Phase C
  continuation effect (−1.7%). The doc handled this correctly with the
  same-run ablation; cross-campaign perplexity comparisons should continue to
  be treated as noise-dominated.
- Process verdict: the gates worked exactly as designed — 035 is a correctly
  rejected experiment, honestly reported. What failed was expectation
  calibration: a full campaign was spent confirming something the Phase C
  multiplier statistics already implied.

## Reframing: what actually produced the Phase A-C gains

The Phase A confirmation (48×512, per-block refits vs the post-KD baseline,
`experiment022-postkd-context-ablation-fit380-val384-confirm412-48x512.json`)
is the decisive dataset:

| Block:policy | Teacher-context NLL Δ | Student-context NLL Δ | Teacher KL Δ | Student KL Δ |
| --- | ---: | ---: | ---: | ---: |
| 25:joint | **−0.4287** | **−0.4387** | **−0.6512** | **−0.6339** |
| 23:joint | −0.0448 | −0.0615 | −0.0518 | −0.0735 |
| 4:joint | −0.0363 | −0.0403 | −0.0491 | −0.0532 |
| 21:joint | −0.0145 | −0.0099 | −0.0152 | −0.0111 |
| 18:joint | −0.0087 | −0.0074 (state) | −0.0104 | −0.0301 (state) |
| 19:joint | +0.0002 | +0.0025 | +0.0007 | +0.0047 |
| 0:output | +0.0196 | +0.0157 | +0.0235 | +0.0196 |

Baseline: NLL 4.7964, KL 2.0435 nats. Three conclusions:

1. **Block 25 is not a placement result; it is a defect report.** One block's
   separable row/column scales account for 31% of the entire compressed
   model's KL to its teacher. The same fit takes the composed KL from 2.04 to
   1.39 nats. Nothing else in the table is within 7× of it. This is a
   pipeline pathology localized at the final block, not evidence that scale
   refitting "works better than expected."
2. **The context innovation is second-order.** Teacher-context fitting
   captures ~98% of the block-25 effect and actually beats student context on
   its KL. Across the table, context choice moves outcomes by ±0.01-0.017
   NLL — real, worth having (student wins at 23 and 4, teacher wins at 21,
   block 18 splits by metric), but not the driver. Doc 70's "root defect"
   framing and Doc 71's H1 are confirmed only in this weak form.
3. **Statistical power and post-KD re-screening did the discovery.** Block 25
   pre-KD at 12×512: −0.009, inconclusive, dropped. Post-KD at 48×512:
   −0.44, unmistakable. Block 4 flipped sign entirely (pre-KD marginal
   +0.049 harmful; post-KD isolated −0.04 helpful — though it regressed
   again when composed after block 25). Pre-KD conclusions about placement
   simply do not survive the KD boundary, and 12-sequence screens mislabel
   even enormous effects.

A corollary worth recording: Phase A's fresh block-0 refits regressed in both
contexts (+0.016 to +0.020 NLL), so the incumbent block-0 refit was retained.
The post-KD model had re-co-adapted around the original refit during KD;
re-fitting a block that the network has already compensated for is harmful.
Refit conclusions are state-specific — another reason the fitted *values*
are not portable artifacts.

## Is Experiment 036 on the right track?

### What is right

- The controlled design is exactly correct: identical recipe to 035, one
  change (initialization), same gates. Whatever happens is attributable.
- The engineering is careful: hash-pinned seed, exact replay of the accepted
  54 component tensors verified at export, covariant application through
  `rescale_factorized_terms` (scales, outlier rows/columns, both patch
  sides), fold-before-continuation so the identity penalty centers on the
  seeded state, byte/shape/dtype neutrality preserved.
- The direction — get the composed-context state into production rather than
  identity — is the correct lesson from 035.

### What is questionable

**1. It transplants values fitted to a different artifact.** The seed's
124,416 multipliers were recovered (by clamped alternating ratio fits,
`_axis_scales`, bounds [0.01, 100]) from Experiment 022's accepted six-block
state — i.e., they are per-channel corrections to *022's* post-KD factor
realization. A fresh campaign has a different measured-KL allocation (the
035 packed payload differed by 8 bytes — the inventories are not identical),
different ADMM factor realizations, and a different KD trajectory. The
channel axes are shared model semantics, so a systematic component (whatever
part of the block-25 defect is reproducible pipeline behavior) may transfer;
the realization-specific component will not, and it is exactly the extreme
values that are most realization-specific:

| Seed tensor (worst cases) | min | p25 | p75 | max | frac > 2× |
| --- | ---: | ---: | ---: | ---: | ---: |
| block 18 up output | 0.080 | 0.818 | 1.116 | **100.0** | 0.045 |
| block 18 gate output | **0.0156** | 0.887 | 1.347 | 8.0 | 0.035 |
| block 23 up output | 0.094 | 0.743 | 2.025 | 26.2 | **0.256** |
| block 24 up output | 0.100 | 0.607 | 2.300 | 7.96 | 0.287 |

The exact-100 and exact-0.0156 values are `_ratio` clamp saturations —
markers that the separable recovery fit diverged on those channels, not
measured corrections. Applying a 100× boost to a channel the fresh factors
reconstruct correctly injects a large error where none existed.

**2. The continuation cannot repair a bad seed — by design.** The 64-step
stage moves multipliers ±0.2-0.3%; seed extremes are two orders of magnitude
from identity; and the identity penalty now anchors *at the seed*. In 035
the regularizer pulled toward a known-safe state; in 036 it holds the model
at a transplanted state of unvalidated fitness. If the seed does not fit,
the held-out gate fails, the ~3.2-hour campaign yields one unattributed bit
("didn't transfer"), and the natural next step — another seed variant,
another campaign — repeats the cost.

**3. `initializer_multiplier_limit=128`.** The Phase C safety bound was 4.0.
The 036 launcher raises the *seed* limit 32× specifically so the clamp-
saturated extremes can pass validation. Weakening a guardrail precisely to
admit the least trustworthy values inverts its purpose. If the seed
survives review, its log-multipliers should instead be winsorized to the
original per-family fit bounds (gate ≤ 2, up ≤ 8, down ≤ 4) — any channel
beyond those bounds got there through ratio-fit divergence, not through an
accepted fit.

### Verdict

As a *controlled experiment*, 036 is acceptable and will produce a clean
answer. As a *strategy for capturing the gains*, it is probably aimed at the
wrong object: the transferable asset demonstrated by Phases A-B is the
**procedure** (post-KD per-block refit + power-adequate screens + composition
gates on the campaign's own factors), not the fitted **values**. The 036
hypothesis text — "recovers the large Phase C quality gain" — presumes the
022 defect pattern reproduces channel-for-channel in a fresh campaign, which
nothing yet supports.

Pre-register the interpretation now, before results exist: if 036 fails its
NLL gate, the correct conclusion is "transplanted values do not transfer,"
not "composed-context initialization is wrong." Without this, a failed gate
invites abandoning the genuinely valuable procedure.

## What we should be doing differently

Ranked, with the cheap items first because they de-risk the expensive ones:

1. **Run the two cheap cross-artifact tests on the retained 035 artifacts
   now** (analysis-only, hours, no campaign):
   - *Seed-transfer test:* fold the 036 seed into the 035 post-KD run and
     measure held-out NLL/KL against the 035 post-KD baseline, with a
     per-block decomposition (fold one seed block at a time). This directly
     answers 036's hypothesis before its campaign finishes, and provides
     per-block attribution its gate cannot.
   - *Fresh-defect probe:* run the per-block refit screen (teacher-context
     is sufficient) on the 035 campaign's own factors. Does a fresh campaign
     have a block-25-class defect? Same block? Same magnitude? This is the
     single most valuable bit for designing the production stage.
2. **Diagnose the block-25 defect as a pipeline bug, not a refit
   opportunity.** 31% of the model's KL in one block's separable scales,
   appearing (or growing ~40×) across the global KD stage, invisible to the
   top-k objective. Establish when it appears (checkpoint the KD run and
   track per-block separable-correctable error), whether it reproduces
   across campaigns, and whether it lives in the top-k objective's blind
   spot (compare top-k KL vs full-vocab KL sensitivity to block-25
   perturbations). If KD plants it, fix KD — a seeded patch shaped like
   022's defect could actively fight a differently-shaped defect on a fresh
   run.
3. **Productionize the refit procedure as a post-KD stage.** Per-block scale
   refit on the campaign's own factors: teacher-context fitting (simple, no
   student capture in the pipeline) as the default, student-context where
   Phase A showed it confirmed better; fit on a few 512-token windows;
   marginal composition with 24-48-sequence screens and the existing
   confirmation/export gates. Fitting cost is minutes per block; screens
   dominate and are bounded. This generalizes by construction — each
   campaign generates its own corrections — and makes the seed machinery
   unnecessary.
4. **Give the continuation an objective that can see the target, or demote
   it.** Mix subsampled full-vocabulary KL (or add tail-mass/normalizer
   matching) into the foldable-stage loss; monitor held-out NLL at every
   16-step checkpoint and select on it (Phase C's own lesson — 64 steps was
   chosen by looking at NLL, but the production stage hard-codes the step
   count and never looks). The same objective upgrade should be evaluated
   for global KD itself, where the payoff would be largest.
5. **If 036 proceeds (reasonable), harden it cheaply:** winsorize the seed to
   original fit bounds instead of raising the safety limit to 128; record a
   per-family bound-hit and extreme-value census in the stage receipt; and
   attach the per-block seed ablation from item 1 to the experiment record
   so a gate failure is attributable.
6. **Add a non-WikiText gate before scaling this stage family.** Every
   fitting window, screen, confirmation, and the perplexity gate is
   WikiText; the only out-of-distribution signal is 1,200 task examples, and
   the six-block state's −20% perplexity came with a flat-to-slightly-down
   mean task score against the five-block incumbent (HellaSwag −0.045).
   Docs/71 states the caveat; make it a gate — held-out NLL/KL on a
   non-WikiText corpus (e.g., a pinned C4 slice) as a required secondary
   check for zero-bit refit acceptance.
7. **Keep the discipline that is working.** Same-run ablations (035's was
   exemplary), fresh-window confirmations, exact fold/export parity, and
   honest rejection reporting all functioned correctly across 035 and the
   Phase C candidate rejections. The corrections needed are strategic
   (what to test next, what a result means), not procedural.

## Correction to Doc 70

Doc 70 ranked teacher-context calibration as "the root defect" behind the
placement wall. Phase A tested that claim properly and the strong form is
rejected: context choice is worth ±0.01-0.017 NLL per block, while post-KD
re-screening power and one pathological block account for the overwhelming
majority of the realized gains. The weak form survives — student context
confirmed better for blocks 23 and 4, and Doc 70's other recommendations
(power the marginal gate, re-screen post-KD, global scale optimization,
NLL/KL dual gating) are the ones Phase A-C validated. This review supersedes
Doc 70's causal ranking.

## Evidence examined

- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-context-ablation-fit380-val384-confirm412-48x512.json`
  (per-block, per-context NLL/KL deltas; block-25 magnitude; baseline levels)
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-postkd-context-ablation-fit380-val384-screen388-24x512-v2.json`
  (state diagnostics, multiplier distributions, state-recovery arm saturation)
- `evidence/035/…` and
  [Experiments/035-foldable-mlp-d2-compression.md](Experiments/035-foldable-mlp-d2-compression.md)
  (same-run ablation, multiplier ranges, top-k loss trajectory, byte deltas)
- `experiments/assets/gemma-3-1b-it-composed-context-six-block-initializer/`
  (seed manifest and per-tensor multiplier distributions)
- Implementation: [foldable_mlp_tuning.py](../src/nanoquant/foldable_mlp_tuning.py)
  (seed application, training loop, objective),
  [foldable_mlp_multipliers.py](../src/nanoquant/application/foldable_mlp_multipliers.py),
  [export_foldable_mlp_initializer.py](../tools/export_foldable_mlp_initializer.py)
  (`_axis_scales` clamp bounds), the
  [036 launcher](../experiments/036-composed-context-initialized-foldable-mlp-d2-gemma-3-1b-it.py)
  (`initializer_multiplier_limit=128.0`).
