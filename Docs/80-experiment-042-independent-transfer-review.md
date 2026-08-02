# Independent Review: Experiment 042 Methodology and Transfer Failure

This review was written from the experiment records (037, 038, 039, 040, 041,
042, 043 predeclaration, Document 78) and the implementation
(`src/nanoquant/application/distillation.py`, `src/nanoquant/global_distillation.py`,
`src/nanoquant/final_norm_calibration.py`, `src/nanoquant/resident_workflow.py`,
`experiments/042-...py`, `experiments/recipes/base_compression.py`,
`tools/probe_topk_tail_mass.py`, and the distillation unit tests). It was
deliberately written without reading
`Docs/79-experiment-042-transfer-failure-analysis.md`.

## TL;DR

Experiment 042 failed its selected-mass gate (0.73052 versus the 0.75 floor)
for reasons that were predictable from arithmetic already available in the
experiment records before launch. Three compounding causes:

1. **The correction recipe was developed on a different training regime than
   it was deployed on.** Experiments 037/038/040 developed the correction
   downstream of a 256-step conditional KD (`--maximum-batches-per-epoch 32`).
   The production template runs conditional KD uncapped: 256 samples × 8
   epochs = 2,048 optimizer steps. The 8× longer horizon collapses selected
   mass far deeper (0.374 in 042 versus 0.475 in 037 on the same slice), so
   the fixed 32-step correction started in a much deeper hole.
2. **A fixed-budget correction was applied to a variable-size deficit.** The
   correction (32 steps, lr 1e-5, weight 2.0) and the fold (1.015) are point
   values frozen from one trajectory. Nothing in the recipe adapts to how much
   mass is actually missing.
3. **The engineered safety margin was ~0.0002 against observed run-to-run and
   slice-to-slice variation of ~0.01–0.02.** The scale-selection rule
   ("lowest scale reaching 0.75") guarantees a near-zero margin by
   construction, and the training target (0.8 × teacher mass ≈ 0.7487) is
   itself *below* the 0.75 deployment floor.

No implementation bug was found that explains the failure. The loss
implementations, the Gemma `(1+w)` effective-weight fold, the
scheduler-horizon handling (fixed after the Document 78 replay caught it), the
workflow wiring, and the gate tooling are all consistent with the documented
protocols. The failure is methodological, and the deeper issue is that the
mass floor is being treated as a capability when it is largely a
temperature-like calibration knob.

## 1. What the chain actually did

| Exp | Conditional KD horizon | Correction | Fold | Mass result (broad gate) | Verdict |
| --- | --- | --- | --- | --- | --- |
| 037 | 256 steps (capped 32/epoch) | none | none / 1.06 | conditional 0.47469; tail-aware+1.06 0.75475 | rejected (task + C4 NLL) |
| 038 | reused 037 (256 steps) | 4×32 steps, first survivor = weight 2.0 epoch 1 | none | fit 0.76165 → broad 0.73890 | rejected (broad mass) |
| 040 | reused 038 checkpoint | retained 32-step epoch-1 | 1.015 (swept on offset 104) | dev 0.75016; untouched offset-300 0.76308 | **accepted** |
| 042 | **2,048 steps (production default, uncapped)** | same 32-step recipe, warm-started | fixed 1.015 | offset-104 0.73052 | rejected (mass) |

Key intermediate states on the offset-104 48×512 slice:

- 037 conditional endpoint: mass 0.47469, NLL 4.47838 (pre-KD 0.82129 / 4.76183).
- 042 conditional endpoint: mass **0.37417**, NLL 4.67584 (pre-KD 0.82895 / 4.82511).
- 042 corrected + 1.015: mass 0.73052, NLL 4.37537.

The correction did enormous work in 042 (0.374 → ~0.72 before the fold) and
the candidate beat every retained comparison on NLL, full KL, C4 (decisive
paired intervals), and packed PPL (171.87 versus accepted 040's 172.71). It
was rejected purely on the mass statistic.

## 2. Root causes of the transfer failure

### 2.1 The development experiments were never run at production settings

`DistillationConfig.maximum_batches_per_epoch` defaults to `None`
([schema.py:575](../src/nanoquant/config/schema.py)), and the base compression
template only sets `enabled=True`, so the 042 campaign's primary KD ran
8 × 256 = 2,048 steps. Every experiment that developed and validated the
correction (037's matched arms, 038's coefficient sweep, 040's fold sweep, the
Document 78 byte-exact replay) ran or reused a 256-step conditional state.

This was not perceived as a variable change — 042 believed it was running "the
ordinary" pipeline — but it changed the input distribution of the correction
stage by more than any deliberately-controlled factor in the whole 037–040
sequence. Conditional mass collapse grows with horizon (0.475 at 256 steps,
0.374 at 2,048), and the conditional NLL gain appears *worse* at the longer
horizon (−0.149 from pre-KD in 042 versus −0.284 in 037, on admittedly
different factorizations). The matched-arm discipline that the sequence
otherwise maintained was silently broken at the exact transition that
mattered: development → production.

The Document 78 replay does not mitigate this. Byte-exact replay of a retained
trajectory validates the *mechanism* (resume, identity binding, scheduler
horizon), not the *policy*. It proves the code reproduces the old trajectory;
it says nothing about behavior from a new initial condition — and the recipe's
statistical validity lived entirely in the second question.

### 2.2 A fixed-budget correction cannot absorb a variable deficit

The correction recipe is a chain of point estimates, each selected as "first
survivor" on one specific trajectory: coefficient 2.0 (first of {0.5, 1.0,
2.0} to pass a fit monitor), epoch 1 (first passing checkpoint; later epochs
fell back below the floor), cosine horizon 128 (an artifact of the original
four-epoch schedule), scale 1.015 (lowest of four screened values to clear
0.75 on one dev slice). These interact with the starting state. In 038 the
correction lifted mass from ~0.51 to 0.739 broad; in 042 it had to lift from
0.374 with exactly the same 32 steps and learning rate, and reached roughly
0.72 before the fold. Nothing measures the deficit and nothing scales the
budget with it.

Estimated from the 040 dev sweep, mass sensitivity to the fold is
≈ 0.74 per unit scale (0.0037 per 0.005 step). Closing 042's 0.0195 shortfall
by scale alone would need ≈ 1.04 — heading toward the 1.06 regime that
Experiment 037 showed gives back C4 and WikiText NLL. In other words, the
"minimal fold" concept is only coherent in the shallow-collapse regime that
the 256-step horizon produces. Deployed after a 2,048-step collapse, the same
recipe arithmetic no longer holds.

### 2.3 Margins were an order of magnitude smaller than known noise

The records themselves quantify the variability of the mass statistic:

| Source of variation | Observed size |
| --- | ---: |
| Fit monitor → broad slice (038, same checkpoint) | −0.0228 |
| Dev slice → untouched slice (040, same candidate) | +0.0129 |
| Dev-time state → fresh production run (042, same recipe) | −0.0196 |

Against that, the margins engineered into the recipe:

| Margin | Size |
| --- | ---: |
| Scale selection above the 0.75 floor (040 dev, 1.015 → 0.75016) | +0.00016 |
| Training target vs deployment floor (0.8 × 0.93587 = 0.74870) | **−0.0013** |

The training constraint targets a value *below* the deployment gate, and the
fold selection left a margin two orders of magnitude smaller than the
documented fit→broad generalization gap. Even with no horizon mismatch at
all, this recipe passes a fresh run only when slice/run noise happens to break
favorably — as it did in 040 (offset 300 came in +0.013 favorable) and did not
in 042.

A related statistical inconsistency: NLL and KL deltas get paired
10,000-resample bootstrap intervals, but the *binding* acceptance statistic —
selected mass — is treated as a noiseless scalar compared against a sharp
threshold. The one statistic that decided 040's acceptance and 042's rejection
is the only one with no uncertainty quantification.

### 2.4 The structural problem the whole apparatus is patching

Conditional top-64 cross entropy constrains only the *shape* of the
distribution within the teacher's selected set. The student's full-vocabulary
normalizer — overall confidence — is an unconstrained direction of the
objective. The observed dynamics (mass 0.83 → 0.47 at 256 steps → 0.37 at
2,048 steps; block-25 snapshot MSE 12,647 → 105,194) show the optimizer
drifting along that null direction, and drifting further the longer it runs.

The correction stage, the one-sided mass loss, and the final-norm fold are all
post-hoc patches for confidence drift that the primary objective permits. The
patches fight the drift's *symptom* (mass) while the drift's *cause* (an
objective with a large unconstrained subspace) is retained for 2,048 steps.
Experiment 038 already observed the tug-of-war directly: "later checkpoints
fell back below the floor" — the correction pushes mass up only while the
relu is active, and the conditional term resumes depressing it the moment the
constraint is satisfied on the training batches. A one-sided constraint stops
pushing exactly at the boundary, so after any generalization gap the deployed
statistic lands systematically *below* the boundary. Undershoot is not bad
luck; it is the expected value of this design.

Experiment 043's two arms (horizon and objective) target exactly this and are
the right next questions.

## 3. Secondary methodology issues

- **The mass floor is nearly a temperature knob.** A single scalar on the
  final RMSNorm moves mass smoothly (≈0.0037 per 0.005 scale) at modest NLL
  cost. A gate that a post-hoc scalar can manufacture is not measuring model
  capability; it is measuring calibration. Rejecting a candidate that
  dominated the accepted production model on NLL, full KL, C4 (decisively),
  and packed PPL — over 0.019 of a scalar-controllable statistic — inverts
  the priority order the project presumably cares about. If the 0.75 floor
  encodes a real downstream requirement, it should be met by per-model
  calibration at deployment; if it does not, it should not be an acceptance
  gate at all. Its provenance ("established floor") deserves re-derivation.
- **Constraint distribution ≠ gate distribution.** The correction enforces
  the floor on calibration batches (50% Gemma-chat-formatted, 50% WikiText
  *train*, ≤512 tokens sampled from one document per optimizer step, 32
  documents total), while the gate measures WikiText *validation* and pinned
  C4. The correction estimates a distribution-level constraint from 32
  documents that are, by seed replay, exactly the first 32 batches the primary
  KD saw in its first epoch.
- **Serial slice reuse.** Offset 104 was 038's untouched broad gate, then
  "released" and used as 040's development slice (where 1.015 was selected),
  then used again as 042's confirmation slice. 042's confirmation therefore
  ran on the very data on which the frozen scalar was tuned. The candidate
  failed anyway, so no optimistic-bias harm materialized here, but the slice
  lifecycle policy ("released" slices re-entering service as gates) undermines
  the otherwise careful predeclaration discipline.
- **Two variables changed at once in the campaign.** Relative to the
  validated 040 state, 042 changed both the factorization (fresh, intended)
  and the conditional-KD horizon (8×, unintended). When it failed, attribution
  required a new experiment (043) that the matched-arm discipline of 037
  would have made unnecessary.

## 4. Code review: bugs looked for and not found

The following were checked and are consistent with the documented protocols:

- `topk_mass_floor_distillation_loss`
  ([distillation.py:385](../src/nanoquant/application/distillation.py)) —
  exact conditional CE above the floor, one-sided logit-space relu below,
  batch-mean semantics, token-weight handling; unit tests cover the
  above/below-floor equivalence and gradient direction claims.
- `topk_tail_distillation_loss` — teacher-mass-weighted conditional term plus
  binary mass CE, matching the Experiment 038 description of why it keeps
  pushing after the floor is met.
- `apply_gemma_final_norm_scale`
  ([final_norm_calibration.py:48](../src/nanoquant/final_norm_calibration.py))
  — correctly scales the Gemma *effective* weight `(1+w)·s − 1`, not the raw
  parameter.
- Scheduler-horizon separation (`scheduler_total_steps=128` with 32 executed
  steps) — the Document 78 replay caught the original 32-step-horizon defect
  and the fix validates horizon ≥ executed steps.
- Workflow wiring
  ([resident_workflow.py:562](../src/nanoquant/resident_workflow.py)) —
  correction warm-starts from the primary KD reference with a distinct state
  namespace; fold derives an immutable artifact.
- Gate tooling ([probe_topk_tail_mass.py](../tools/probe_topk_tail_mass.py))
  — selects top-64 by teacher logits and averages student mass per token,
  consistent with the training-side definition; no train/eval metric
  mismatch.

Two design sharp edges worth flagging (not defects): the teacher-cache
protocol normalizes `weight_decay` to 0.01 for hash compatibility, which is
correct today but fragile if decay ever becomes real; and
`_selected_parameters` selects trainable norms by class-name substring
(`"norm"`), which is convention-dependent across architectures.

## 5. What to do differently

Ordered by expected impact:

1. **Develop at deployment settings.** Any recipe intended for the production
   pipeline must be developed and validated downstream of the production KD
   horizon (2,048 steps) — or the production horizon must change. Experiment
   043's horizon arm is the right first question; if conditional-256
   dominates conditional-2048 on NLL/KL (as 037-versus-042 hints), the cheap
   fix is to cap production KD at 32 batches/epoch and re-validate the
   existing correction in its native regime.
2. **Replace the fixed 1.015 fold with a per-run calibration rule.** Derive
   the smallest scale reaching `floor + margin` on a calibration-only slice,
   freeze it before any gate is opened, and record it as part of the run
   artifact. A fixed scalar transferred across factorizations assumes
   run-to-run mass variation ≪ fold margin; the records show the opposite by
   ~100×. (Experiment 042's own decision section points this way; this review
   agrees and adds: the margin must be sized explicitly.)
3. **Size margins from measured variance.** Cross-slice and cross-run spread
   of the mass statistic is ~0.01–0.02. The calibration target should
   therefore be ≈ 0.77–0.78 to deploy a 0.75 floor, and the training-side
   `minimum_teacher_mass_ratio` should imply a value *above* the floor plus
   the observed fit→broad gap (≈ 0.85 rather than 0.8), or become an absolute
   target aligned with the deployment gate.
4. **Put a confidence interval on the binding statistic.** Report a bootstrap
   interval for selected mass on every gate and require the interval's lower
   bound to clear the floor. It is inconsistent that the decisive statistic
   is the only ungated-by-uncertainty one.
5. **Make the correction budget adaptive.** Stop the correction when a
   held-out fit monitor reaches `floor + margin` (with a step cap), rather
   than running a fixed 32 steps. A fixed budget cannot absorb the
   run-to-run variation in deficit that the pipeline demonstrably produces.
6. **Attack the drift at its source.** The durable fix is a primary objective
   without the unconstrained confidence direction: the tail-aware objective
   (043's second arm), or conditional CE plus a weak always-on normalizer
   anchor (e.g., penalizing student-minus-teacher log-normalizer drift),
   which would preserve the conditional shape that wins tasks while removing
   the pathology the correction exists to repair. If the primary objective
   holds mass near the teacher's, the correction and fold stages shrink to
   safety nets instead of load-bearing repairs.
7. **Re-justify or retire the 0.75 floor.** Decide what downstream behavior
   the floor protects (sampling quality? speculative-decoding acceptance?),
   measure that behavior directly at least once, and either derive the floor
   from it or gate on it directly. As it stands, the gate rejected the best
   model the project has produced on every capability metric it tracks.
8. **Tighten slice lifecycle rules.** A slice used for any selection decision
   should be permanently retired from gating. Confirmation slices for a fresh
   campaign should be freshly declared offsets, which the token-hash
   predeclaration machinery already makes cheap.

## 6. What the methodology gets right

For balance: predeclared gates with frozen token hashes, first-survivor rules
that prevent post-hoc tuning, refusal to tune on failed gates, paired
bootstrap intervals for NLL/KL, matched-arm forking from validated common
states, byte-exact resumable artifacts with identity binding, and the
held-out block-25 refit screen in 042 (which correctly rejected a locally
attractive refit that harmed held-out NLL/KL) are all unusually disciplined.
The failure mode here was not sloppiness — it was transferring point
estimates across an unmodeled regime change with margins too small for the
system's measured variance.
