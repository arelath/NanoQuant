# Analysis: Tuned Frozen MLP Scale Placement — Methodology, Placement Limits, and Extensions

Review of [69-tuned-frozen-mlp-scale-placement.md](69-tuned-frozen-mlp-scale-placement.md)
and its implementation (`tools/probe_tuned_mlp_scale_refit.py`,
`tools/probe_corrected_codebook_splice.py`,
`src/nanoquant/domain/mlp_operator_refit.py`,
`tools/probe_mlp_policy_frozen_transfer.py`). Written 2026-07-31.

## Summary

The experiment is methodologically disciplined where it matters most —
predeclared tiers, disjoint windows, paired bootstrap gates, fresh-window
confirmation that correctly caught two selection-overfit blocks, and exact
byte-parity proof of the zero-bit claim. The 5.3-5.4% perplexity gain is
real and well-attested.

But the reason it works on only 5 of 26 blocks is structural, not
incidental: **every scale vector is fitted on dense-teacher activations and
deployed inside the composed student**. The fit answers "how would I correct
this block if everything upstream were exact," which is only the question
actually being asked at block 0 (near-zero upstream drift) and at the last
few blocks (near-zero downstream co-adaptation to break). Everywhere in the
middle, the calibration context and the deployment context diverge, and the
greedy one-block-at-a-time search then runs into a statistical power floor:
the evidence shows several rejected blocks with *negative* (helpful) point
estimates whose intervals simply include zero at 12 sequences.

The large gain from few blocks follows from three compounding facts: the
per-block improvement distribution is heavy-tailed and the accepted five are
its tail; the accepted placements are exactly where local gains transfer
~1:1 into the global objective; and perplexity percentages exponentially
amplify modest NLL deltas on a deeply degraded (228 ppl) baseline. The
overlay closes about 6.3% of the compression-induced NLL gap to the dense
teacher at zero bits.

The highest-leverage change is to stop fitting in teacher context: either
fit sequentially in the composed student (propagated calibration), or —
simpler and probably strictly better — make all refit-eligible scale
vectors trainable and reuse the existing global KD tuning loop restricted
to them (~570K parameters), which dissolves the placement-selection problem
entirely.

## What the implementation actually does

Verified against the code, since the doc leaves some of this implicit:

- Baseline arm = the full composed Experiment 022 model:
  `collect_splice_reconstructions` materializes all 130 factorized layers
  and `DenseKlSpliceEvaluator` splices them into the pinned teacher for KL
  against cached teacher logits.
- **All calibration activations come from the dense teacher.**
  `_capture_linear_inputs(teacher, gate_module, ...)` hooks the *teacher's*
  gate projection while running the *teacher* forward
  ([probe_corrected_codebook_splice.py:400](../tools/probe_corrected_codebook_splice.py#L400)).
  Candidate outputs are computed by applying candidate weights to these
  teacher inputs. Fit and validation windows differ only in tokens, never
  in context distribution.
- Four transforms per block, all per-channel positive multipliers folded
  into existing tensors: `operator` (coupled gate/up output scales via
  71-point gate grid + closed-form up scale), `output` (operator + down
  output scales, closed-form per-channel least squares with per-channel
  identity fallback), `input` (operator + down input scales, 50-step
  diagonal-preconditioned descent with backtracking), `joint` (input +
  output).
- Fit data is 4 sequences × 512 tokens = 2,048 token positions, fitting
  ~21,888 scale parameters per block (gate 6,912 + up 6,912 + down-in
  6,912 + down-out 1,152).
- Bounds: gate [0.25, 2], up [0.1, 8], down [0.25, 4], with hard clamping.
- Note that the within-block "student activation" insight of
  [61-student-activation-downstream-scale-refit.md](61-student-activation-downstream-scale-refit.md)
  (fit down against the *quantized* gate/up product) is carried forward
  here — but the block *input* is still the teacher's hidden state. The
  student-context correction stops at the block boundary.

## Methodology assessment

### Strengths

- **The zero-bit claim is proven, not asserted.** Byte-identical component
  replacement (599,040 → 599,040 bytes), exact logical and packed parity,
  identical packed byte count and effective BPW, and a factor-form replay
  within +0.000299 NLL of the dense reference (interval spanning zero).
- **Selection discipline caught its own failure modes.** Blocks 11 and 21
  looked good on the exact 64×128 gate (−0.78 and −0.68 ppl) and were
  rejected when fresh sequences 308-319 regressed. This is the
  winner's-curse control most compression papers lack.
- **The final gate is comprehensive**: full retained protocol including six
  task benchmarks, ruling out a perplexity-vs-task tradeoff.
- **Correct refusal of the exact retained benchmark as a selection set**
  (stated explicitly in the Decision section).

### Weaknesses

1. **Teacher-context calibration is the root defect** (detailed below).
   The doc's own framing — "again proving that local reconstruction
   improvement is not an additive placement rule" — treats non-additivity
   as a brute fact; the code shows it is a predictable consequence of
   fitting every block at an operating point it will never see.

2. **The local validation metric cannot predict composability, yet it is
   the tier-nomination rule.** "Held-out" validation (sequences 52-55) is
   held out in tokens only — same teacher context as the fit. Ranking
   blocks by best-arm local validation improvement reproduces tier 1
   exactly (0, 24, 18, 17: −26.9%, −16.7%, −16.0%, −14.5%) and tier 2
   (22, 21, 11, 20: −13.2%, −13.1%, −12.8%, −11.8%). Tier 2 contained both
   reversal culprits (block 20: +7.0 ppl; block 22: +3.6 ppl on the exact
   gate), while block 23 — the only later addition to survive fresh-window
   confirmation — ranked ~9th-11th locally. Beyond the extreme tail, local
   rank carries roughly zero information about composed outcome.

3. **The marginal gate is underpowered exactly where the answer lives.**
   From `experiment022-prekd-four-plus-block*-nll308-12x512.json`, single
   additions layered on the confirmed four-block base:

   | Block | Policy | NLL delta | 95% interval | Verdict |
   | ---: | --- | ---: | --- | --- |
   | 23 | joint | −0.0199 | [−0.0400, −0.0010] | accepted |
   | 19 | joint | −0.0095 | [−0.0279, +0.0088] | inconclusive |
   | 25 | joint | −0.0089 | [−0.0394, +0.0189] | inconclusive |
   | 21 | joint | −0.0038 | [−0.0207, +0.0124] | inconclusive |
   | 22 | operator | −0.0015 | [−0.0190, +0.0148] | inconclusive |
   | 11 | operator | +0.0039 | [−0.0083, +0.0152] | inconclusive |
   | 1 | joint | +0.0052 | [−0.0089, +0.0187] | inconclusive |
   | 13 | joint | +0.0065 | [−0.0117, +0.0241] | inconclusive |
   | 20 | joint | +0.0154 | [−0.0006, +0.0318] | harmful |
   | 4 | joint | +0.0487 | [+0.0161, +0.0825] | harmful |

   The interval half-width at 12×512 is ~±0.02 NLL; the typical marginal
   effect past the top five is ~±0.01. Blocks 19, 25, 21 may well be
   genuinely helpful. The "sharp composition boundary" the doc reports is
   partly a detection threshold, not only a physical wall.

4. **The doc under-reports a metric divergence.** On the all-26 uniform
   screen, the `output` arm regressed KL 1.647→1.773 (+7.4%) but *improved*
   NLL 4.612→4.589. The other three arms regressed both metrics, so the
   uniform-failure conclusion stands, but the completion gate is
   perplexity (NLL-based) while the screen gate is KL-to-teacher; a
   uniform placement that helps the deployed metric was rejected by the
   proxy metric without comment.

5. **Fit statistics are thin and the fits are aggressive.** 2,048 token
   positions from four contiguous WikiText articles fit ~22K parameters
   per block. The fitted multipliers saturate their bounds nearly
   everywhere: gate spans the full [0.25, 2.0] on every block, up hits
   8.0 from block 10 onward, down-input hits both 0.25 and 4.0 on most
   blocks ≥10. Saturation means the unconstrained per-channel solution is
   wild (near-dead candidate channels get huge ratios) and the hard clamp
   is doing the regularization. The per-channel identity fallback in
   `fit_linear_output_scales` checks improvement on *fit* data only, so
   channels that overfit the 2,048 tokens keep their scales.

6. **Protocol-sensitivity is observed but not investigated.** Pre-KD, the
   four-block policy improved 512-token functional NLL by −0.0346 but the
   128-token retained gate by only −0.0036 (10× discrepancy); post-KD the
   two protocols agree within ~1.6×. Effects this context-length-sensitive
   deserve a note in the selection rule, since screens (512) and the
   retained gate (128) sit on opposite sides of it.

## Why it only works on a few blocks

Fitting block *k*'s scales toward the teacher's local function has one
benefit and two costs, and the accepted placements are exactly where the
benefit survives:

- **Realized benefit decays with upstream drift.** The scales are optimal
  for teacher inputs. At eval, block *k* sees student activations carrying
  all upstream compression error (composed KL is 1.65 nats — a heavily
  perturbed regime). The correction is computed at the wrong operating
  point; per-channel gains of up to 8×, calibrated on the wrong
  distribution, can amplify error components rather than cancel them.
- **Downstream co-adaptation cost decays with depth.** Experiment 022's
  factors are globally KD-tuned: the network sits near a local optimum of
  the *composed* KL objective, meaning downstream layers already partially
  compensate each block's residual error pattern. Refitting block *k*
  toward its *layer-local* teacher function changes the error pattern that
  25−*k* downstream blocks were tuned against. That is why every block
  improving locally can coexist with uniform composition regressing KL by
  7-20% — near a global-objective optimum, most local-objective moves go
  uphill globally.

The block-position pattern falls out directly:

- **Block 0** is the only block whose fit context ≈ eval context (its MLP
  input is one compressed attention sublayer away from the exact shared
  embedding stream). Its local improvement — also the largest in the model
  (−27% operator RMSE vs a −3 to −7% typical) — transfers almost fully.
- **Late blocks (17, 18, 23, 24)** have the *most* upstream drift, but
  their corrections feed nearly directly into the final norm and logits:
  there is almost no downstream co-adapted machinery to break, and no
  re-mixing between the correction and the loss. Small net benefit, which
  is why several late blocks sit just under the significance floor.
- **Middle blocks** pay both costs at once: heavily drifted inputs
  (discounted, sometimes negated benefit) and many downstream co-adapted
  blocks (large breakage cost). Block 4's significantly *harmful* marginal
  (+0.049 NLL) is the archetype. The marginal-delta table above is close
  to monotone in depth — every negative point estimate is at block ≥19
  (plus block 0 accepted earlier), every clearly positive one at ≤20.

Secondary contributors: bound-saturated fits are most extreme precisely in
the middle/late-middle blocks; and the greedy protocol tests candidates
against a base that keeps changing the very input distributions the
candidate scales were fitted for.

## Why the results are so good despite touching only five blocks

1. **The improvement distribution is heavy-tailed and the five are the
   tail.** Block 0's local error reduction is 4-8× the typical block's;
   17/18/24 are the next outliers. Most of the *transferable* headroom
   lives in those few placements.
2. **The accepted placements convert local gain to global gain at ~1:1.**
   By construction (see above), block 0 and the late blocks are where the
   local objective and the composed objective align; the middle blocks'
   larger aggregate headroom is locked behind the context mismatch.
3. **Exponential amplification on a degraded baseline.** The retained gate
   NLL delta is −0.0545, i.e. exp(−0.0545) ≈ −5.3% perplexity. Against the
   dense teacher, the compression gap is ln(228.59/96.46) ≈ 0.863 nats, so
   the overlay closes ≈ 6.3% of the quality gap — substantial for zero
   bits, but not mysterious.
4. **It is an objective upgrade, not new capacity.** The touched
   `scale_pre`/`scale_post` slots were originally set by weight-space
   reconstruction fitting inside the factorization pipeline. The refit
   re-aims the same ~300K BF16 values at a *functional* objective
   (teacher activation matching through the SiLU nonlinearity). Zero-bit
   gains of this size are the measure of how misaligned weight-space
   Frobenius error is with what the network actually needs — consistent
   with the sensitivity-weighting theme in
   [33-error-budget-driven-quality-improvements.md](33-error-budget-driven-quality-improvements.md).
5. **Post-KD restart compounded the effect.** The same placement family
   moved the retained gate −0.36% pre-KD but −5.3% post-KD. Global KD
   shrinks upstream drift, so teacher-context calibration becomes *more
   faithful* after KD — indirect confirmation of the context-mismatch
   mechanism, and a hint that scale refits and global tuning are
   complementary rather than redundant.

## What to do differently

Ranked by expected leverage:

1. **Fit the scales under the global objective directly (recommended).**
   All refit-eligible scale vectors total ≈ 26 × 21,888 ≈ 570K
   parameters. Make them the only trainable parameters and run the
   existing global KD tuning loop (KL vs teacher on the KD token budget,
   existing held-out gates) for a modest number of steps, with decay
   toward the identity. This makes the composed objective the fit
   objective, so the "which blocks, which arm" search — the entire
   placement problem, its power limits, and its selection-overfit risk —
   disappears: blocks whose scales shouldn't move stay near identity.
   Zero-bit is guaranteed by construction (same tensors, same encoding
   path already proven in this experiment). Given that five crudely
   fitted blocks recovered 5.3%, a globally consistent fit of all 26
   should be expected to meet or beat it.

2. **If per-block fitting is retained, calibrate in composed context.**
   Capture block inputs from the *student* (current composed candidate,
   including previously accepted refits) and fit scales to map candidate
   outputs toward the teacher's outputs at the same token positions,
   sweeping blocks in order. This is the propagated/sequential calibration
   used by GPTQ-family pipelines and is the cross-block completion of what
   [61-student-activation-downstream-scale-refit.md](61-student-activation-downstream-scale-refit.md)
   already does within a block. It converts each refit from "assume
   upstream is exact" into "correct what actually arrives," which is
   precisely the property that made block 0 work. Middle blocks should
   become viable. Cost: one student forward per fit window per sweep —
   the capture machinery already exists.

3. **Re-fit the accepted set in its own context.** Even the confirmed
   five-block overlay is internally inconsistent: each block's scales were
   fitted assuming the other four were absent. One or two coordinate
   sweeps (re-fit each accepted block with the others installed) is free
   in bits and should recover additional KL.

4. **Regularize the per-channel solutions instead of clamping them.**
   Replace hard bounds with ridge shrinkage toward identity,
   s = (⟨t,c⟩ + λ)/(⟨c,c⟩ + λ), and add per-channel cross-fit acceptance
   (fit on half the window, keep only channels that improve the other
   half). Bound saturation on nearly every block is the current symptom;
   a per-block damping factor s ← 1 + λ(s−1), with λ chosen on a small
   composed screen, is the cheapest retrofit and would likely rehabilitate
   some middle-block placements on its own.

5. **Power the marginal gate to the effect size it is trying to detect.**
   Marginal effects past the top five are ~0.01 NLL; the 12×512 screen
   resolves ~0.02. Either widen marginal screens to 24-48 sequences, or
   test predeclared *groups* (e.g. {19, 21, 25} after context-corrected
   refitting) so the summed effect clears the same interval. The current
   protocol conflates "does not help" with "cannot be detected at n=12."

6. **Weight the local objective by downstream sensitivity.** Where local
   fitting is kept, weight the output-error Frobenius norm by an estimate
   of the final-KL sensitivity to that block's output (diagonal
   Fisher/Hessian over a few sequences — machinery that exists in the
   codebase but is currently dormant). This aligns local fits with the
   global metric and should widen the set of transferable placements.

7. **Track NLL alongside KL at every gate, and both context lengths.**
   The uniform `output` arm's KL/NLL disagreement and the 10× pre-KD
   discrepancy between 512-token screens and the 128-token retained gate
   both indicate the proxy metrics are not interchangeable in this
   zero-bit regime. Report both; gate on the one the completion criterion
   actually uses.

8. **Alternate scale refits with global KD.** The post-KD restart was 15×
   more effective than the pre-KD study on the retained gate. Whatever
   fitting scheme is chosen, iterate: scales → global KD → scales. Each
   pass shrinks the drift that limits the next.

## Evidence examined

- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-all26-fit48-val52-kl272-12.json`
  (per-block validation RMSE, multiplier extrema, uniform arm KL/NLL)
- `evidence/m4/sign-word-codebook-probe/tuned-mlp-refit/experiment022-prekd-four-plus-block{1,4,11,13,19,20,21,22,23,25}-*-nll308-12x512.json`
  (marginal composed deltas and intervals)
- Implementation: [probe_tuned_mlp_scale_refit.py](../tools/probe_tuned_mlp_scale_refit.py),
  [probe_corrected_codebook_splice.py](../tools/probe_corrected_codebook_splice.py)
  (capture/fit/compose helpers),
  [mlp_operator_refit.py](../src/nanoquant/domain/mlp_operator_refit.py)
  (scale solvers),
  [probe_mlp_policy_frozen_transfer.py](../tools/probe_mlp_policy_frozen_transfer.py)
  (overlay transfer evaluation).
