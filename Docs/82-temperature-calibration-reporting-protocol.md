# Temperature Calibration Reporting Protocol

## Purpose

Experiments 037-047 showed that a scalar final-norm fold can improve NLL and
selected top-k mass without changing the model's ordering of logits. That is a
calibration effect, not evidence that compression preserved more capability.
Experiment 048 therefore reports raw and temperature-fitted results separately
and never allows temperature fitting to rescue a failed raw capability gate.

This protocol is frozen before an Experiment 048 evaluation slice is reserved.
It does not open or evaluate new data.

## Quantity being fitted

For one student arm, let `z` be its raw next-token logits. Fit one positive
logit scale `alpha` and report the equivalent temperature `T = 1 / alpha`.
The calibrated distribution is `softmax(alpha * z)`. Gemma 3 1B has no final
logit soft-capping, so this is the same distributional operation as a global
final-norm scale; the report nevertheless treats it as an evaluation-only
calibration unless a later campaign explicitly materializes the fold.

The teacher remains raw and fixed. Each student arm receives its own fitted
scale. Sharing one scale would not distinguish arm-specific miscalibration;
fitting on different data would make the arms incomparable.

## Data roles and ordering

The adaptive checkpoint selector continues to use only raw C4 NLL and raw
full-vocabulary KL under its already-frozen rule. Temperature is not fit while
checkpoint selection is in progress.

After the selector decision is immutable:

1. reuse that campaign's calibration-only C4 selection slice;
2. fit one scale independently for the uncorrected baseline and the selected
   arm using the same ordered tokens and protocol;
3. freeze a scale receipt bound to the model/checkpoint identities, token hash,
   and fitting protocol;
4. apply those frozen scales to the untouched final C4 confirmation slice;
5. report raw and calibrated metrics side by side without refitting;
6. keep raw WikiText, task, packed, and GGUF results as the promotion gates.

No final-confirmation token may contribute to the fit. A rejected checkpoint,
failed final gate, or interrupted evaluation cannot trigger a replacement
scale rule or fitting range.

## Fitting objective and deterministic solver

The sole fitting objective is exact ground-truth next-token NLL on the
calibration-only slice. KL-to-teacher is evaluated after fitting but does not
choose a second scale. This gives each arm one conventional calibration
parameter rather than a different oracle for every reported metric.

NLL is convex in positive logit scale. The solver uses deterministic bounded
Newton updates over the complete ordered slice:

- initial scale: `1.0`;
- allowed scale interval: `[0.5, 1.5]`;
- maximum update passes: `4`;
- convergence tolerance: absolute scale change at most `1e-4`;
- accumulation: float64 scalar sums in fixed token order;
- softmax/log-sum-exp evaluation: float32, with the existing bounded token
  chunks;
- update: `alpha <- clamp(alpha - gradient / hessian, 0.5, 1.5)`;
- a non-finite derivative, non-positive aggregate Hessian, or unconverged
  boundary solution fails closed rather than silently accepting scale `1.0`.

For label `y`, the per-token derivative and Hessian are:

```text
d NLL / d alpha  = E_softmax(alpha*z)[z] - z[y]
d2 NLL / d alpha2 = Var_softmax(alpha*z)[z]
```

Each pass recomputes logits from the immutable loaded arm. It does not retain a
full vocabulary-logit cache. This deliberately trades several output-head
passes for bounded memory and an auditable exact full-vocabulary objective.

## Required receipt

The immutable fit receipt records:

- source model and revision;
- frozen resident model, plan, and selected global-tuning/checkpoint identity;
- calibration slice ID, registry hash, token hash, sample count, and sequence
  length;
- solver version and every fixed setting above;
- every iteration's scale, token count, mean NLL, gradient, Hessian, proposed
  update, and clamped update;
- final scale and equivalent temperature;
- raw and fitted calibration NLL;
- convergence and boundary status.

Resume must reproduce the receipt exactly or fail. Baseline and candidate
receipts must have identical data and solver identities before a paired report
is emitted.

## Final report interpretation

For baseline and candidate, report on the untouched C4 confirmation slice:

- raw NLL and full-vocabulary KL with paired intervals;
- temperature-fitted NLL and full-vocabulary KL with paired intervals;
- raw/fitted selected mass as calibration diagnostics only;
- teacher top-1 agreement once, because positive scalar temperature cannot
  change it except at exact ties;
- the raw candidate-minus-baseline effect, the fitted effect, and the portion
  of the raw marginal removed by comparable per-arm calibration.

Promotion still requires the predeclared raw C4 gates and the independent task
guardrail. A calibrated improvement supports a deployment-calibration choice;
it is not evidence for a better compression mechanism. A raw improvement that
survives comparable fitting is stronger evidence of preserved predictive
structure.

## Implementation gates

Before the fresh campaign:

- unit-test the derivative and Hessian against PyTorch autograd;
- prove the bounded solver recovers a synthetic known scale;
- test boundary, non-finite, non-positive-Hessian, and non-convergence failures;
- bind fit receipts to exact arm and slice identities;
- add interruption/resume equality for calibration passes;
- extend the C4 evaluator without changing the frozen raw checkpoint selector;
- run focused tests, repository Ruff, mypy, and the full suite.

The implementation now provides the deterministic full-vocabulary statistics
and bounded Newton solver, identity-bound resumable fit receipts, the
calibration-slice/selector-decision checks, and opt-in receipt consumption by
the final C4 evaluator. The evaluator preserves its existing raw result and
primary gate, then emits fitted NLL/KL, top-1, teacher-top-k mass, raw/fitted
student mass, paired intervals, and the portion of the raw NLL/KL marginal
removed by comparable calibration. Naked scale values are not accepted by the
final evaluator; both primary arms must supply completed receipts from one
shared retired selection slice. The implementation gate passes with 30 focused
tests, 1,216 repository tests, 49 expected deselections, clean Ruff, and clean
mypy.
