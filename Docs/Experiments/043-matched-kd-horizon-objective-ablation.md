# Experiment 043: Matched KD Horizon and Objective Ablation

## Status

Complete. Both the horizon and objective hypotheses passed on a newly reserved
WikiText interval, tail-256 passed a newly reserved C4 interval, and the full
six-task guardrail found no established regression.
Experiment 042 is complete and rejected as a production default.
Its block-25 refit branch is also rejected on the held-out screen, so no refit
or factor overlay is part of this experiment.

## Question

Experiment 042 changed two scientifically important conditions relative to the
retained state on which the correction was developed:

1. its primary conditional top-64 KD ran 2,048 optimizer steps, whereas the
   Experiment 037 matched arms ran 256; and
2. it retained the conditional objective even though the matched Experiment
   037 result showed that a top-64-plus-tail objective reduces selected-mass
   collapse and the final-block snapshot pathology.

Experiment 043 isolates those axes without another factorization. Both arms
start from the exact fresh Experiment 042 pre-KD frozen state, use the same
calibration tokens and cached teacher selections, execute eight epochs capped
at 32 batches per epoch, and use a 256-step cosine horizon.

## Frozen source and arms

Source run:
`evidence/042/042-low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it`.
The loader must ignore all active Experiment 042 global-tuning artifacts and
start from its committed resident factorization. The frozen identity is:

- config `sha256:08d2e59056ddc6c4878d847b0a8802fd2b3b194bbc7b6567052a83cfd96def0b`;
- model `sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130`;
- plan `sha256-14ae9d3f5f11ee9c07ae1631fef39330289d9032a75c74f7751ff76c1f50c846`;
- effective BPW `1.0244947118`.

The two arms are fixed:

| Arm | Objective | Epochs | Batches/epoch | Total steps | Tail weight |
| --- | --- | ---: | ---: | ---: | ---: |
| conditional-256 | conditional teacher top-64 | 8 | 32 | 256 | 0 |
| tail-256 | teacher top-64 plus aggregated tail | 8 | 32 | 256 | 0.5 |

No mass-floor correction, final-norm scale, temperature fit, MLP refit, or
other post-KD transformation is allowed. Checkpoints are committed after every
epoch and each completed arm is materialized as an isolated loadable derived
run with an explicit global-tuning reference.

## Measurements and gates

The per-epoch 4x128 validation monitor at offset 104 is diagnostic only; it was
not used for early stopping or hyperparameter selection. The final comparison
uses the newly reserved 48x512 WikiText validation slice at offset 348, token
hash
`sha256:a1ae1fe5d43b570e7472c6a12b891e162628083c10d292a730f5360bcd79a0e6`,
and registry identity `experiment043-final-wikitext-validation348-48x512`.
It reports causal NLL, full-vocabulary KL, top-64-plus-tail KL, selected mass,
absolute sequence-bootstrap intervals, and paired 10,000-resample intervals.

The original draft incorrectly described offset 300 as untouched. That slice
accepted Experiment 040 and was later opened by Experiment 042 and mean-only
Experiment 043 exploratory probes. Those exploratory results are retained as
evidence but cannot select or accept either arm. Offset 300 is permanently
retired under the slice-lifecycle rules in
[`../81-experiment-methodology-guardrails.md`](../81-experiment-methodology-guardrails.md).
This amendment was made before any arm was evaluated on offset 348.

The final evaluator includes three immutable arms and verifies their observed
step counts before inference: Experiment 042 conditional-2048 (2,048 steps),
conditional-256 (256 steps), and tail-256 (256 steps). Thus the horizon and
objective comparisons use one common fresh token inventory without silently
changing the training budget.

The experiment asks two separate questions:

1. **Horizon:** conditional-256 must improve NLL and full KL over Experiment
   042 conditional-2048 on the same factorization. This establishes whether
   the canonical long horizon creates the damage later repaired by the
   correction.
2. **Objective:** tail-256 must improve both NLL and full KL by at least 0.02
   nats over conditional-256, with both paired upper bounds below zero. It must
   also improve top-64-plus-tail KL and reduce the block-25 snapshot/monitor
   pathology. Selected mass is reported with an absolute 95% interval as a
   mechanism and calibration diagnostic, not used as a capability gate.

If tail-256 passes those WikiText gates, run the pinned C4 48x512 paired NLL/KL
gate without a norm fold on newly reserved offset 152, token hash
`sha256:49d347cadb5df2d9b5aa5b16dd106f8932534c3696e2614032591703da8cd1eb`,
rather than reusing the retired offset-104 interval. Then run the six-task 1,000-example comparison and
treat its paired interval as a guardrail, not a selection objective. A
200-example task mean cannot accept or reject an arm.

Advancement requires both questions to resolve coherently. If the horizon gate
passes, 256 steps becomes the explicitly proposed deployment horizon and the
tail-256 arm is regime-matched to that proposal. If the horizon gate fails,
tail-256 cannot be promoted even if it wins the objective comparison; retaining
the 2,048-step production horizon would require a new matched conditional-2048
versus tail-2048 experiment.

No packed export or GGUF is justified for a rejected analysis arm. A surviving
arm must later be integrated as the production primary objective and complete
the full packed/GGUF quality lifecycle before becoming a default.

## Results

The immutable epoch-8 checkpoints are:

- conditional-256:
  `sha256-343fbe53e33ff4dd729210c36864742375068a28c96da92c10bf3a98b32844a8`;
- tail-256:
  `sha256-664039dca82467b09d59c22b8d073f33f6d0dd98c619cff28ad2de450b9263ef`.

Both have exactly 256 optimizer steps. On the diagnostic offset-104 monitor,
tail-256 reduced NLL from 3.97596 to 3.84850, full KL from 1.37072 to
1.10930, and block-25 output NRMSE from 0.59782 to 0.39120. Selected mass rose
from 0.56804 to 0.78707.

### Fresh WikiText gate

Evidence:
`evidence/043/experiment043-final-validation348-48x512-kd-quality.json`.
The exact 48x512 validation-offset-348 token hash is
`sha256:a1ae1fe5d43b570e7472c6a12b891e162628083c10d292a730f5360bcd79a0e6`.

| Arm | NLL | Full KL | Top-64 + tail KL | Selected mass | Selected-mass 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| conditional-2048 | 4.52503 | 1.77022 | 1.68353 | 0.39793 | [0.38824, 0.40762] |
| conditional-256 | 4.36973 | 1.55402 | 1.48431 | 0.49258 | [0.48293, 0.50244] |
| tail-256 | **4.22362** | **1.26144** | **1.19545** | **0.71559** | [0.70765, 0.72331] |

The horizon comparison, conditional-256 minus conditional-2048, improved NLL
by 0.15529 nats (95% paired interval `[-0.16612, -0.14444]`) and full KL by
0.21619 (`[-0.22565, -0.20707]`). The horizon gate passes.

The objective comparison, tail-256 minus conditional-256, improved NLL by
0.14611 nats (`[-0.15984, -0.13148]`), full KL by 0.29258
(`[-0.30541, -0.27936]`), and top-64-plus-tail KL by 0.28887
(`[-0.30148, -0.27597]`). It exceeds both 0.02-nat minimum deltas and both
primary upper bounds are below zero. The objective gate passes. Conditional
top-64 KL worsened by 0.14224; this is reported as the expected trade from a
conditional-only shape metric toward a better normalized full distribution,
not hidden by the aggregate decision.

### Fresh C4 gate

Evidence: `evidence/043/experiment043-c4-validation152-48x512.json`. The exact
offset-152 token hash is
`sha256:49d347cadb5df2d9b5aa5b16dd106f8932534c3696e2614032591703da8cd1eb`.
Tail-256 improved C4 NLL by 0.06702 nats (95% paired interval
`[-0.08044, -0.05320]`) and full KL by 0.21810
(`[-0.23076, -0.20544]`). The C4 gate passes.

### Six-task guardrail

Evidence:

- `evidence/043/experiment043-conditional256-tasklimit1000-quality.json`;
- `evidence/043/experiment043-tail256-tasklimit1000-quality.json`;
- `evidence/043/experiment043-conditional-vs-tail256-tasklimit1000-paired.json`.

Across PIQA, ARC Easy, ARC Challenge, HellaSwag, Winogrande, and BoolQ at 1,000
examples each, the unweighted task mean changed from 0.45267 to 0.45100. The
candidate-minus-baseline delta is -0.00167 with paired task-stratified 95%
interval `[-0.00733, +0.00417]`; no regression is established. The standard
64x128 WikiText PPL also improved from 197.4020 to 184.2194. This guardrail
passes under its predeclared non-regression interpretation.

## Decision

Experiment 043 accepts both changes as one coherent production proposal:
explicitly cap primary global KD at 32 batches per epoch for 8 epochs (256
observed steps), and replace conditional-only top-64 KD with top-64 plus the
aggregated tail objective at weight 0.5. Do not carry forward the mass-floor
correction, fixed 1.015 fold, temperature fit, or block-25 refit.

This is still an analysis-arm decision, not production completion. The next
experiment must apply the policy in a fresh complete compression run and pass
strict resident validation, packed reload, GGUF export, and the full quality
lifecycle before changing the default recipe.

## Interpretation constraints

- Block snapshots are cumulative, unweighted, unnormalized hidden-state MSE
  against BF16 teacher block outputs. They are not independent per-block gains
  and must not be added across blocks.
- Absolute results against the common pre-KD state and BF16 teacher take
  priority over marginal improvements against an unstable prior stage.
- Global temperature calibration, if studied later, must be reported
  separately from compression capability. It cannot rescue this experiment.
- The failed Experiment 042 confirmation slices are evaluation-only here; no
  coefficient, horizon, or checkpoint may be chosen from them.
