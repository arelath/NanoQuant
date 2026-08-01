# Experiment 035 Top-K Tail-Mass KD Objective Ablation

## Status

Completed on 2026-07-31 against the retained Experiment 035 pre-KD frozen
state. This implements the first controlled experiment recommended by
[74-block25-anomaly-and-topk-tail-mass-audit.md](74-block25-anomaly-and-topk-tail-mass-audit.md).
Experiment 036 remains paused and was not modified or resumed.

## Question

The inherited global KD objective normalizes teacher and student logits only
over the teacher's selected top 64 tokens. It cannot observe how much student
probability lies outside those entries. Does adding one aggregated vocabulary
tail bucket prevent the probability-mass collapse without giving up the useful
part of KD?

## Implementation

The repeatable ablation is implemented by
`tools/probe_topk_tail_distillation.py`. It starts from Experiment 035's exact
pre-KD factorized model and exact retained top-k teacher batches. The two arms
share:

- selected parameters, optimizer, learning rate, cosine schedule, and seed;
- calibration samples and retained per-epoch token selections;
- teacher top-64 values and indices;
- held-out validation sequences 104-107 at 128 tokens;
- checkpoint and monitor cadence.

The conditional control uses the inherited 64-category cross entropy. The new
arm uses 65 categories: the teacher's 64 selected tokens plus one aggregated
tail category. The teacher's full-vocabulary log-normalizer is cached only for
the exact selected training tokens. The student's normalizer is computed
differentiably in vocabulary and token chunks, so no resident full-vocabulary
logit tensor is required.

This selected-token cache is important. The calibration inventory contains
524,288 tokens, while the 256-step probe uses at most 131,072 selected tokens.
For the 64-step probe it uses at most 32,768. Caching the exact target
selections preserves the objective and avoids computing unused teacher
normalizers.

## 64-step comparison

The first matched run used 8 retained batches per epoch for 8 epochs.

| Metric | Pre-KD start | Conditional top-64 | Top-64 + tail |
| --- | ---: | ---: | ---: |
| Held-out NLL | 4.057146 | **3.830999** | 3.906891 |
| Conditional top-64 KL | 1.160953 | **0.947367** | 1.091145 |
| Top-64 + tail KL | 1.177747 | **1.133990** | 1.147656 |
| Full-vocabulary KL | 1.234453 | **1.182263** | 1.200366 |
| Student mass on teacher top 64 | 0.882204 | 0.746330 | **0.843628** |
| Absolute tail-mass error | 0.080393 | 0.211085 | **0.114441** |
| Block-25 output NRMSE | 0.283883 | 0.417912 | **0.333293** |

At this short horizon, early conditional training gives better NLL and KL,
but it already loses 0.136 top-64 mass. The repaired objective loses only
0.039 and keeps the final-block drift much smaller. This result alone does not
justify replacing the objective.

## 256-step comparison

The second matched run used 32 retained batches per epoch for 8 epochs.

| Metric | Pre-KD start | Conditional top-64 | Top-64 + tail |
| --- | ---: | ---: | ---: |
| Held-out NLL | 4.057146 | 3.898659 | **3.868449** |
| Conditional top-64 KL | 1.160953 | **0.845605** | 1.031112 |
| Top-64 + tail KL | 1.177747 | 1.327505 | **1.111251** |
| Full-vocabulary KL | 1.234453 | 1.378077 | **1.163367** |
| Student mass on teacher top 64 | 0.882204 | 0.580109 | **0.825772** |
| Absolute tail-mass error | 0.080393 | 0.377248 | **0.132010** |
| Block-24 output NRMSE | 0.215856 | 0.281248 | **0.235067** |
| Block-25 output NRMSE | 0.283883 | 0.589610 | **0.337467** |

The conditional loss continues improving exactly what it observes while its
unobserved probability mass and late-block behavior collapse. Relative to the
same pre-KD start, it improves conditional KL by 0.315 but worsens full KL by
0.144. The tail-aware arm improves full KL by 0.071, improves NLL by 0.189,
and retains most selected-token probability mass.

The separation appears well before the original 2,048-step global KD run.
Under the conditional objective, the best monitored checkpoint is already the
first 32-step epoch: NLL 3.814150 and full KL 1.182868. Later conditional
epochs improve the training-aligned conditional KL while making the reference
metrics worse. Under the tail-aware objective, full KL continues improving
through epoch 8; held-out NLL is effectively flat from epochs 6-8 and is best
at epoch 6 (3.867173).

## Tail-mass weighting

The exact 65-category objective still moved held-out top-64 mass from 0.882 to
0.826. A decomposition into conditional-shape cross entropy plus binary
top-mass/tail cross entropy made it possible to strengthen mass calibration
without changing the optimum. Two matched 256-step arms tested 2x and 4x
binary-mass weights:

| Metric | Exact 1x | Mass 2x | Mass 4x |
| --- | ---: | ---: | ---: |
| Held-out NLL | **3.868449** | 3.923999 | 4.014416 |
| Full-vocabulary KL | **1.163367** | 1.206134 | 1.235078 |
| Student mass on teacher top 64 | 0.825772 | 0.839349 | **0.855813** |
| Absolute tail-mass error | 0.132010 | 0.118801 | **0.103709** |
| Block-25 output NRMSE | 0.337467 | 0.332815 | **0.301560** |

The initial upward weighting sweep is a real tradeoff rather than a Pareto
improvement. Increasing the mass term preserves more selected-token mass and
local block fidelity, but gives back the NLL and full-KL improvement. The 1x
objective was therefore the candidate carried into the first quality gate;
the continuation below also tests coefficients below 1x.

## Complete retained quality gate

The selected 1x/256-step checkpoint was materialized as an isolated derived
run. The materializer hard-links the immutable source graph, freezes only the
new tuned state, commits an identity-bound global-tuning artifact, and never
changes Experiment 035's active pointer. The derived run passed the complete
26-block, 708-artifact audit and reloaded through the standard factorized
loader before evaluation.

The canonical WikiText 64x128 and six-task 200-example gate produced:

| Candidate | WikiText PPL | Mean task score |
| --- | ---: | ---: |
| BF16 teacher | **96.459609** | **0.623333** |
| Pre-KD compressed | 257.485852 | 0.456667 |
| Tail-aware KD, 256 steps | 188.715654 | 0.455000 |
| Conditional KD, 2,048 steps | 221.035336 | 0.472500 |
| Conditional KD + fresh block-25 refit | **148.747976** | **0.479167** |

Tail-aware KD is a substantial language-model improvement over the pre-KD
state and the defective conditional-KD endpoint. Its six-task mean, however,
is essentially unchanged from pre-KD and below the conditional-KD arms. It is
therefore not by itself a production winner despite fixing the objective
pathology.

## Coverage, duration, and sub-1x continuation

The next probe held the optimizer at 256 total steps while changing how many
distinct calibration batches those steps covered. A separate 128-step arm
tested early stopping. All new arms used a larger 16x128 validation monitor;
their materialized checkpoints passed the same complete 26-block,
708-artifact audit before the canonical quality benchmark.

| Schedule | Total steps | WikiText PPL | Mean task score |
| --- | ---: | ---: | ---: |
| 8 epochs x 32 batches | 256 | **188.715654** | **0.455000** |
| 1 epoch x 128 batches | 128 | 207.313645 | 0.445833 |
| 1 epoch x 256 batches | 256 | 188.317820 | 0.450833 |
| 2 epochs x 128 batches | 256 | 189.854044 | 0.450000 |

The 128-step arm undertrains. At 256 steps, broader calibration coverage moves
WikiText by less than one PPL point and slightly lowers the task mean. Coverage
is not the missing quality lever for this retained cache; the original 8x32
schedule remains the better balanced schedule.

The coefficient sweep was then extended below 1x on that schedule. A value of
zero is the defective conditional-only objective, while 1x is the exact
top-k-plus-tail cross entropy. Intermediate values retain the tail bucket but
let conditional shape matching carry more of the update.

| Mass coefficient | Held-out NLL | Held-out full KL | Student top-64 mass | Block-25 NRMSE | WikiText PPL | Mean task score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 3.868449* | 1.163367* | 0.825772* | 0.337467* | 188.715654 | 0.455000 |
| **0.50** | **3.656679** | **1.233479** | 0.806770 | 0.393253 | 178.363145 | **0.462500** |
| 0.25 | 3.638683 | 1.234190 | 0.769383 | 0.430556 | **169.960621** | 0.455000 |

`*` The 1x monitor used the original 4x128 set, so its absolute monitor values
are not directly comparable with the 16x128 values below it. The quality
protocol is identical across all three rows.

The 0.5x checkpoint is the first tail-aware candidate to improve both retained
quality gates: relative to 1x it lowers PPL by 10.35 and raises the six-task
mean by 0.0075; relative to pre-KD it lowers PPL by 79.12 and raises the task
mean by 0.0058. The task gain is distributed across PIQA, HellaSwag,
Winogrande, and BoolQ rather than coming from a single benchmark. Reducing the
coefficient to 0.25 buys another 8.40 PPL but loses the entire task gain while
tail-mass error and block-25 drift continue to grow. The selected coefficient
is therefore 0.5, not the NLL-minimizing endpoint.

## Independent non-WikiText confirmation

The selected candidate also passed the non-WikiText gate requested by the
Experiment 035/036 review. On a pinned 48x512 C4 validation slice, 0.5x versus
1x improves NLL by `-0.044465` with paired 95% interval
`[-0.050693, -0.038023]` and full-vocabulary teacher KL by `-0.018483` with
interval `[-0.021759, -0.015215]`. Its C4 perplexity is 144.165461 versus
150.720479 at 1x and 191.963656 pre-KD.

The 0.25 arm reaches 133.725278 C4 perplexity but remains rejected because it
loses the six-task gain. C4 therefore confirms both that the tail-aware update
generalizes beyond WikiText and that the 0.5 selection is a balanced-quality
decision rather than the minimum-NLL choice. The exact protocol, hashes,
production integration gates, and final fresh-campaign design are recorded in
[76-tail-aware-global-kd-final-experiment-plan.md](76-tail-aware-global-kd-final-experiment-plan.md).

## Remaining block-25 marginal

The exact fresh teacher-context block-25 refit was repeated from the
tail-aware checkpoint with the same fit/validation/confirmation partitions as
the original anomaly. Its marginal reversed sign on the untouched 48x512
confirmation set:

| Metric | Tail-aware baseline | + fresh block 25 | Delta |
| --- | ---: | ---: | ---: |
| NLL | **4.539853** | 4.599563 | +0.059709 |
| Full-vocabulary KL | **1.516113** | 1.536328 | +0.020216 |

Both paired 95% intervals are entirely harmful: `[+0.055774, +0.063600]` for
NLL and `[+0.017331, +0.023166]` for KL. The fitted block-25 local prediction
NRMSE still falls from 0.299001 to 0.291015, but that local improvement no
longer transfers to the full model.

This is the strongest causal check on the block-25 anomaly. Block 25 is not an
intrinsically exceptional compression target. Under conditional KD it becomes
the final nonlinear compensator for an unobserved output-distribution error;
after the tail term reduces that error, forcing the same local correction is
counterproductive.

## Interpretation

The tail bucket fixes the identified invariance. It is not merely a diagnostic
metric: its gradient prevents the optimizer from cheaply moving all selected
student logits downward relative to the rest of the vocabulary. The much
smaller block-25 drift supports the audit's explanation that the final MLP was
absorbing a model-wide output-distribution error.

However, the experiment also shows that the conditional component carries a
useful quality signal. At 32 steps it reaches lower held-out NLL than the exact
tail-aware arm, although its full KL and mass calibration are already worse.
At 256 steps, a 0.5 mass coefficient preserves enough of that signal to improve
both retained quality gates without reverting to the zero-mass collapse. The
production policy must therefore compare checkpoints on held-out NLL and full
or tail-bucket KL, then require the secondary quality gate. Training loss or
NLL alone would select the inferior 0.25 endpoint.

## Decision

The objective diagnosis is confirmed and the bounded 0.5x candidate clears
both retained quality gates, but it is not yet integrated into the resident
workflow:

1. keep Experiment 036 paused;
2. reject a full 2,048-step conditional continuation and reject a block-25
   refit after tail-aware KD;
3. retain the exact tail bucket, but expose its binary mass term as an explicit
   coefficient and select 0.5 for the current Gemma candidate;
4. require held-out full-KL/NLL checkpoint selection followed by the secondary
   quality gate, since training loss, conditional KL, and NLL alone select
   defective or inferior states;
5. reject broader-coverage and 128-step schedule changes for this cache;
6. before changing the production default, confirm the 0.5 coefficient on a
   second model or a larger independent task sample so the 0.0075 limited-task
   gain is not treated as universal evidence; the independent C4 NLL/KL gate
   has passed, but it is not a replacement for cross-model evidence;
7. integrate the objective and checkpoint-selection policy only with explicit
   model-level evidence; keep the coefficient configurable rather than
   hard-coding Gemma's selected value.

## Evidence

- `evidence/035/experiment035-topk-tail-kd-bounded8x1/report.json`
- `evidence/035/experiment035-topk-tail-kd-bounded8x8/report.json`
- `evidence/035/experiment035-conditional-topk-kd-control-bounded8x8/report.json`
- `evidence/035/experiment035-topk-tail-kd-bounded8x32/report.json`
- `evidence/035/experiment035-conditional-topk-kd-control-bounded8x32/report.json`
- `evidence/035/experiment035-topk-tail-mass2-bounded8x32/report.json`
- `evidence/035/experiment035-topk-tail-mass4-bounded8x32/report.json`
- `evidence/035/experiment035-topk-tail-kd-bounded8x32-derived-run/topk-tail-materialization.json`
- `evidence/035/experiment035-topk-tail-kd-bounded8x32-quality.json`
- `evidence/035/experiment035-topk-tail-kd-broad1x128-monitor16-quality.json`
- `evidence/035/experiment035-topk-tail-kd-broad1x256-monitor16-quality.json`
- `evidence/035/experiment035-topk-tail-kd-broad2x128-monitor16-quality.json`
- `evidence/035/experiment035-topk-tail-mass0p5-bounded8x32-monitor16/report.json`
- `evidence/035/experiment035-topk-tail-mass0p5-bounded8x32-monitor16-quality.json`
- `evidence/035/experiment035-topk-tail-mass0p25-bounded8x32-monitor16/report.json`
- `evidence/035/experiment035-topk-tail-mass0p25-bounded8x32-monitor16-quality.json`
- `evidence/035/experiment035-tail-mass-c4-validation104-48x512.json`
- `evidence/035/experiment035-prekd-quality.json`
- `evidence/035/experiment035-tailkd-block25-teacher-context-fit380-val384-confirm412-48x512.json`

The earlier `evidence/035/experiment035-topk-tail-kd/` directory is retained
as an interrupted full-batch canary. It completed the original all-token
teacher-normalizer cache and the pre-training monitor, but no training epoch or
checkpoint. It is not used for the comparisons above.
