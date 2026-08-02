# Experiment 051: Maximum Exhaustive Sign Coverage Under 20 Minutes

## Question

How far can Experiment 050's exact discrete sign coverage be scaled while
keeping one complete, numerically credible run below 20 minutes on the current
machine?

## Discrete scaling law

For a square full-rank `N x N` factorization, fixing the exact row, column, and
component sign gauges leaves

```text
2^((N - 1)(2N - 1))
```

sign pairs in the current enumeration. Component permutations still duplicate
represented matrices, but removing that symmetry does not make the continuous
scale optimization free and would require a substantially different canonical
orbit generator.

| Size | Gauge-reduced sign pairs | Increase from prior size |
|---:|---:|---:|
| 3x3 | 1,024 | - |
| 4x4 | 2,097,152 | 2,048x |
| 5x5 | 68,719,476,736 | 32,768x |

The probe was extended to generate this space in contiguous batches, so 4x4 no
longer requires materializing every candidate simultaneously. Singular
mid-scale normal equations are handled per candidate with a stronger ridge and
fail-soft fallback rather than aborting or silently dropping a sign pair.

## Benchmark and selected protocol

Hardware and objective match Experiment 050: pinned Gemma-3-1B, block-12
`q_proj`, full-Fisher input/output importance with shrinkage 0.6, and CUDA 0.

The initial complete 4x4 benchmark used one continuous scale start and eight ALS
passes for every one of the 2,097,152 sign pairs:

- exhaustive-search time: 0.830 seconds;
- total command time including model/calibration loading and four 800-step ADMM
  controls: 12.6 seconds;
- production NRMSE: 0.339250;
- exhaustive NRMSE: 0.129592.

The benchmark showed that continuous search depth, not sign generation, could
use most of the time budget. A proposed 32-start/256-pass arm crossed 20 minutes
and was explicitly stopped. The selected deep protocol halves its multistart
width:

- every 4x4 sign pair, batched 65,536 at a time;
- 16 continuous scale starts per sign pair;
- 256 ALS passes per start;
- 32 production ADMM seeds and 32 production scale passes for the real control;
- 800 ADMM outer iterations and five inner iterations.

This evaluates approximately 33.6 million sign/start combinations and more than
8.5 billion ALS pass-candidates while retaining complete discrete coverage.

## Timing

| Run | Targets | Exhaustive search per target | Total command time | Under 20 minutes? |
|---|---:|---:|---:|---:|
| Real Fisher crop | 1 | 330.93 s | 465.5 s (7:46) | Yes |
| Gaussian + planted confirmation | 2 | 336.61 / 336.51 s | 761.4 s (12:41) | Yes |
| Rejected 32-start/256-pass sizing attempt | 1 unfinished | - | exceeded 1,204 s | No; stopped |

The two-target confirmation is the strongest complete invocation and remains
7 minutes 19 seconds inside the requested ceiling.

## Results

| Target | Best production NRMSE | Exhaustive NRMSE | Squared-error reduction |
|---|---:|---:|---:|
| Real Fisher crop | 0.298857 | 0.124561 | 82.63% |
| Gaussian | 0.108868 | 0.108625 | 0.445% |
| Format-generated, known optimum zero | 0.087624 | 0.00000052 | approximately 100% |

The planted control validates that the batched exhaustive procedure can recover
an essentially exact same-format solution. The Gaussian result also prevents an
overgeneralization: production ADMM can already be very close on some 4x4
targets. The real Fisher crop, however, independently extends Experiment 050's
finding that a dramatically better same-format solution can exist at tiny full
rank.

The deep real oracle improves further over the shallow exhaustive result
(NRMSE 0.129592 to 0.124561), showing that scale multistart/depth matters even
after all signs are covered.

## Why 5x5 is not reasonable under the bound

At the measured shallow 4x4 throughput of roughly 2.53 million sign pairs per
second, directly visiting 68.7 billion 5x5 gauge-reduced pairs would take about
7.5 hours before applying the stronger continuous search. Applying the selected
16-start/256-pass depth would move the estimate from hours to many days.

Component-permutation quotienting could reduce duplicate representations, but
even the optimistic division by `5!` does not make the validated deep search fit
under 20 minutes, and correct orbit handling has fixed-point cases that make the
actual count and generator more complicated than simple division. Trading the
validated continuous oracle for a very shallow, newly canonicalized 5x5 sweep
would produce a larger nominal size but weaker evidence.

Therefore **4x4 is the largest reasonably complete exhaustive-sign oracle under
20 minutes on this hardware**. Larger experiments should use the component-
window enumeration proposed by Experiment 050 rather than claim global sign
coverage.

## Evidence and implementation

- Shallow sizing benchmark:
  `evidence/051/exhaustive4-benchmark-v5-s1-p8.json`
- Deep real Fisher result:
  `evidence/051/exhaustive4-deep-s16-p256.json`
- Deep Gaussian/planted confirmation:
  `evidence/051/exhaustive4-synthetic-confirmation-s16-p256.json`
- Tool: `tools/probe_tiny_factorization_optimality.py`
- Unit coverage includes exact 3x3 batch partition equivalence and the 4x4/5x5
  configuration counts.
