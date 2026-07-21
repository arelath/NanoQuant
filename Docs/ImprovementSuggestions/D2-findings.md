# D2 Campaign Review — Corrected Findings (2026-07-21)

**Scope:** Post-hoc audit of the Experiment 020 D2 (KL-calibrated allocation) arms on
Gemma-3-270M. This revision corrects the original 2026-07-20 review after tracing the complete
profile and reconstruction-planning data flow.

## Verdict

The exact-unit D2 arm did have a metric-semantics bug, but it was **not** a double-square bug.
The profile stored the square root of an absolute weighted squared error, and the sensitivity
calculation squared that amplitude correctly. The actual defect was that the design's normalized
functional-error coefficient was estimated from an absolute error while the allocator optimized a
different, raw absolute error measure.

The implementation now uses one explicit quantity throughout KL-calibrated allocation:
`weighted_normalized_squared_error`. Old profiles fail closed under evaluator v3 / profile schema 2.
The retained evaluator-v2 profiles and failed D2 arms remain valid historical negative evidence, but
they are not compatible inputs to the corrected planner.

## Finding 1 — Fixed: inconsistent error semantics

The original review missed the square root in `DenseKlSpliceEvaluator.__call__`. Before the fix:

1. `export_weighted_error` was an absolute weighted squared-error energy.
2. `kl_splice.py` stored `sqrt(export_weighted_error)` in the profile's `weighted_error` field.
3. `kl_budget.py` calculated `KL / weighted_error**2`.

Therefore the old calculation was `KL / E²_abs`, not `KL / E⁴_abs`. Across the 90 retained unit
arms, the absolute amplitudes span `0.32465 → 37.29165`; squaring them creates a denominator-scale
range of about `13,194×`, not `10⁶×`.

The normalization problem was nevertheless real. The originating anatomy design defines
`s_u = KL_u / E_func,u²`, where `E_func` is a relative functional-error amplitude. The closest
persisted run metric is `export_weighted_normalized_error`, already a normalized squared-error
energy. In the Experiment 016 source artifacts it spans `0.06988 → 0.44764`. Using it moves the
largest measured coefficients away from late, small-matrix `o_proj` units and toward the high-KL
block-3 `down_proj` and blocks 10–11 QKV units.

### Implemented correction

- KL profile unit arms now store an explicitly named `weighted_normalized_squared_error`; there is
  no implicit square root.
- Exact sensitivity is `KL / weighted_normalized_squared_error`.
- KL-calibrated reconstruction allocation predicts the same normalized weighted squared-error
  quantity. Ordinary reconstruction-aware allocation retains its prior raw-error behavior.
- The evaluator is version 3, the profile payload is schema 2, rank-probe evidence is schema 2,
  and the resident algorithm version was incremented. Stale profiles and probes fail closed.

No heuristic upper bound such as `error <= 2` is used. A normalized squared error is dimensionless
but can legitimately exceed 2. Positivity, finiteness, schema identity, source identity, and exact
unit coverage are the valid guards.

The previous suggestion to check that `s × E²` reproduces KL was also removed: when `s` is computed
from the same pair, that check is algebraic and provides little independent protection. The useful
test is a held-out, end-to-end KL measurement of the resulting plan.

## Finding 2 — Fixed observability; 12 sequences were insufficient for this gate

The evaluator-v2 arms contained only aggregate metrics for 12 sequences / 6,132 scored tokens.
That was enough to report a point estimate but not enough to estimate uncertainty. The difference
between the trust-arm KL delta, WikiText NLL delta, and BoolQ delta does not by itself prove sampling
error: those are different metrics on different evaluation samples.

Evaluator v1 versus v2 was also not a repeatability experiment. Evaluator v2 intentionally changed
shared-QKV reconstruction semantics from factor-first BF16 reconstruction to committed-precision
reconstruction followed by a final cast.

Evaluator v3 now persists NLL, KL, and token count per sequence for every arm. The campaign's D2
adoption gate performs a deterministic paired bootstrap over those sequences and requires the upper
95% confidence bound to meet the predeclared 1% relative-improvement threshold. This makes the gate
data-driven.

The corrected exact-unit candidate measured the following paired results:

| Slice | Baseline KL | Candidate KL | Relative delta | Paired 95% interval | 1% gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 12×512 (6,132 tokens) | 2.745538 | 2.733696 | −0.431% | [−2.710%, +1.787%] | Fail |
| 48×512 (24,528 tokens) | 2.817520 | 2.737990 | −2.823% | [−4.287%, −1.393%] | Pass |

The 12-sequence slice could not distinguish the required improvement from a regression. The
48-sequence slice resolved this candidate decisively, with the upper confidence bound beyond the
predeclared 1% improvement threshold. This proves that 12 sequences were insufficient for this
decision; it does not establish 48 as a universal minimum, so future gates continue to use the
measured interval rather than a hard-coded sample-count claim.

The retained full quality benchmark remains the final arbiter.

## Finding 3 — Expected headroom and structural direction were overstated

This conclusion remains supported:

- ErrorAnatomy's allocation opportunity was measured from uniform ranks on Gemma-3-1B, while
  Experiment 016 was already reconstruction-allocated on Gemma-3-270M.
- The fresh 270M profile differs materially from the 1B anatomy: `down_proj` is the worst type,
  and the depth profile is flatter.
- The type×block arm moved ranks in the intended aggregate directions but regressed full KL by
  3.2167%. The 0.25 trust arm reduced that regression to 0.168% on the old point-estimate harness,
  but it cannot be called statistically neutral because evaluator v2 retained no per-sequence data.

These results establish that large allocation steps and cross-model expectations were unsafe. The
corrected 0.25-trust exact-unit arm moved MLP rank by `−480`, attention rank by `+576`, blocks 0–10 by
`+384`, and blocks 11–17 by `−288`. It therefore reversed the inherited type direction while still
improving both held-out KL and retained WikiText NLL. The Gemma-1B-derived MLP-gain/attention-drain
expectation is now a historical diagnostic, not an adoption condition for the fresh 270M profile.

## Corrected measurement and disposition

The evaluator-v2 per-unit KL scalar measurements remain useful diagnostic evidence. They cannot be
used directly as a planner profile because the old artifact does not store the corrected denominator
or per-sequence statistics. The normalized denominators can be recovered from all 90 retained source
units, but a formal adoption attempt must generate a new versioned profile rather than rewriting old
evidence.

That measurement is now complete:

1. The evaluator-v3 baseline and candidate profiles each contain all 114 arms and explicit
   per-sequence statistics.
2. The corrected 0.25-trust candidate completed and freshly validated 18/18 blocks and 492 transitive
   artifacts at effective BPW `1.022967310`, below Experiment 016's `1.025280423`.
3. Its 48-sequence paired KL gate passed, as shown above.
4. The exact packed quality benchmark reproduced the BF16 base, proved packed/reference parity with
   maximum absolute error `0.0`, and improved WikiText NLL from `7.222190376` to `7.171782786`.

The pinned downstream tasks were mixed: PIQA `+0.010`, ARC Easy `−0.005`, ARC Challenge `+0.005`,
HellaSwag `−0.010`, WinoGrande `+0.055`, and BoolQ `−0.210` (macro `−0.025833`). Corrected D2 is
therefore a demonstrated KL/NLL improvement at lower BPW, not an across-the-board task-quality win.

The complete human- and machine-readable records are
[`d2-corrected-v3-summary.md`](../../evidence/020/formal/d2-corrected-v3-summary.md) and
[`d2-corrected-v3-summary.json`](../../evidence/020/formal/d2-corrected-v3-summary.json).
