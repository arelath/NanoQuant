# Experiment 004: selective `v_proj` rank expansion

## Question

Can Experiment 003 improve reconstruction and matched quality by adding 30% more packed bits only to
`self_attn.v_proj`, without repeating the complete 4B compression run or changing any other quantized layer?

## Method

Experiment 004 is a derivative of the final globally tuned Experiment 003 artifact. It does not mutate or replace
Experiment 003.

- The source is the validated packed artifact from `outputs/003-gemma-3-4b-it/packed`.
- Every one of the 204 non-`v_proj` quantized layers is copied tensor-exactly.
- For each of the 34 `v_proj` layers, the existing factors, middle scales, outer scales, outliers, and bias are retained
  as an exact prefix/state.
- New binary factors approximate the residual between the original BF16 weight and the final Experiment 003
  reconstruction. Only their new middle-scale coefficients are fitted; a zero vector is always a feasible rollback.
- The stored BF16 correction must reduce the original diagonal weighted objective or the layer fails closed.
- Ranks remain multiples of 32 and are capped at the physical maximum rank of 1024.
- Each completed layer is content-addressed and journaled, so the derivative resumes without repeating completed
  expansions.

This is an additive isolation experiment, not a fixed-total-BPW reallocation. The requested 1.30x `v_proj` bits
became 1.3112x after alignment/capping and increased the complete packed decoder artifact by 0.8795%.

## Artifact invariants

The derivative audit passed:

- target layers expanded: 34;
- non-target layers verified exact: 204;
- target layers with exact original factor/scale prefixes: 34;
- source packed bytes: 402,802,128;
- derivative packed bytes: 406,344,600;
- GGUF bytes: 1,131,806,400;
- GGUF SHA-256: `a3c53fb7c6b6a5201545585f27bde2fbb92803e0baebff913f5794a7d5404c8a`;
- token embedding type: Q8_0.

Ranks increased from 608–800 (mean 714.35) to 800–1024 (mean 945.88). Seventeen layers reached the 1024 cap.

## Reconstruction result

Every `v_proj` layer improved its reconstruction objective.

| Metric | Experiment 003 state | Expanded state | Relative change |
| --- | ---: | ---: | ---: |
| Mean weighted normalized error | 0.282931 | 0.204735 | -27.64% |
| Mean raw normalized error | 0.324355 | 0.235036 | -27.54% |
| Sum of weighted error | 1,504.654130 | 1,079.931153 | -28.23% |

## Matched quality result

The candidate and Experiment 003 used identical WikiText tokens, task rows, tokenizer, sample counts, sequence
length, and dense replay backend. Their BF16 results were also identical.

| Benchmark | Experiment 003 | `v_proj` +30% | Delta |
| --- | ---: | ---: | ---: |
| WikiText-2 perplexity (lower is better) | 84.106649 | 84.369879 | +0.263229 (+0.31%) |
| PIQA `acc_norm` | 0.660 | 0.635 | -0.025 |
| ARC Easy `acc_norm` | 0.450 | 0.425 | -0.025 |
| ARC Challenge `acc_norm` | 0.300 | 0.295 | -0.005 |
| HellaSwag `acc_norm` | 0.480 | 0.430 | -0.050 |
| Winogrande `acc` | 0.535 | 0.590 | +0.055 |
| BoolQ `acc` | 0.645 | 0.640 | -0.005 |

Peak WDDM shared memory was 220,200,960 bytes, below the 805,306,368-byte fail-fast limit.

## Corrected interpretation

This derivative is not sufficient evidence to reject the allocation change. It modifies the final post-KD artifact:
the added factors never participate in the original per-layer tuning, post-block refit, or global distillation, and
the retained KD-tuned scales/outliers/norms were optimized for the lower-rank model. The quality comparison therefore
measures an untuned overlay on a stale downstream optimum, not the result of applying the allocation policy at the
start of the full pipeline.

The reconstruction result is a strong positive signal. It improves every target layer, lowers the aggregate weighted
error by 28.23%, and costs only 0.8795% more packed decoder storage. That is enough to carry the +30% allocation
forward as a promising full-run candidate.

The task deltas should also not be treated as a definitive rejection gate. The protocol contains only 200 examples
per task and performs several simultaneous comparisons. It is useful for detecting large regressions, but these small
mixed changes do not establish the quality effect of a fully retuned model. The retained WikiText result contains
only aggregate NLL, so a paired per-window uncertainty estimate cannot be reconstructed from this artifact.

## Decision

Do not reject `v_proj` expansion, and do not promote it as proven quality behavior yet. Run the allocation through
the complete factorization → layer/block tuning → post-block refit → global KD sequence, then compare the fully tuned
candidate against Experiment 003 on the same protocol. The full run must record the realized size increase separately
from the quality result because this additive experiment does not keep total BPW fixed.
