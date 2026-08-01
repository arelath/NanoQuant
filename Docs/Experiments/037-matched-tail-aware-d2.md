# Experiment 037: Matched Tail-Aware D2

## Status

Complete; the fixed candidate is rejected. Experiment 036 remains paused and
was not reused. The common-state worker used block-bounded process slices so
Windows released retired rolling activation file handles between resumes.
Every prefix was validated, and the final one-block slice forced a fresh
`--require-complete` audit before export or quality evaluation.

## Question

Does the calibrated top-k-plus-tail policy reproduce its retained Experiment
035 improvement on a fresh Gemma D2 factorization when compared with
conditional KD from the exact same pre-KD state?

## Outcome

No. Tail-aware KD reproduced the language-model and full-KL improvement, and
the 1.06 fold restored the broad selected-mass floor, but the calibrated arm
regressed the six-task mean relative to the matched conditional arm. On C4 it
also improved KL while making NLL significantly worse than conditional KD.

All arms share resident identity
`(32d5b5d0..., 08d2e590..., 2eaebd95...)` and effective BPW
`1.0244947118`. The conditional and tail-aware optimizers each completed 256
steps through four independently loaded two-epoch checkpoints. Every
checkpoint contained 3,385 finite parameter/optimizer tensors.

### Broad WikiText-validation gate

The serialized calibrated result was reloaded normally before its final gate.

| Arm | NLL | Full KL | Top-k-plus-tail KL | Teacher-top-64 mass |
| --- | ---: | ---: | ---: | ---: |
| Pre-KD | 4.76183 | 1.53873 | 1.45476 | 0.82129 |
| Conditional top-k | 4.47838 | 1.66403 | 1.59103 | 0.47469 |
| Tail-aware 0.5 | **4.34053** | 1.34517 | 1.27625 | 0.70831 |
| Tail-aware 0.5 + 1.06 | 4.38038 | **1.32528** | **1.25248** | **0.75475** |

The raw tail-aware endpoint dominates conditional KD on NLL and both KL
metrics but misses the 0.75 mass floor. The identity-bound 1.06 fold passes
that floor and remains better than conditional KD on every broad distribution
metric.

### Complete retained quality benchmark

All rows use the same WikiText 64x128 token hash and the same first 200 rows of
PIQA, ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, and BoolQ.

| Arm | WikiText PPL | Six-task mean |
| --- | ---: | ---: |
| BF16 | 96.45961 | 0.62333 |
| Pre-KD | 279.20834 | 0.44000 |
| Conditional top-k | 187.52287 | **0.47417** |
| Tail-aware 0.5 | **174.08334** | 0.46250 |
| Tail-aware 0.5 + 1.06 | 186.91545 | 0.45667 |

Calibration preserves a small PPL advantage over conditional KD, but the
task mean is lower by `0.01750`. This fails the predeclared no-regression gate.

#### Larger task-inventory diagnostic

The same three active serialized arms were subsequently evaluated on the first
1,000 examples per task. This was a diagnosis after the predeclared decision,
not a replacement acceptance gate.

| Arm | WikiText PPL | Six-task mean |
| --- | ---: | ---: |
| Conditional top-k | 187.52287 | **0.45317** |
| Tail-aware 0.5 | **174.08334** | 0.44650 |
| Tail-aware 0.5 + 1.06 | 186.91545 | 0.44633 |

The conditional advantage shrinks from `0.01750` to `0.00683`, but it does not
reverse. The fold itself changes the larger task mean by only `-0.00017`.
Therefore the task deficit comes from the tail-aware training tradeoff, while
the post-hoc fold is specifically responsible for giving back WikiText and C4
NLL to meet the selected-mass floor.

### Pinned C4 gate

The four-arm C4 comparison used the pinned 48x512 slice with token hash
`sha256:e34b788d48b021857df1130779e98d3936d4275bb17553e079a027e496bc2bef`.

| Arm | C4 NLL | C4 PPL | BF16-teacher KL |
| --- | ---: | ---: | ---: |
| Pre-KD | 5.27119 | 194.65 | 1.22395 |
| Conditional top-k | 4.95102 | 141.32 | 1.28124 |
| Tail-aware 0.5 | **4.91396** | **136.18** | **1.05546** |
| Tail-aware 0.5 + 1.06 | 4.97351 | 144.53 | 1.06334 |

Uncalibrated tail-aware versus conditional passes both paired gates: NLL
`-0.03706`, 95% interval `[-0.05299, -0.02158]`, and KL `-0.22577`, interval
`[-0.24161, -0.21114]`. The calibrated arm keeps the KL improvement but has
NLL `+0.02249`, interval `[+0.00300, +0.04188]`, so the matched C4 gate fails.

The combined result is stronger than a single noisy task row: tail-aware
training improves NLL/KL but retains less task quality, while the post-hoc
scalar needed to meet the mass floor gives back useful NLL. A further scalar
sweep on the same validation slice would not resolve both structural
tradeoffs and would turn the gate into a tuning set.

## Common state

The numbered launcher
`experiments/037-matched-tail-aware-d2-base-gemma-3-1b-it.py` creates one fresh
resident state with global distillation disabled. It uses the immutable
Experiment 035 exact-unit KL profile solely to keep allocation fixed and avoid
building another multi-gigabyte uniform control. It does not import ranks,
factors, tuning values, or post-KD weights.

The common-state run must complete all 26 blocks and pass
`tools/validate_resident_run.py --require-complete` before branching. Both arms
are zero-copy hard-link forks of that validated directory.

## Matched arms

Both arms use 256 calibration samples, eight epochs, at most 32 batches per
epoch, batch size 1, learning rate 1e-5, top 64, at most 512 selected tokens,
8,192-token vocabulary chunks, and 128-token chunks. They differ only in the
loss contract:

| Arm | Loss | Tail mass weight | Final RMSNorm calibration |
| --- | --- | ---: | ---: |
| Control | conditional top-k | n/a | none |
| Candidate | top-k plus aggregated tail | 0.5 | 1.06, after checkpoint selection |

The candidate's 1.06 fold is model-specific and identity-bound. It is applied
only after the uncalibrated tail-aware checkpoint passes NLL/full-KL direction;
the final broad mass gate must be checked on the serialized calibrated model.

## Execution sequence

After the common state completes:

1. validate all resident artifacts and the 26-block prefix;
2. create `conditional` and `tail-aware` hard-link forks with
   `tools/fork_resident_run.py`;
3. run `tools/run_gemma_global_distillation.py` on the control with
   `--objective top_k --maximum-batches-per-epoch 32`;
4. run it on the candidate with
   `--objective top_k_tail --maximum-batches-per-epoch 32
   --tail-mass-weight 0.5`;
5. screen durable checkpoints on the predeclared WikiText-validation 48x512
   mass/NLL/full-KL/tail-KL gate;
6. fold scale 1.06 into the selected candidate using
   `tools/fold_global_tuning_final_norm_scale.py` and validate ordinary reload;
7. run the complete 64x128 WikiText and six-task/200-example benchmark on the
   common state, conditional control, uncalibrated candidate, and calibrated
   candidate;
8. run the pinned C4 48x512 paired gate;
9. complete logical/packed reload, GGUF export, byte/BPW accounting, and the
   numbered export summary for the accepted arm.

## Acceptance

The candidate must retain the common state's effective BPW and represented
payload bytes, keep teacher-top-64 mass at least 0.75 on the broad gate, and
improve NLL, full KL, and tail KL over conditional KD. It must improve
WikiText perplexity over the common pre-KD state without regressing the
six-task mean versus conditional KD, pass C4 with paired confidence, and
complete all artifact/export contracts. Training loss, a short monitor, or a
retained-state result alone cannot accept Experiment 037.

The fixed candidate fails the six-task and matched C4-NLL requirements and is
therefore not accepted or published as the Experiment 037 deployment arm.
The common state did complete the full export contract: packed quantized-layer
payload `89,480,664` bytes and GGUF `417,340,544` bytes with SHA-256
`79830c8d259f289cb2a175a5da89db4794522db6945a1213a6e0dc5d5da25ddd`.
Its validated, regenerable logical intermediate was removed after export to
recover disk space; the resident state, packed artifact, llama.cpp checkpoint,
GGUF, receipts, summary, and quality report remain.

## Retained evidence

- `evidence/037/experiment037-complete-validation.json`
- `Results/037/037-matched-tail-aware-d2-base-gemma-3-1b-it-quality.json`
- `Results/037/gemma-3-1b-it-nanoquant.export-summary.json`
- `evidence/037/experiment037-conditional-epoch8-validation104-48x512-kl.json`
- `evidence/037/experiment037-tail-aware-epoch8-validation104-48x512-kl.json`
- `evidence/037/experiment037-tail-aware-folded1p06-serialized-validation104-48x512-kl.json`
- `evidence/037/experiment037-conditional-standard-quality.json`
- `evidence/037/experiment037-tail-aware-uncalibrated-standard-quality.json`
- `evidence/037/experiment037-tail-aware-folded1p06-standard-quality.json`
- `evidence/037/experiment037-matched-c4-validation104-48x512.json`
- `evidence/037/experiment037-conditional-tasklimit1000-quality.json`
- `evidence/037/experiment037-tail-aware-uncalibrated-tasklimit1000-quality.json`
- `evidence/037/experiment037-tail-aware-folded1p06-tasklimit1000-quality.json`

## Next direction

Do not spend another fresh factorization on a different fixed scalar. First,
use the existing matched state to test a KD constraint or adaptive dual term
that applies mass pressure only when the trained endpoint violates the broad
floor. Couple it to an explicit trust-region or conditional-shape preservation
term so the optimizer does not spend the task-friendly behavior measured in
the conditional control. This targets both diagnosed failures: the raw
tail-aware endpoint misses the mass floor, and fixed mass pressure loses a
small but persistent task advantage. Candidate selection must use a separate
fit/validation split and leave the existing broad and C4 slices untouched.
