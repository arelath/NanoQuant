# Product-codebook packed component replay

## Gate

The mixed 1-BPW product-codebook policy is promotable only after its accepted
separable dense corrections are replayed through the compact factors and the
result is evaluated from a packed representation. Dense BF16 replacement
weights remain diagnostic evidence; they are not a compressed format.

This gate keeps the allocation and all discrete payloads unchanged. It folds
the pass-2 correction into already charged scale axes and outlier values:

- gate/up physical output-row multipliers map to factorization `scale_pre`
  because these tall projections were factorized transposed;
- down input-column and output-row multipliers map to `scale_pre` and
  `scale_post`, respectively;
- fixed outlier values receive the same physical row/column multipliers, while
  their corresponding physical `scale_pre` entries remain exactly zero.

The product-code tables, 16-bit assignments, free sign rows, rank, outlier
indices, and allocation are immutable during this replay.

## Compact layout v1

`product-codebook-free-k16-v1` stores each replacement in its factorization
orientation:

- the full left sign factor as llama.cpp-compatible packed I32 words;
- a packed I32 prefix of free right-factor sign rows;
- two unsigned 8-bit table selectors per coded 32-sign word, persisted as one
  tightly packed 16-bit index;
- two learned 256 by 16 sign tables, each persisted as 256 I16 words;
- three BF16 scale axes and the existing BF16 fixed-outlier payload;
- a 16-bit logical free-row count.

Tall transposed projections are decoded and transposed once during runtime
preparation into the canonical `llama.cpp-i32-lsb-v1` layer state. The compact
artifact is an overlay bound by SHA-256 to the Experiment 056 packed base; all
unreplaced attention layers fall back to that base.

Logical BPW charges the exact allocation widths, including sign-word padding,
16-bit coded records, both half tables, BF16 scales, fixed outlier values and
indices, and the free-row count. Safetensors and descriptor overhead are
reported separately and do not alter logical BPW.

## Required evidence

The materializer must retain exact factors before dense reconstruction, prove
that binary search did not modify coded rows, infer the separable correction
from the accepted base/pass-2 dense pair, and report component-replay residuals
against the pass-2 tensors. The packed evaluator must use the same retained
WikiText token windows and teacher protocol as the dense comparison.

This overlay does not alter resident quantization or its numerical path, so it
does not change `RESIDENT_ALGORITHM_VERSION`.

## Experiment result

The v3 replay completed all 64 regeneration jobs and retained 78 MLP
replacements. The compact MLP payload is exactly 618,889,878 logical bits; with
the unchanged Experiment 056 attention payload, the full allocation remains
697,753,234 bits, or 0.999987735 BPW. The persisted component tensor file is
77,478,736 bytes with SHA-256
`c6bfa97bebf73b54ee01c0b9f8270b2d778c0c76f82b2a7c09f0f7bdf71c6831`.

Against the accepted dense pass-2 tensors, the largest per-matrix packed replay
RMSE is 0.000441437 and the largest absolute BF16 difference is 0.03125. On the
two untouched 48-sequence WikiText test windows (offsets 192 and 240), the
combined results are:

| Representation | KL nats/token | NLL |
|---|---:|---:|
| Experiment 056 packed base | 2.135262 | 4.770271 |
| Dense product-codebook pass 2 | 3.548307 | 6.302698 |
| Packed product-codebook replay | 3.535626 | 6.290203 |

Packed replay improves slightly over the dense pass-2 diagnostic: KL delta
-0.012681 (-0.357%, paired 95% CI [-0.015935, -0.009457]) and NLL delta
-0.012495 (-0.198%, paired 95% CI [-0.015854, -0.009207]). This passes the
unchanged-bit representation gate and promotes the compact layout for continued
candidate work. It does not promote the candidate as an Experiment 056 model
replacement: versus the packed base, KL is 65.583% worse and NLL is 31.863%
worse, with both paired intervals wholly above zero.

## Candidate-specific distillation result

The packed candidate subsequently completed eight epochs and 2,048 steps of
top-64 teacher distillation over only the already-budgeted foldable MLP scale
axes. Held-out one-standard-error selection chose epoch 6 (step 1,536). On its
eight-sequence validation selection split, NLL improved from 6.177293 to
5.457286 and top-k KL from 2.252597 to 1.387740. The folded representation was
exactly equal to the selected in-memory checkpoint.

An independent payload audit found no changes to any product-code table,
16-bit assignment, free sign row, rank, orientation, or outlier index. Exactly
182 scale/outlier-value tensors changed. Both overlays occupy 77,478,736 tensor
bytes and charge the same 697,753,234 logical bits (0.999987735 BPW). The
distilled tensor SHA-256 is
`8c150a5dbdaf46488bd4042ee9b33fe3caf192351e40bd17af359d21ac23860a`.

The exact two-window WikiText test protocol gives:

| Representation | KL nats/token | NLL |
|---|---:|---:|
| Experiment 056 packed base | 2.135262 | 4.770271 |
| Packed product codebook before candidate KD | 3.535626 | 6.290203 |
| Packed product codebook after candidate KD | 2.826872 | 5.441854 |

Candidate KD improves KL by 0.708754 (20.046%, paired 95% bootstrap CI
[0.669505, 0.749690]) and NLL by 0.848349 (13.487%, paired 95% bootstrap CI
[0.808937, 0.888243]). This is a material recovery, but the candidate remains
worse than the Experiment 056 packed base by 32.390% KL and 14.079% NLL, with
both paired intervals wholly above zero. Therefore the candidate-specific KD
gate completes successfully, while model-replacement promotion still fails.

## Full retained quality benchmark

The distilled packed overlay was then evaluated on the Experiment 056 retained
quality protocol: 64 WikiText-2 sequences of length 128 and 1,000 examples from
each of PIQA, ARC-Easy, ARC-Challenge, HellaSwag, Winogrande, and BoolQ. The
protocol-matched BF16 result was reused from the terminal Experiment 056
receipt; the candidate inputs were freshly prepared and their token hash is
identical. The quality receipt records product-overlay descriptor SHA-256
`f9880bfdd159a6ae0aaa7b4640e6409bd94f1b60dac0d0fd3e6dd21d6cdd9eab`.

| Representation | WikiText PPL | PIQA | ARC-E | ARC-C | HellaSwag | Winogrande | BoolQ | Task mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 96.460 | 0.730 | 0.638 | 0.376 | 0.528 | 0.594 | 0.768 | 0.606 |
| Experiment 056 packed base | 253.146 | 0.613 | 0.366 | 0.232 | 0.378 | 0.525 | 0.628 | 0.457 |
| Distilled product-codebook overlay | 416.349 | 0.556 | 0.335 | 0.212 | 0.367 | 0.523 | 0.440 | 0.406 |

Relative to the Experiment 056 packed base, the distilled product overlay is
64.470% worse in WikiText perplexity and 11.269% worse in unweighted task mean;
it is lower on all six individual tasks. The full retained benchmark therefore
confirms the earlier KL/NLL non-promotion decision. The candidate-specific KD
improves the held-out diagnostic windows but does not generalize sufficiently
to the retained quality distribution.
