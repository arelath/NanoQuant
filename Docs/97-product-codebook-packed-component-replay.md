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
