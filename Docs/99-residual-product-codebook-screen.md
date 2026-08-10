# Residual product-codebook screen

## Question

Test whether replacing the retained concatenated half-word code

\[
c_{ij}=\operatorname{concat}(A_i,B_j),\quad A_i,B_j\in\{-1,+1\}^{16}
\]

with a multiplicative full-word residual product

\[
c_{ij}=A_i\odot B_j,\quad A_i,B_j\in\{-1,+1\}^{32}
\]

improves the Experiment 057 down-projection candidate. Both representations
use the same fixed 16-bit selector. The residual product has two 256-by-32
sign tables, so it adds exactly 8,192 table bits per coded layer relative to
the half-word representation.

## Implementation and protocol

This is an analysis-only codebook mode. It does not change the resident
algorithm, packed schema, or runtime. Decoding selects two full 32-sign words
and multiplies them elementwise. Fitting uses alternating exact 256-way
coordinate assignments. Multiplying both learned tables by the same sign word
fixes the multiplicative gauge and preserves every represented codeword.

The real-Gemma screen changed only the codebook family relative to the matched
half-word controls:

- projection: `down_proj` only;
- blocks: 0, 12, and 24;
- rank: 1,152 candidate versus rank-970 free-word control;
- free rows: 672 for blocks 0 and 12, 704 for block 24;
- the exact seven Experiment 056 outlier columns for each layer;
- 1,200 ADMM iterations with a 100-iteration unconstrained warmup;
- one alternating residual-product assignment sweep;
- the retained 8+8 control-then-tabu binary search;
- fixed-width 16-bit selectors and exact table-bit accounting.

The block-0 search-depth check repeated the full protocol with two alternating
assignment sweeps.

## Results

| block | free rows | free NRMSE | half-word NRMSE | residual-product NRMSE | residual vs half-word | used pairs | selector entropy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 672 | 0.514894 | 0.513221 | 0.549434 | +7.06% | 42,472 | 14.36 bits |
| 12 | 672 | 0.529194 | 0.528297 | 0.566033 | +7.14% | 42,535 | 14.43 bits |
| 24 | 704 | 0.500256 | 0.497147 | 0.533725 | +7.36% | 39,837 | 14.16 bits |

The extra assignment sweep on block 0 changed NRMSE only from 0.549434 to
0.549140. It remained 6.65% worse than the matched free-word control. The
selector used 42,494 pairs at 14.36 bits of entropy, so the failure is not a
collapsed or unused 16-bit selector.

Each residual candidate is exactly 8,192 bits larger than its half-word
control. That is about 0.00103 BPW for one down-projection matrix and roughly
0.0002--0.0003 BPW model-wide if applied to all 26 down projections. The rate
increase is negligible, but the measured reconstruction regression is not.

Evidence:

- `evidence/m4/residual-product-codebook-screen/block-00.json`
- `evidence/m4/residual-product-codebook-screen/block-00-sweeps2.json`
- `evidence/m4/residual-product-codebook-screen/block-12.json`
- `evidence/m4/residual-product-codebook-screen/block-24.json`
- matched half-word controls under
  `evidence/m4/product-codebook-warmup-prefix-block0-24` and
  `evidence/m4/product-codebook-warmup-prefix-block12`.

## Decision

Do not promote the residual-product representation and do not spend a
held-out KL benchmark on it. It fails the reconstruction gate consistently at
early, middle, and late down projections, while the half-word representation
slightly beats the corresponding free controls.

The result does not prove that every optimizer for a multiplicative product
must fail. It does show that the tractable alternating assignment proposed for
this rate point does not recover the apparent representational capacity, and
doubling its search depth has negligible effect. Continue with the retained
half-word product representation. Before considering a raw-word escape stream,
add a separate diagnostic that records the concentration of per-word weighted
codebook distortion; escapes are justified only if a small word fraction owns
most of that loss.
