# Mixed Free/Coded V Basis

**Date:** 2026-07-30
**Status:** representative reconstruction and joint functional gates passed

## Question

[Sparse-Corrected Asymmetric Sign Codes](53-sparse-corrected-sign-code-screen.md)
showed that leaving U free and compressing only V is substantially better
than constraining both factors. Its best all-coded V arm used rank 1,472 and
beat the rank-970 free-word reconstruction by 0.56% at equal storage.

This follow-up asks whether every V component should pay the same price.
Instead of choosing between:

- fewer fully free components; or
- more uniformly codebooked components;

the mixed basis combines both. A prefix of V rows remains ordinary free
32-bit sign words. The remaining rows use a 10-bit codebook plus two
correction positions encoded in 9 bits.

## Why one-sided freedom helps

For a factorization `W ~= U V`, each V row is a basis pattern and each U
column supplies its coefficient signs across output rows.

When both U and V are codebooked:

- the available basis patterns are restricted; and
- the per-output mixtures of those patterns are also restricted.

Leaving U free removes the second restriction. Every output row can choose
an independent sign mixture over the structured V dictionary.

The mixed V basis removes part of the first restriction as well. Fully free
V rows act as expressive anchors for directions that the codebook represents
poorly, while cheaper coded rows retain enough breadth to exceed the
production rank.

Component permutation is an exact factorization symmetry. The free rows can
therefore be stored as a contiguous prefix by applying the same permutation
to U columns, V rows, and component scales. The representation stores only a
16-bit free-row count; it does not need a rank-sized bitmap.

## Implementation

`src/nanoquant/domain/sign_word_codebook.py` now supports:

- exact mixed-basis bit accounting;
- selection of the largest aligned free V prefix within a target budget;
- projection of free and corrected-code rows within one factor;
- exact export of the free prefix followed by decoded corrected rows;
- codebook utilization metrics that identify the free-row count.

The solver still jointly assigns each coded word over the 16 nearest
base-codeword candidates and its correction pair. Free rows use ordinary
independent signs throughout fitting.

`tools/probe_sign_word_codebook.py` accepts an explicit candidate rank and
free-row count. `tools/probe_corrected_codebook_splice.py` now evaluates one
or several blocks against one shared teacher cache, allowing a true joint
three-block splice rather than adding isolated KL estimates.

This remains analysis-only. No persisted artifact, GGUF, or runtime contract
has changed.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Matrices: blocks 0, 12, and 24 `mlp.down_proj`, each 1,152 x 6,912
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Baseline: free U and V words, rank 970, 800 outer iterations
- Coded V rows: 10-bit full codebook index plus a 9-bit unordered correction
  pair
- U: entirely free
- Candidate rank alignment: 32
- Scale fit: two alternating passes
- Primary arm: rank 1,344, first 256 V rows free, remaining 1,088 rows coded
- Seeds: 0, 1, and 2 on block 12
- Functional gate: individual block-12 splice and joint blocks 0/12/24
  splice on held-out WikiText-2 sequences of length 512

The primary reconstruction command is:

```powershell
.\.venv\Scripts\python.exe tools\probe_sign_word_codebook.py `
  --model <pinned-snapshot>\model.safetensors `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\sign-word-codebook-probe\block12-down-r970-800-mixed-right-flip2-k10-rank1344-free256.json `
  --block 12 --projection down --baseline-rank 970 `
  --candidate-rank 1344 --right-free-rows 256 `
  --index-widths 10 --outer-iterations 800 `
  --codebook-mode full-right-flip2 `
  --assignment-batch-words 8192
```

## Exact equal-bit accounting

| Item | Free-word baseline | Mixed candidate |
| --- | ---: | ---: |
| Rank | 970 | 1,344 |
| Rank multiple | 1.000x | 1.386x |
| Free U sign bits | included below | 1,548,288 |
| Free V sign bits | included below | 1,769,472 |
| Coded V payload bits | 0 | 4,465,152 |
| Total sign/payload bits | 7,822,080 | 7,782,912 |
| Scale bits | 144,544 | 150,528 |
| Codebook + free-count bits | 0 | 32,784 |
| Total bits | 7,966,624 | 7,966,224 |
| Actual BPW | 1.000502 | 1.000452 |
| Unused baseline budget | 0 | 400 |
| Factorized work / dense | 0.982x | 1.361x |

The mixed arm spends essentially the entire baseline budget. It buys 38.6%
more components, of which 19.0% are fully free V anchors.

## Allocation sweep

Every tested k=10 mixed allocation beats the baseline:

| Rank | Free V rows | Free share | Actual BPW | Weighted RMSE | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,216 | 480 | 39.5% | 0.994697 | 0.531040 | -0.42% |
| 1,280 | 352 | 27.5% | 0.991932 | 0.531149 | -0.40% |
| 1,312 | 288 | 22.0% | 0.990550 | 0.531504 | -0.34% |
| 1,344 | 256 | 19.0% | 1.000452 | 0.528163 | **-0.96%** |
| 1,376 | 192 | 14.0% | 0.999070 | 0.528481 | -0.90% |
| 1,408 | 128 | 9.1% | 0.997687 | 0.529053 | -0.80% |
| 1,472 | 0 | 0.0% | 0.994920 | 0.530318 | -0.56% |

The curve directly answers the motivating question. Near the optimum, 256
free basis vectors are more valuable than the 128 extra coded components
available in the all-coded rank-1,472 arm.

The result is not monotonic in rank alone because each point changes both
the number of components and their freedom. Rank 1,344 is the best measured
balance.

### Smaller-table counterfactual

A k=8 table permits more free anchors at the same rank, but loses too much
diversity in the coded portion:

| Arm | Rank | Free V rows | Weighted RMSE change |
| --- | ---: | ---: | ---: |
| k=8, two corrections | 1,344 | 384 | -0.07% |
| k=8, two corrections | 1,408 | 288 | -0.07% |
| k=10, two corrections | 1,344 | 256 | **-0.96%** |

Free anchors cannot compensate for an undersized dictionary. The coded and
free subsets both contribute material capacity.

## Stability

| Seed | Free-word RMSE | Mixed RMSE | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.533293 | 0.528163 | -0.962% |
| 1 | 0.533245 | 0.528143 | -0.957% |
| 2 | 0.533205 | 0.528138 | -0.950% |

All seeds agree within 0.012 percentage points. The mixed optimum is at least
as stable as the earlier all-coded candidate.

## Cross-block reconstruction

| Block | Free-word RMSE | Mixed RMSE | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.519717 | 0.514197 | **-1.06%** |
| 12 | 0.533293 | 0.528163 | **-0.96%** |
| 24 | 0.505094 | 0.503029 | **-0.41%** |

The mixed basis fixes the all-coded arm's block-24 regression and improves
all three representative depths. Block 0's unweighted RMSE rises by 0.11%
while its Fisher-weighted RMSE falls by 1.06%; the functional gate below
supports the weighted choice.

## Held-out splice KL

### Block 12 alone

On the standard 12-sequence splice:

| Arm | KL nats/token |
| --- | ---: |
| Free words | 0.037163 |
| Mixed basis | 0.034795 |

The mixed basis reduces KL by **6.37%**. Its paired 95% delta interval is
`[-0.004349, -0.000031]`, entirely below zero. This passes the individual
functional gate that the all-coded candidate missed.

### Joint blocks 0, 12, and 24

| Sequences | Free-word KL | Mixed KL | Relative change | Paired 95% delta interval |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 0.211320 | 0.187932 | **-11.07%** | [-0.053809, +0.003438] |
| 24 | 0.242574 | 0.221897 | **-8.52%** | [-0.037027, -0.006234] |

The 12-sequence point is large but its interval narrowly crosses zero.
Doubling the independently ordered sequence inventory retains an 8.52% gain
and moves the complete paired interval below zero. The gains therefore
combine constructively across depth rather than canceling.

## Interpretation

The experiment supports three conclusions:

1. **One-sided coding is structurally preferable.** Free U coefficients let
   every output row adapt independently over the V dictionary.
2. **A heterogeneous V basis is better than uniform coding.** A minority of
   fully free atoms repairs directions that are expensive for the dictionary,
   while coded atoms provide breadth.
3. **Breadth still matters.** Making too many atoms free reduces total rank
   and loses some of the gain; making the table too small also loses.

This resembles an over-complete dictionary with a high-capacity residual
subdictionary, not ordinary symmetric matrix factorization.

## Decision

Promote rank 1,344 with 256 free V rows, a k=10 codebook, and two corrections
per coded word to a broader `down_proj` screen. It passes:

- equal-bit weighted reconstruction;
- three-seed stability;
- representative early/middle/late block reconstruction;
- individual block-12 splice KL;
- 24-sequence joint three-block splice KL.

Do not yet change GGUF or the runtime. The remaining gates are:

1. screen all 26 `down_proj` matrices and retain the mixed form only where it
   beats the local free-word control;
2. evaluate the selected 26-block joint splice;
3. prototype decode throughput, because rank 1,344 raises factor work from
   0.982x to 1.361x dense and adds table/correction decode;
4. only then define a packed schema and complete-run experiment.

The current evidence is strong enough to justify those costs. It is not yet
evidence that applying the format to every tensor type or every block is
beneficial.
