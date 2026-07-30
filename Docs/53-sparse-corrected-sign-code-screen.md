# Sparse-Corrected Asymmetric Sign-Code Screen

**Date:** 2026-07-30
**Status:** reconstruction screen passed; functional promotion gate
directionally favorable but statistically inconclusive

**Follow-up:** [Mixed Free/Coded V Basis](54-mixed-free-coded-v-basis.md)
replaces the all-coded candidate with a better mixed basis. Keeping 256 V
components fully free improves every tested block and passes a 24-sequence
three-block splice-KL gate.

## Question

The pure fitted codebook and global progressive-fixing screens failed:

- arbitrary k=12 codebooks on both factors regressed weighted RMSE by 9.26%;
- globally fixing 20 bit positions regressed it by 62.68%.

Those results did not exhaust compact sign codes. This follow-up asks three
more specific questions:

1. Are balanced bit positions predictable from one another even though their
   marginal signs are random?
2. Should U and V pay the same compression price when V contains six times as
   many sign words?
3. Can a small number of explicit word corrections recover the signs that a
   compact codebook represents poorly?

## Variants

### Signed-relation forest

`src/nanoquant/domain/relational_sign_code.py` represents every 32-sign word
with `k` root signs and a compact forest of signed-copy relations. A relation
such as `bit 17 = -bit 4` preserves balanced signs, unlike fixing bit 17 to a
constant. Progressively merging the most correlated roots leaves exactly
`k` stored signs per word and needs only 384 metadata bits for both factors.

This variant found no useful pairwise structure. At k=12, the strongest
learned agreement was 50.64% in U and 50.37% in V. Its rank-2,560 equal-bit
fit regressed weighted RMSE by 58.47%.

### Asymmetric codebooks

For `down_proj`, U has 36 sign words per rank while V has 216. Compressing
both equally spends representational freedom on the smaller side even though
V supplies six sevenths of the word-storage opportunity.

The asymmetric arm therefore:

- keeps every U sign free at 32 bits per word;
- compresses only V with a full fitted codebook;
- spends the resulting savings on aligned rank.

This halves the pure k=12 codebook regression, from +9.26% to +5.00%.
Warm-starting the over-complete factors for 25% of the solve changes that to
+4.99%, showing that the remaining loss is code capacity rather than
initialization.

### Sparse-corrected V codebook

The successful representation keeps U free and encodes each V word as:

- a 10-bit index into a 1,024-entry table of 32-sign words;
- an unordered pair of distinct bit positions to flip.

There are `C(32, 2) = 496` correction pairs, so their exact fixed-width price
is 9 bits. A V word therefore costs 19 bits instead of 32. The decode table is
4 KiB. U remains in the existing free-word representation.

The joint projector evaluates the 16 nearest base-codeword candidates for
each word, then selects the codeword and correction positions together.
Codebook centroid updates invert the selected correction positions before
accumulation, so the table is fitted to the underlying codewords rather than
to their decoded corrections. This detail matters: nearest-codeword-first
assignment followed by two flips regressed by 1.64%; joint assignment improves
the same stored representation to a 0.56% win.

Correction pairs are stored in the analysis result as positions, but the bit
accounting charges their 9-bit combinatorial rank. A future packed format
would store that rank and recover the two positions during decode.

## Protocol

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Primary matrix: block 12 `mlp.down_proj`, shape 1,152 x 6,912
- Cross-block matrices: blocks 0 and 24 `mlp.down_proj`
- Importance: retained 256-sample corrected-CCE Fisher state, shrinkage 0.6
- Baseline: free-word production ADMM, rank 970, 800 outer iterations,
  two-pass scale fit
- Candidate: U free, V k=10 plus two correction positions, 800 outer
  iterations, two-pass scale fit
- Codebook updates: every 10 iterations through iteration 400
- Corrected assignment shortlist: 16 base codewords
- Rank alignment: 32
- Primary candidate rank: 1,472, selected at no more than the baseline's
  complete sign-and-scale bit budget
- Seeds: 0, 1, and 2 on block 12
- Functional gate: paired dense splice of block 12 `down_proj` on 12 and 24
  held-out WikiText-2 sequences of 512 tokens

The reconstruction command is:

```powershell
.\.venv\Scripts\python.exe tools\probe_sign_word_codebook.py `
  --model <pinned-snapshot>\model.safetensors `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\sign-word-codebook-probe\block12-down-r970-800-full-right-flip2-k10-joint16.json `
  --block 12 --projection down --baseline-rank 970 `
  --index-widths 10 --outer-iterations 800 `
  --codebook-mode full-right-flip2 `
  --assignment-batch-words 8192
```

`tools/probe_corrected_codebook_splice.py` reruns both fitted
reconstructions in one device lease and evaluates them against one shared
teacher-log-probability cache.

## Equal-bit arithmetic

| Item | Free words | Corrected code |
| --- | ---: | ---: |
| Rank | 970 | 1,472 |
| Rank multiple | 1.000x | 1.518x |
| U bits per word | 32 | 32 |
| V bits per word | 32 | 10 + 9 |
| Sign/index/correction bits | 7,822,080 | 7,736,832 |
| Scale bits | 144,544 | 152,576 |
| Table bits | 0 | 32,768 |
| Total bits | 7,966,624 | 7,922,176 |
| Actual BPW | 1.000502 | 0.994920 |
| Unused baseline budget | 0 | 44,448 |
| Factorized work / dense | 0.982x | 1.491x |

The representation does produce large per-V-word savings, but the primary
screen deliberately reinvests them in 51.8% more components. It therefore
tests quality at equal storage, not a direct 40.6% reduction of the final
matrix size.

The table and correction streams are fully exercised. V uses all 1,024 table
entries with 9.993 bits of index entropy. Correction-position entropy is
4.9999 of 5 bits and the most frequent position accounts for only 3.21% of
corrections. Variable-length entropy coding has no obvious additional gain.

## Reconstruction results

### Representation progression on block 12

| Arm | Rank | Weighted RMSE | Change |
| --- | ---: | ---: | ---: |
| Free words | 970 | 0.533293 | - |
| Signed-relation k=12 | 2,560 | 0.845134 | +58.47% |
| V-only codebook k=8 | 2,688 | 0.575748 | +7.96% |
| V-only codebook k=12 | 2,048 | 0.559957 | +5.00% |
| V-only codebook k=14 | 1,728 | 0.559727 | +4.96% |
| V k=12 + one flip, joint | 1,568 | 0.537212 | +0.73% |
| V k=12 + two flips, joint | 1,344 | 0.530961 | **-0.44%** |
| V k=8 + two flips, joint | 1,600 | 0.535796 | +0.47% |
| V k=10 + two flips, joint | 1,472 | 0.530318 | **-0.56%** |

The k=10/two-correction point brackets the useful exchange rate: k=8 loses
too much base-table capacity, while k=12 gives up more rank than its larger
table repays.

### Seed repeat

| Seed | Free-word RMSE | Candidate RMSE | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.533293 | 0.530318 | -0.558% |
| 1 | 0.533245 | 0.530216 | -0.568% |
| 2 | 0.533205 | 0.530304 | -0.544% |

All three seeds win, spanning only 0.024 percentage points. The reconstruction
result is stable rather than an initialization outlier.

### Cross-block check

| Block | Free-word RMSE | Candidate RMSE | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.519717 | 0.516346 | **-0.65%** |
| 12 | 0.533293 | 0.530318 | **-0.56%** |
| 24 | 0.505094 | 0.505727 | +0.13% |

The candidate is beneficial in two representative blocks and approximately
neutral in the third. It is not a universal per-matrix win, so a future plan
must select it by measured unit rather than applying it blindly.

## Size frontier

Holding the code fixed while retaining some of the storage savings gives a
steep quality frontier:

| Rank | Actual BPW | Weighted RMSE | Change |
| ---: | ---: | ---: | ---: |
| 1,344 | 0.910172 | 0.557140 | +4.47% |
| 1,408 | 0.952546 | 0.543366 | +1.89% |
| 1,440 | 0.973733 | 0.536954 | +0.69% |
| 1,472 | 0.994920 | 0.530318 | **-0.56%** |

At the original rank 970 the representation would cost about 0.666 BPW, but
the measured curve shows that spending most of the savings on additional
components is essential. This screen supports a better quality-per-bit
exchange, not a claim that one third of the matrix can be removed for free.

## Held-out splice KL

| Sequences | Free-word KL | Candidate KL | Relative change | Paired 95% delta interval |
| ---: | ---: | ---: | ---: | ---: |
| 12 | 0.037163 | 0.035514 | **-4.44%** | [-0.003714, +0.000732] |
| 24 | 0.042196 | 0.041078 | **-2.65%** | [-0.003857, +0.001946] |

Both held-out measurements agree with the reconstruction win in direction.
Candidate NLL is also lower in both measurements. Neither paired bootstrap
interval excludes zero, however; adding the second 12-sequence tranche
increased heterogeneity rather than producing artificial confidence.

The correct verdict is therefore not a functional failure, but an
inconclusive promotion gate. A single down-projection splice has low absolute
KL impact, and this sample does not establish that its favorable point
estimate will survive a multi-unit plan.

## Runtime implications

The decoder is plausible but not free:

1. load a 10-bit V index;
2. gather one 32-bit sign word from a 4 KiB table;
3. decode a 9-bit combinatorial pair;
4. XOR the two correction bits;
5. execute the existing sign-word reduction.

The equal-bit arm also increases factor work from 0.982x to 1.491x dense.
Even if table and correction decode are cheap, 51.8% more components can
raise inference time materially. Runtime benchmarking remains mandatory
before promotion.

## Decision

Retain U-free, V-k=10, two-correction codebooks as a real research candidate.
It is the first sign-word relaxation tested here that:

- beats free words at equal storage;
- repeats across seeds;
- generalizes to another block;
- improves held-out splice KL in point estimate.

Do not yet change the artifact schema, GGUF, or runtime, and do not launch a
complete 26-block compression. The paired functional interval still crosses
zero and block 24 is slightly worse.

The next justified experiment is a selected multi-block `down_proj` splice
using this code only where the equal-bit reconstruction screen wins, followed
by a decode-throughput prototype if that aggregate KL interval excludes
zero. Increasing the corrected-assignment shortlist beyond 16 is another
bounded solver refinement; variable-length coding of indices or correction
positions is not supported by the measured near-maximum entropies.
