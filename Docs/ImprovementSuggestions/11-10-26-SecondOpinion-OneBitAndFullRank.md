# Second Opinion: 1-BPW Quality, and What To Do When Every Rank Is Full

**Date:** 2026-11-10
**Status:** Recommendation. Section 2 reports new CPU measurements made for this document; they are
weight-space screens on one retained fixture, not functional gates. Everything in Sections 3 and 4
is untested unless it cites a measurement.
**Builds on:** [9-3-26-LargeModelsAndMultiBitIdeas.md](9-3-26-LargeModelsAndMultiBitIdeas.md),
[8-12-26-Theory-Test-Ledger.md](8-12-26-Theory-Test-Ledger.md),
[FormatCapacity.md](FormatCapacity.md), and the September experiments
([Doc 100](../100-adaptive-free-row-functional-tuning-gate.md) through
[Doc 107](../107-downloaded-gemma-iq2-quality-profile.md)).

## Recommendation in brief

1. **Above the rank cap, stack a second binary factor on the residual.** Measured on the retained
   block-12 `down_proj` fixture, a second factor stage removes 74% of the remaining weighted error
   for 1.17 BPW, and 46% for 0.58 BPW. The INT8 outlier columns that Experiment 060 uses remove 14%
   for 0.70 BPW. Near 2.0 BPW that is roughly 0.09 weighted error against 0.19 to 0.20, with no
   new kernel primitive: it is the existing factor unit, evaluated twice and summed.
2. **Stop looking for 1-BPW quality in the coded bits.** Below the cap, each extra bit of rank
   already buys 1.8 to 1.9 bits of error reduction against a theoretical 2.0, and three hour-long
   searches found nothing further. What remains at 1 BPW is training signal, zero-bit capacity,
   and where the allocator is allowed to spend.
3. **Three 1-BPW changes I would make**, in order: let the allocator buy above the cap with
   residual stages inside the same 1.0 budget; replace the short logit-only distillation with a
   full-distribution, hidden-state-anchored one; add token-adaptive scale experts.
4. **Reset the target.** At 1 BPW the 1B model keeps 37% of the source model's above-chance task
   accuracy. The best published 1-bit result at 7B, with full training, keeps about 63%. If the
   required quality is near the published IQ2_M file (88%), that is a 2 to 4 BPW target at this
   model size, and the residual-stage ladder is the way to reach it.

I also have to withdraw one claim from my September document. I wrote that the factor format is the
wrong tool for near-square attention matrices at 2 BPW. Measured, it is not (Section 2.3).

---

## 1. What the September evidence says

### 1.1 The fitter is finished

Experiments 066 to 068 spent about 2.6 GPU-hours on one matrix. The first hour lowered weighted
error energy by 4.6% using 51,200 ADMM iterations in place of 800. The second hour added 0.03%.
All 172 perturb-and-refit restarts lost to the incumbent. The final state has zero improving
single-bit flips.

The fit at the cap is also close to an information bound. At rank 1,152 the block-12 `down_proj`
costs 1.167 BPW and reaches weighted NMSE 0.2275. A white Gaussian source at that rate cannot be
coded below 0.198. The format is about 0.6 dB from that reference. The weights are not white, so
their true bound is somewhat lower, but the practical reading holds: no encoder change at fixed
bits will move 1 BPW much.

### 1.2 The remaining error is white

[Doc 106](../106-row-column-exact-encoding-coverage.md) shows half the error needs 42% of the
columns, and [Doc 105](../105-row-column-bias-diagnostic.md) shows row and column offsets recover
0.1%. My own check on the same residual gives kurtosis 3.08 after row and column scaling, which is
Gaussian. White Gaussian residual is the worst case for sidecars that store selected entries
exactly, and the best case for dense low-rate codes.

### 1.3 Local functional proxies still misrank

[Doc 100](../100-adaptive-free-row-functional-tuning-gate.md) improved block-output error on both
fit and validation sets and still regressed logit KL with a confidence interval that excludes zero.
Only installed-candidate KL can decide a discrete choice.

### 1.4 The IQ2_M file is not a 2-bit baseline

[Doc 107](../107-downloaded-gemma-iq2-quality-profile.md) reports whole-file rates. Splitting the
tensor inventory in `Results/069/gguf-inventory.json` by decoder and embedding gives a different
picture:

| File | Decoder bytes | Decoder BPW | Six-task mean | WikiText-2 ppl |
|---|---:|---:|---:|---:|
| NanoQuant 048 | about 90 MB | 1.03 | 46.1% | 153.7 |
| NanoQuant 060 | 171.9 MB | 1.97 | 50.2% | 115.7 |
| Published IQ2_M | 342.4 MB | 3.92 | 57.7% | 111.6 |
| BF16 | — | 16 | 60.6% | 96.5 |

The 048 and IQ2_M rows are llama.cpp scores; the 060 row is its PyTorch reference, as in Doc 107.
The 048 decoder size is the file minus the embedding and about 6.6 MB of metadata.

Two thirds of the IQ2_M decoder is stored as IQ4_NL at 4.5 BPW. The cause is in
`llama.cpp/src/llama-quant.cpp`: the IQ1, IQ2 and IQ3 types need a column count divisible by 256,
and Gemma 3 1B's hidden size is 1,152, so every tensor with 1,152 inputs falls back to IQ4_NL.
Three consequences follow.

- Experiment 069 compared a 1.97-BPW decoder with a 3.92-BPW decoder. It does not show the factor
  format losing at 2 bits.
- A true 2-bit external baseline still does not exist for this model. Stock `llama-quantize`
  cannot produce one for Gemma 3 270M or 1B, because most of the decoder has 1,152 inputs. It can
  for Gemma 3 4B (2,560), Llama 3.2 1B (2,048) and larger models.
- The hybrid export I proposed in September (standard 2-bit tensors for attention matrices) is not
  available on the 1B testbed, and Section 2.3 shows it is not needed.

### 1.5 Perplexity is flattering the 1B results

The calibration and distillation mixture is half WikiText-2, so WikiText-2 test perplexity is close
to in-domain. NanoQuant 060 nearly matches IQ2_M on perplexity (115.7 against 111.6) and trails it
by 7.5 points on tasks. Subtracting chance from each task (50, 25, 25, 25, 50, 50) gives a clearer
scale:

| Model | Decoder BPW | Above-chance points | Share of BF16 |
|---|---:|---:|---:|
| BF16 Gemma 3 1B | 16 | 23.1 | 100% |
| Published IQ2_M | 3.92 | 20.2 | 88% |
| NanoQuant 060 | 1.97 | 12.7 | 55% |
| NanoQuant 048 | 1.03 | 8.6 | 37% |
| LLaMA-2-7B, BinaryMoS, 1 bit, full QAT | about 1 | 16.5 of 26.2 | 63% |
| LLaMA-2-7B, OneBit, 1 bit, full QAT | about 1 | 14.9 of 26.2 | 57% |

At 1 BPW the 1B model is at chance on Winogrande (50.9%) and ARC-Challenge (22.3%), and BoolQ
(62.1%) sits near the roughly 62% majority-class rate of its validation set. The two LLaMA rows are from the BinaryMoS paper's Table 3
([arXiv 2406.12311](https://arxiv.org/abs/2406.12311)); its six-task suite is the same family but I
have not verified the task list matches exactly. They are the fairest available ceiling: seven
times the parameters, every weight trained for three epochs, and still under two thirds retention.

**Implication.** This project's 1-BPW result is not far behind the state of the art once model size
and training budget are accounted for. No published method reaches near-source task quality at
1 BPW on a 1B to 7B model. I would report above-chance retention as the headline metric and add an
out-of-domain perplexity, so distillation gains on WikiText are not mistaken for capability.

---

## 2. New measurements

All runs are CPU-only, on the hash-verified Experiment 066 fixture: Gemma 3 1B block-12
`mlp.down_proj`, 1,152 by 6,912, the retained Experiment 041 diagonal-Fisher objective, seven
protected columns. Fits use production `factorize_admm_with_parameters` at 800 iterations plus the
two-pass scale fit. The metric is weighted NMSE (error energy over target energy). "Efficiency" is
bits of error-energy reduction per bit spent; an ideal Gaussian code gives 2.0.

Scripts and raw JSON are in `outputs/second-opinion-2026-11-10/` (`part1.py` to `part4.py`).
They are scratch analysis, not retained experiments. Limits: one matrix, weight space only, greedy
fits, no functional gate.

### 2.1 Below the cap, rank is the right purchase

| Representation | BPW | Weighted NMSE |
|---|---:|---:|
| Factor, rank 832 | 0.843 | 0.3605 |
| Factor, rank 992 | 1.005 | 0.2942 |
| Factor, rank 1,152 (cap), 800 steps | 1.167 | 0.2386 |
| Factor, rank 1,152, Experiment 067 best | 1.167 | 0.2275 |
| Dense sign matrix with row and column scales | 1.000 | 0.3860 |
| Rank 992 plus sign-coded residual on 16.2% of columns | 1.167 | 0.2514 |

Rank buys 1.81 bits per bit from 832 to 992 and 1.87 from 992 to 1,152. Trading rank for anything
else below the cap loses (last row against the third). This agrees with every earlier equal-bit
falsification, and it means the 1-BPW coded representation has no slack to recover.

### 2.2 Above the cap: five ways to spend the next bit

Each row starts from the Experiment 067 best fit (1.167 BPW, 0.2275) and adds one thing.

| Addition | Total BPW | Weighted NMSE | Efficiency |
|---|---:|---:|---:|
| Exact columns, 8.8% chosen by residual energy, INT8 cost (060's method, idealised) | 1.87 | 0.197 | 0.30 |
| Left factor made 4-bit, right factor re-optimised | 1.667 | 0.185 | 0.59 |
| Left factor made real-valued (INT8 cost) | 2.33 | 0.182 | 0.28 |
| Sign-coded residual on 50% of columns | 1.667 | 0.144 | 1.33 |
| Sign-coded residual on 83% of columns | 2.00 | 0.100 | 1.42 |
| Dense sign plane on the residual | 2.167 | 0.083 | 1.45 |
| **Second factor stage on the residual, rank 576** | **1.75** | **0.122** | **1.54** |
| **Second factor stage on the residual, rank 1,152** | **2.333** | **0.0595** | **1.66** |
| Dense sign plane, then a second sign plane | 3.167 | 0.030 | 1.46 |
| Dense 2-bit scalar residual | 3.167 | 0.027 | 1.54 |

References on the raw target, with no factor: 2-bit scalar quantiser 0.139 at 2.0 BPW, two stacked
sign planes 0.154 at 2.0, 3-bit scalar 0.051 at 3.0.

What I take from this:

- **A second factor stage is the best option measured.** It keeps almost the efficiency of the
  first stage (stage one: 1.77; stage two: 1.66). Interpolating to rank 832 gives about 0.089 at
  2.0 BPW, against about 0.197 at 1.87 BPW for the current INT8-column recipe and 0.139 for a
  plain 2-bit quantiser.
- **Earlier negatives on multi-stage fitting do not apply here.** They were measured below the cap,
  where the alternative was more rank. Section 2.1 confirms rank wins there. At the cap that
  alternative is gone.
- **Whole exact columns are the least efficient option by a factor of five**, because the residual
  is white. Keep the handful of true outlier channels that Experiment 058 showed matter
  functionally (13 columns for 0.013 BPW) and stop there.
- **The bottleneck is the right factor's row space, not the left factor's precision.** Making the
  small factor real-valued recovers only 20% of the error.
- **Do not alternate the factor and a sign plane.** Refitting the factor to the target minus the
  plane made things worse (0.113 to 0.118 against 0.087 greedy).

### 2.3 Near-square attention matrices: a correction

Unweighted, block 12, production ADMM:

| Representation | BPW | `o_proj` NMSE | `q_proj` NMSE |
|---|---:|---:|---:|
| Factor at half rank (512) | 0.944 | 0.229 | 0.182 |
| Dense sign matrix | 1.000 | 0.369 | 0.364 |
| Factor at full rank (1,024) | 1.889 | 0.084 | 0.067 |
| Half-rank factor plus sign plane | 1.944 | 0.083 | 0.066 |
| 2-bit scalar quantiser | 2.000 | 0.122 | 0.119 |
| Full-rank factor plus sign plane | 2.889 | 0.031 | 0.024 |
| 3-bit scalar quantiser | 3.000 | 0.036 | 0.036 |

The full-rank factor beats a 2-bit scalar code by 30 to 45% on both matrices. These matrices are
strongly low-rank (the top half of the spectrum holds 95 to 97% of the energy), which the factor
exploits and a scalar code cannot. My September statement that square matrices want a standard
2-bit tensor above 1.5 BPW was an extrapolation and is wrong for this model. These two rows are
unweighted, so the Fisher-weighted numbers will differ in level.

---

## 3. Ideas for 1 BPW

Section 2.1 rules out the coded representation. These are ordered by expected value for cost.

### 3.1 Let the allocator buy above the cap (new, uses Section 2.2)

**Observation.** In Experiment 054, 20 of 26 `down_proj` units sat at the rank ceiling: the
KL-calibrated allocator wanted more there and could not buy it. KL is front-loaded (blocks 0 to 10
carry 65%, blocks 18 to 24 carry 8%).

**Mechanism.** Give every capped unit a second response curve: a residual stage at rank 288, 576
or 864. The allocator stays at exactly 1.0 BPW and pays for early-block residual stages with rank
from late blocks.

**Why Experiment 055 does not close this.** 055 raised the ADMM rank past `min(m, n)`, where the
continuous relaxation has no extra directions to offer, used 100-iteration response probes, and
enabled broad tabu search in the same run. A residual stage is a different construction with a
measured efficiency of 1.5 to 1.7.

**Screen.** Fit rank-576 residual stages on the MLP units of blocks 0, 4 and 8, funded by rank
cuts in blocks 18 to 24 chosen from the retained KL profile. Install and measure paired splice KL
on a fresh 48 by 512 slice against the 054 plan at equal bits.

**Expected value.** Unknown sign, real size. It is the only 1-BPW idea here that changes where
bits go, and it reuses every existing component.

### 3.2 Anchored, full-distribution distillation

The 048 recipe distils for 256 steps of at most 512 tokens (about 131k token targets) on top-64
logits plus a tail bucket, training scales and norms only. BinaryMoS and OneBit train every weight
for three epochs at sequence length 2,048 with a cross-entropy term plus a layer-to-layer hidden
state term weighted 10. The gap is two to three orders of magnitude in signal.

Three changes, none of which the ledger has tested:

- **Online teacher with full-vocabulary KL.** At 1B to 4B the teacher and student fit on the GPU
  together. The top-64 cache and its tail approximation exist only because targets were cached.
  Doc 79's failure mode (mass driven off the teacher's support) cannot occur under full KL. Keep
  the tail-aware cached objective for models too large to co-reside.
- **A hidden-state anchor.** [Doc 79](../79-experiment-042-transfer-failure-analysis.md) recorded
  block-output MSE growing from 12,647 to 105,194 during the long logit-only run. Add a
  per-block normalised hidden-state term so intermediate representations cannot drift while the
  logits are fitted. This is what makes long horizons safe in the published recipes.
- **More data, an EMA of the trainable parameters, and sign movement.** Use 10 to 50 times the
  tokens from the existing mixture, average checkpoints instead of selecting one, and compose the
  confirmed coordinate binary QAT (ledger R3) so discrete capacity moves as well as scales.

**Screen.** On the retained 048 pre-distillation state: {current objective, full KL, full KL plus
anchor} at {256, 2,048, 8,192} steps. Gate on held-out NLL, full KL, above-chance task retention,
and an out-of-domain perplexity.

**Expected value.** This is where published 1-bit quality comes from. I would expect several
points of task retention at 1B, not a jump to source quality.

### 3.3 Token-adaptive scale experts

**Mechanism.** Replace the single `scale_mid` vector of each unit with a mixture of four,
`mid(x) = sum_e g_e(x) * mid_e`, where `g = softmax(W_r x)` comes from one small router per
sublayer reading the residual stream. `scale_mid` is the component-strength vector, so a
token-dependent `mid` re-weights rank components per token: four weight matrices that share every
sign bit.

**Cost.** For Gemma 3 1B, about 0.4 Mbit per block at four experts, roughly 1.5% of the block.
Fund it with rank.

**Why it is not already falsified.** Rank-group scales and cyclic scale banks were static; they
lost to rank. This adds input-dependent capacity, which static rank cannot buy.

**Safety.** Initialise the experts as copies of the current scale and the router at zero. The
model is then bit-for-bit the incumbent at step zero.

**Evidence.** BinaryMoS over OneBit on LLaMA-2-7B: perplexity 8.60 to 7.88, six-task mean 52.4 to
54.0, for about 2% more storage. That is a modest, consistent gain, and I would expect the same
order here.

**Order of work.** Test in PyTorch through the Section 3.2 distillation first. The llama.cpp
kernel needs a per-token `mid` vector between its two stages; do that only after a functional pass.

### 3.4 Budget in bytes: the embedding

The Q8 embedding is 320.9 MB of Experiment 048's 417.3 MB file. The whole decoder is about 90 MB.
Moving the decoder from 1 to 2 BPW costs 82 MB. Storing the embedding at Q5_0 (the type llama.cpp
falls back to for 1,152-wide tensors) would save about 113 MB.

So a 2-BPW decoder with a Q5_0 embedding is about 386 MB, smaller than today's 1-BPW file. The
quality cost of a 5-bit tied embedding is untested here and needs one evaluation run.

This is not only a small-model effect. With a 1-BPW decoder and Q8 token matrices, my arithmetic
from the public architecture sizes puts the embedding share of the file at about 55% for Llama 3
8B (untied input and output matrices), 44% for Gemma 3 12B, 32% for Gemma 3 27B and 21% for
Llama 3 70B. These are estimates, not measured files. For the "very large model at 1 BPW" goal,
the token matrices are the next-largest item after the decoder itself, and they have had no
attention. [CapacityRelaxations.md](CapacityRelaxations.md) §8 proposed a factorized tied
embedding; a plain 4- or 5-bit type is the cheap first step.

### 3.5 Scale remains the largest lever

Unchanged from September: no model at or above 8B has completed, and the cap that binds 20 of 26
`down_proj` units at 1B does not bind at 8B to 70B, where full rank costs 1.28 to 2.0 BPW per
unit. The published 7B results above are the best external evidence of what 1 bit buys with more
parameters. One completed Gemma 3 4B or Llama 3 8B run would tell more about the 1-BPW goal than
any further 1B experiment.

---

## 4. Ideas for 2 BPW and beyond, when ranks are full

### 4.1 Residual factor stages (primary recommendation)

**Representation.** `W ≈ F1 + F2`, each a standard factor unit with its own three scale vectors.
`F1` is the current unit at the physical cap. `F2` is fitted to the Fisher-weighted residual at
whatever rank the budget allows.

**Rates on Gemma 3 1B.** MLP matrices are 89% of decoder parameters and all have the 1:6 aspect
measured in Section 2. Cap plus a rank-832 second stage is 2.01 BPW. Attention `q` and `o` are
already at 1.89 BPW at full rank and need nothing. A third stage or a sign plane reaches 3 BPW.

**What is reused.** ADMM, scale fit, functional block tuning, the packed layout, and the two-stage
kernel. The new work is: a stage index on units in `resident_quantization.py` (increment
`RESIDENT_ALGORITHM_VERSION`), a second response curve in the allocator, and a summed second
evaluation in the packed runtime, GGUF converter and llama.cpp kernel.

**Runtime cost.** Roughly twice the factor kernel time on staged units.

**Fit order.** Greedy: fit and functionally tune `F1`, freeze it, fit `F2` to the residual of the
tuned `F1`, then tune both stages' scales together. Section 2.2 shows naive alternation hurts.

**Functional gate.** Blocks 0, 12 and 25, dense splice, at equal bits against the 060 state:
(a) 060's INT8 columns, (b) cap plus second stage, (c) cap plus partial sign plane. Paired KL on a
fresh 48 by 512 slice. The weight-space gap is about 2x in error energy; that is far larger than
the 1 to 5% effects that failed to transfer in the ledger, but it still has to pass this gate.

### 4.2 Sign planes as the fallback and the fine adjustment

A sign plane is a dense or column-partial `sign(residual)` with a row and a column scale vector.
It is slightly less efficient than a second stage (1.45 against 1.66) but has two uses:

- exact budget matching in steps smaller than a 32-rank increment allows, by coding only the
  highest-residual columns;
- the third bit: on top of two factor stages the residual is whiter, and the plane's efficiency
  does not depend on structure.

It needs one new primitive (a factor unit without the left stage). I would build it only if the
gate in 4.1 shows the second stage tuning badly.

### 4.3 Fix the 2-BPW distillation

Still open from September: Experiments 059 and 060 used the conditional top-64 objective at 2,048
steps, the configuration Doc 79 identified as harmful. Any new 2-BPW state should be distilled with
the Section 3.2 recipe, and the retained 060 state should be re-distilled once as the baseline the
residual-stage run has to beat.

### 4.4 One artifact, several rates

Residual stages nest by construction. A file holding `F1`, `F2` and optionally a plane can be
loaded at about 1.2, 2.0 or 3.0 BPW by dropping later stages, with a separate small set of tuned
scale vectors per operating point. This is a larger step per level than the prefix-ordered rank
idea in the September document, and it falls out of 4.1 for almost no extra work.

### 4.5 Where the binary format stops making sense

In weight space the factor-plus-residual ladder stays ahead of plain scalar quantisers through
about 3 BPW (0.027 to 0.030 at 3.17 BPW against 0.051 for a 3-bit scalar code at 3.0). Beyond that
the margins close, and standard 4-bit tensors have faster kernels. I would treat 1.2 to 3 BPW as
this format's range and not pursue it past 3.

### 4.6 A real external baseline

Run `llama-quantize` IQ2_XXS and IQ2_S on Gemma 3 4B or Llama 3.2 1B, where the 256-column
requirement holds, with an imatrix from the same calibration mixture and the embedding held at the
NanoQuant export's type. Report decoder BPW, not whole-file BPW. Llama 3.2 1B already has a
NanoQuant run (Experiment 025), which makes it the cheapest matched pair.

---

## 5. What I would stop

- **Further fit search on a fixed format.** Experiments 066 to 068 and Section 2.1 close it.
- **Exact-column and structured INT8 sidecars beyond the true outlier channels.** Efficiency 0.30.
- **Multi-bit or real-valued small factors.** Efficiency 0.3 to 0.6.
- **The aspect-ratio hybrid export.** Withdrawn per Section 2.3 and 1.4.
- **Discrete decisions made on block-output error.** Doc 100.
- **WikiText-2 perplexity as the headline number.** Section 1.5.

## 6. Order of work

| Step | Work | Cost | Decides |
|---|---|---|---|
| 1 | Functional gate for residual stages on blocks 0/12/25 (4.1) | 1 to 2 days, PyTorch only | whether the 2x weight-space gain transfers |
| 2 | Uncapped allocation screen at 1.0 BPW (3.1) | 1 day after step 1 | whether 1 BPW gains from above-cap stages |
| 3 | Anchored full-KL distillation ablation on the 048 state (3.2) | 2 to 3 days | how much task retention training signal buys |
| 4 | Token-adaptive scale experts through step 3's recipe (3.3) | 2 days | whether zero-bit dynamic capacity adds to it |
| 5 | Complete 2-BPW run with residual stages and step 3's distillation; re-distilled 060 as control | one full run | the new 2-BPW result |
| 6 | Matched IQ2 baseline on Llama 3.2 1B or Gemma 3 4B (4.6) | hours | external position at true 2 bits |
| 7 | Packed, GGUF and llama.cpp support for stages, only after step 5 passes; Q5_0 embedding evaluation (3.4), which can run at any time | 3 to 5 days | deployability and file size |
| 8 | One model at or above 4B through the 1-BPW recipe (3.5) | RunPod | whether 1 BPW is a quality or an execution problem |

Steps 1 to 4 need no format, kernel or export change. If step 1 fails its KL gate, the fallback is
the partial sign plane (4.2) through the same gate, and the weight-space table in Section 2.2
should then be read as one more proxy that did not transfer.

## Addendum (2026-11-10): product codebook plus a second stage at 1 BPW

Question raised after this document was written: the product codebook (Experiment 057) ran out of ranks before it
reached 1 BPW, so free rows had to fill the budget. Does a coded second stage use those bits better?

Measured on the same block-12 `down_proj` fixture, CPU only, 800 ADMM steps, seed 0, no binary search, greedy second
stage. Rates include 16-bit scale vectors for every stage and the codebook tables. Coded rows use the Experiment 057
format (free left factor, k16 half-word product code on the right factor).

| Arm at about 1.02 BPW | Total rank | BPW | Weighted NMSE | Versus free |
|---|---:|---:|---:|---:|
| Free factors, rank 992 | 992 | 1.0228 | 0.2942 | baseline |
| Coded, 768 free rows, one stage (the Experiment 057 shape) | 1,152 | 1.0195 | **0.2883** | -2.4% at equal bits |
| Coded, 384 free rows, plus coded rank-256 second stage | 1,408 | 1.0188 | 0.2909 | -1.6% |
| Fully coded, plus coded rank-544 second stage | 1,696 | 1.0194 | 0.2905 | -1.7% |

The second stage does not beat the single-stage mixed codebook. Above a fully coded full-rank factor
(0.4163 at 0.686 BPW), converting coded rows to free rows buys 1.59 bits of error reduction per bit and a coded
second stage buys 1.56; both match the 1.54 to 1.66 measured for free second stages above the cap. The second
stage also pays about 0.017 BPW for its own scale vectors on this shape, which is roughly the size of the gap.

The codebook advantage over free factors grows as the rate falls: -6.4% at 0.69 BPW (fully coded, rank 1,152,
0.4163 against an interpolated 0.4447 for free rank 660), -4.6% at 0.85 BPW, -2.4% at 1.02 BPW. Its natural use is
below 1 BPW, where it keeps full rank, and that needs no second stage.

At 1 BPW the whole codebook effect is about 2% of weight error, which matches the full-model record: Experiment 057
against its matched control 056 gave perplexity 248.4 against 253.1 and a six-task mean of 0.4625 against 0.4570.
Combining residual stages with product coding is currently rejected by `fit_residual_stages` and config rule
CFG181; this measurement does not justify lifting that. Script and results: `outputs/second-opinion-2026-11-10/part5.py`
and `part5_*.json`.
