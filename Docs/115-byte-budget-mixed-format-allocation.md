# Byte-budget allocation across binary stages and llama.cpp formats

Written 2026-11-16. Status: design, with a CPU pilot on Gemma 3 1B. Nothing here has been
compressed or scored on a model yet.

## Summary

Every decoder projection gets a menu of storage options: one, two or three full-rank binary
stages, or a standard llama.cpp format. Each option has a byte cost and an estimated KL. A
solver picks one option per projection so the summed estimate is as low as possible within a
total file size.

A pilot built that menu for `google/gemma-3-1b-it` from Experiment 080's export and sixteen
llama.cpp quantizations of the same checkpoint. All numbers are the additive estimate from
`Results/080/080-imatrix-error-attribution.md`, not measured model KL.

| File size | Allocation | Estimated KL (nats/token) |
| ---: | --- | ---: |
| 547.4 MB | Experiment 080: two stages everywhere | 3.66 |
| 547.4 MB | Mixed allocation | 2.41 |
| 508 MB | Mixed allocation that matches 080's estimate | 3.66 |
| 706.0 MB | 080's Q8_0 hybrid (26 projections over 1% share) | 1.49 |
| 706.0 MB | Mixed allocation | 0.38 |

Three things in the pilot change the plan as you described it.

- **080 has two full-rank stages, not three.** Its recipe is 074's two-stage recipe at 2.515
  bits per weight. Three stages are Experiment 081, on 270M, which has not run. The third
  stage below is an assumption, marked as such.
- **The switch to a standard format pays off before the third stage.** For MLP down and
  attention output, a 3-bit format already matches or beats two stages at the same bytes. A
  third stage is only competitive for MLP gate and up.
- **Q6 and Q4 K-quants fit only two of the seven matrix types.** The other five have 1,152
  inputs, which is not a multiple of 256, so llama.cpp stores them in 32-wide formats: IQ4_NL,
  Q4_1, Q5_0, Q5_1 and Q8_0.

The estimate has not been checked against measured KL on 1B, and the one model-level test we
have (the Q8_0 hybrid) delivered a third to a half of the predicted gain. Section 8 covers that
risk, and section 10 puts a measurement gate ahead of any implementation.

## 1. What the 080 report shows

- **Error is concentrated.** Blocks 0–3 hold 44% of the estimated KL. Blocks 9–25 hold 33% of
  it and take 65% of the projection bytes, because every block gets the same 8.44 MB.
- **The MLP holds 88% of it.** Attention key and value together are 2%.
- **Most of the file is not projections.** Of 547,391,648 bytes, 219,368,448 are decoder
  projections. The other 328,023,200 are fixed, and 320,864,256 of those are the token
  embedding at Q8_0. A total byte budget therefore has to include the embedding.

## 2. The problem

Choose one option `o(u)` for each storage unit `u` to

```
minimise   Σ_u  KL_u(o(u))
subject to Σ_u  bytes_u(o(u))  ≤  budget − fixed bytes
```

**Storage units.** A unit is whatever must be stored together. Gemma 3 1B has 130: per block,
MLP gate, MLP up, MLP down, attention output, and the stacked QKV group. The QKV group is one
unit because its staged form shares factors across query, key and value. When it uses standard
formats, the three matrices are stored separately and may each take a different format.

**Estimated KL of an option.** For each matrix in the unit:

1. Take the option's reconstructed weights and subtract the original.
2. Weight the squared error of each input column by its mean-one imatrix importance, sum,
   divide by the squared norm of the original, and take the square root. This is the
   noise-equivalent error the 080 report already prints.
3. Put that error through the matrix's fitted curve from the sensitivity sweep.

Below the lowest error the sweep measured (5% on 1B), the estimate falls off as the square of
the error from the curve's value at 5%. The fitted exponent is used inside and above the range.
Some block-0 curves have exponents near 1.6, and extending those downward would overstate
the cost of the precise formats.

**Bytes of an option** are the stored tensor bytes in the GGUF: signs plus scales for every
stage, or the quantized tensor.

## 3. The menu

### Staged binary factors

| Option | Source of its error in the pilot |
| --- | --- |
| 1 stage | The first term of 080's two-stage export, alone. Likely a little pessimistic, since that term was tuned as half of a pair. |
| 2 stages | 080's export, as reported. |
| 3 stages (assumed) | Not measured. Bytes are 1.5× two stages. KL assumes the third stage repeats the second stage's gain. |

On 080 the second stage multiplies the weighted error by a median of 0.59–0.63 for the MLP
matrices and 0.40–0.57 for attention. Later stages fit a residual with less structure, so repeating that
gain is optimistic. The pilot reports results with and without the third stage.

### Standard llama.cpp formats

A format's block size must divide the matrix's input width. This is checked in
`tensor_type_fallback` in `llama-quant.cpp`.

| Matrix | Inputs | 256-wide formats (K-quants, IQ2, IQ3, IQ4_XS) | 32-wide formats |
| --- | ---: | --- | --- |
| MLP down | 6,912 | Yes | Yes |
| Attention output | 1,024 | Yes | Yes |
| MLP gate, MLP up | 1,152 | No | Yes |
| Attention query, key, value | 1,152 | No | Yes |
| Token embedding | 1,152 | No | Yes |

So for five of the seven types the ladder is IQ4_NL (4.5 bits), Q4_1 (5.0), Q5_0 (5.5), Q5_1
(6.0) and Q8_0 (8.5). Nothing standard exists for them below 4.5 bits. Asking llama-quantize
for Q6_K on those matrices produces Q8_0, and Q4_K produces Q5_0.

All standard options are quantized with the same imatrix the staged recipe uses.

### Token embedding

The embedding is one more unit with the same 32-wide ladder. Its KL cannot come from the
sensitivity curves, which cover projections only. It needs one direct KL measurement per
format. The bytes at stake are large next to the 219 MB of projections:

| Embedding format | Bytes | Freed against Q8_0 |
| --- | ---: | ---: |
| Q8_0 (today) | 320,864,256 | |
| Q5_1 | 226,492,416 | 94.4 MB |
| Q5_0 | 207,618,048 | 113.2 MB |
| Q4_1 | 188,743,680 | 132.1 MB |
| IQ4_NL or Q4_0 | 169,869,312 | 151.0 MB |

On 270M the embedding cost 0.003 nats/token at Q8_0 and 0.10 at Q5_1. If 1B behaves the same
way, moving it to Q5_1 and spending the 94 MB on projections is a clear win. The pilot leaves
it fixed at Q8_0 because that cost is unmeasured on 1B.

## 4. The measured menu

Estimated KL summed over all 26 blocks, as if every matrix of the type used the option.
Bits are per weight.

**MLP gate and MLP up (55% of 080's estimate)**

| Option | Bits | Gate | Up |
| --- | ---: | ---: | ---: |
| 1 stage | 1.19 | 2.658 | 2.910 |
| 2 stages | 2.37 | 0.968 | 1.048 |
| 3 stages (assumed) | 3.56 | 0.356 | 0.380 |
| IQ4_NL | 4.50 | 0.115 | 0.126 |
| Q4_1 | 5.00 | 0.096 | 0.098 |
| Q5_0 | 5.50 | 0.038 | 0.052 |
| Q5_1 | 6.00 | 0.023 | 0.023 |
| Q8_0 | 8.50 | 0.001 | 0.001 |

**MLP down (33%)**

| Option | Bits | Estimated KL |
| --- | ---: | ---: |
| 1 stage | 1.19 | 2.977 |
| IQ2_XXS | 2.06 | 1.533 |
| 2 stages | 2.37 | 1.206 |
| IQ2_S | 2.56 | 0.939 |
| IQ3_XXS | 3.06 | 0.461 |
| IQ3_S | 3.44 | 0.308 |
| 3 stages (assumed) | 3.56 | 0.493 |
| IQ4_XS | 4.25 | 0.117 |
| Q4_K | 4.50 | 0.093 |
| Q5_K | 5.50 | 0.026 |
| Q6_K | 6.56 | 0.007 |
| Q8_0 | 8.50 | 0.001 |

**Attention output (5%)**

| Option | Bits | Estimated KL |
| --- | ---: | ---: |
| 1 stage | 1.93 | 0.681 |
| IQ2_S | 2.56 | 0.577 |
| IQ3_XXS | 3.06 | 0.283 |
| IQ3_S | 3.44 | 0.181 |
| 2 stages | 3.86 | 0.182 |
| IQ4_XS | 4.25 | 0.064 |
| Q4_K | 4.50 | 0.052 |
| Q5_K | 5.50 | 0.014 |
| Q6_K | 6.56 | 0.004 |

**Stacked query, key and value (7%)**

| Option | Bits | Query | Key | Value |
| --- | ---: | ---: | ---: | ---: |
| 1 stage | 1.78 | 0.567 | 0.187 | 0.245 |
| 2 stages | 3.57 | 0.179 | 0.029 | 0.047 |
| IQ4_NL | 4.50 | 0.024 | 0.019 | 0.093 |
| Q5_0 | 5.50 | 0.010 | 0.008 | 0.043 |
| Q5_1 | 6.00 | 0.004 | 0.003 | 0.016 |
| Q8_0 | 8.50 | 0.000 | 0.000 | 0.001 |

What the menu says:

- **Attention is expensive in binary form.** Its matrices are close to square, so two
  full-rank stages already cost 3.6–3.9 bits. For attention output, IQ3_S gives the same
  estimate with fewer bits, and Q4_K gives less than a third of it for 0.64 bits more.
- **MLP down has a full ladder of standard formats.** Two stages sit close to the line
  between IQ2_XXS and IQ2_S. From 3 bits up, the standard formats are well ahead, and IQ3_S beats the
  assumed third stage with fewer bits.
- **MLP gate and up have a gap.** Nothing standard exists between two stages at 2.37 bits and
  IQ4_NL at 4.50. This is the one place a third stage could earn its bytes.
- **Value matrices favour binary factors.** Two stages at 3.57 bits beat IQ4_NL at 4.50 for
  the value matrix. Query is the opposite.
- **Q8_0 is rarely the right step.** Q5_1 removes about 98% of a two-stage matrix's estimate
  for 6.0 bits. Q8_0 spends another 2.5 bits on the last 2%.

## 5. The solver

This is a multiple-choice knapsack: one option per unit, one shared budget.

1. For each unit, list its options. For a QKV group, the standard options are every mix of
   formats over the three matrices.
2. Keep the efficient ones: drop any option that costs more bytes for no less KL, and any that
   lies above the straight line between two others. What remains has falling returns per
   byte.
3. Start every unit at its cheapest option. Take upgrades in order of KL saved per byte,
   across all units, while they fit.

The result is optimal except for at most one upgrade that did not fit, which is small against
130 units. It runs in milliseconds, so the whole size-against-KL curve can be produced along
with any single budget. The pilot implementation is `allocate` in
`tools/probe_byte_budget_allocation.py`.

**Family multipliers.** The objective takes one multiplier per option family, default 1. They
exist because isolated estimates treat the two families unequally (section 8) and are set from
measurement, not assumed.

**Guard rails.**

- An option whose error is outside the sweep's measured range is flagged. In the pilot, 66 of
  the 182 one-stage errors are above the 35% ceiling.
- A unit may be pinned to an option by hand, for experiments and for bisecting a regression.

## 6. Pilot allocations

Embedding fixed at Q8_0. Budgets are whole-file bytes.

| Budget | Estimated KL | With assumed third stage | Units staged / standard |
| ---: | ---: | ---: | --- |
| 460 MB | 6.59 | 6.59 | |
| 500 MB | 3.97 | 3.90 | |
| 547.4 MB (080's size) | 2.41 | 2.36 | 65 / 65 |
| 600 MB | 1.34 | 1.32 | 50 / 80 |
| 650 MB | 0.78 | 0.76 | 33 / 97 |
| 706.0 MB (Q8_0 hybrid's size) | 0.38 | 0.36 | 10 / 120 |
| 800 MB | 0.09 | 0.09 | 0 / 130 |
| 900 MB | 0.03 | 0.03 | 0 / 130 |

For comparison at the same budgets:

| Variant | 547.4 MB | 650 MB | 706.0 MB |
| --- | ---: | ---: | ---: |
| Full menu | 2.41 | 0.78 | 0.38 |
| Stages plus Q8_0, Q6_K, Q4_K and Q4_0 only | 2.69 | 1.00 | 0.56 |
| Standard formats only (smallest possible file is 648 MB) | not possible | 2.30 | 0.41 |
| Stages only, with the assumed third stage | 2.74 | 1.35 (all three stages, 657 MB) | |
| 080's Q8_0-over-1% rule | | | 1.49 |

**What the 547.4 MB allocation does.** It moves bytes from the late blocks to the early ones.

- Blocks 0–3 go from 33.7 MB to 57.9 MB and their estimate falls from 1.61 to 0.29. Every
  matrix there becomes a standard format: IQ4_NL for gate and up, Q4_K or IQ3_S for down,
  Q5_K or Q4_K for attention output.
- Blocks 9–25 go from 143.4 MB to 111.6 MB and their estimate rises from 1.21 to 1.62. Gate,
  up and QKV mostly drop to one stage, and MLP down drops to IQ2_XXS or IQ2_S.
- Inside a QKV group that leaves binary form, the value matrix usually gets Q5_1 while query
  and key get IQ4_NL.

## 7. What this says about a third stage

- With the optimistic assumption, offering a third stage lowers the estimate by at most 4% at
  any budget. The solver uses it for MLP gate and up only.
- For MLP down, attention output and QKV it is never on the efficient set. For the first two,
  a 3-bit standard format is cheaper and better.
- From about 800 MB the solver uses no binary factors at all.

So on this model the answer to "when two stages are not enough" is a standard format, chosen
per matrix type by what its shape allows. A third stage is a narrow option worth one small
probe on MLP gate and up, not a recipe-wide setting.

One alternative would close the gate and up gap directly: a 256-wide format that accepts
1,152-wide rows, added to the modified llama.cpp. That would give those matrices the same
3-bit ladder MLP down has. It is a runtime and quantizer change and is not part of this
design.

## 8. Why the estimate cannot be trusted yet

**The one model-level test under-delivered.** 080's Q8_0 hybrid replaced 26 projections that
carry 59% of the estimate. Measured against BF16:

| Measure | Predicted change | Measured change |
| --- | ---: | ---: |
| Additive KL estimate | −59% | |
| WikiText-2 loss gap to BF16 | | −19% (0.303 → 0.244 nats) |
| MMLU accuracy gap to BF16 | | −30% (7.7 → 5.4 points) |

There are two possible reasons, and the data does not separate them.

- **The estimate overweights those projections.** It has not been checked against isolated or
  joint KL on 1B. Its total is 3.66 nats/token for a model whose WikiText loss gap is 0.30, so
  its absolute scale is clearly too high.
- **The swap broke compensation.** Binary factors are fitted block by block, with each block
  seeing the compressed output of the blocks before it. On 270M that made the whole-file KL
  51–61% of the sum of the isolated costs, against 82–92% for llama.cpp files. Replacing early
  projections after the fact leaves later blocks compensating for an error that is no longer
  there.

**Consequences for the design.**

- The allocation has to be decided before compression, so each block is fitted against the
  mixed model it will actually run in. A post-hoc swap is a lower bound, not the method.
- Isolated estimates undervalue binary factors against standard formats. Applying the 270M
  ratios as family multipliers (0.6 for staged, 0.87 for standard) changes 38 of the 130
  choices at 547.4 MB, and 16 of them change family. The main shift is MLP down keeping two
  stages in ten blocks where the unadjusted solver picks a low-bit format. The close calls
  between two stages and a 2-bit or 3-bit format are therefore undecided. The moves to 4-bit
  and higher formats in the early blocks are not close and survive the adjustment.

**Other limits of the pilot.**

- The 1B sweep ran in FP16 at 5–35% error. Everything at 4 bits and above lies below that range.
- One imatrix and one calibration text. The imatrix that weights the error is the one the
  quantizers optimise against.
- The one-stage numbers come from a truncated two-stage fit, and the three-stage numbers are
  assumed.

## 9. Pipeline design

### Plan (CPU, before compression)

1. **Standard options.** Run llama-quantize once per format with the recipe's imatrix,
   dequantize, and compute each matrix's bytes and weighted error. This is exact and takes
   about fifteen seconds per format on 1B.
2. **Staged options.** Use measured errors from a control run of the same recipe when one
   exists, as 080 does for this model. Otherwise run the existing rank-response probes
   (`domain/planning.py`, `allocation_profile_dataset_workflow.py`) with imatrix-weighted
   error as the response, at one, two and optionally three stages.
3. **Embedding.** Measure KL directly for each candidate format, once per model.
4. **Solve** for the byte budget and write a storage plan: for every unit, either a stage
   count with ranks or a format per matrix, with predicted bytes and predicted KL.

This is the existing D2 flow (control run, profile, allocate) with a byte objective and a
wider menu. The storage plan is a new persisted artifact and belongs in the run identity.

### Compress

- **Staged units** are fitted and tuned as today, with the planned stage count per unit
  instead of one recipe-wide ceiling (`overcomplete_rank_ceiling_fraction`).
- **Standard units** are quantized by llama-quantize up front, from the same source and
  imatrix that the export will use. The resident flow loads their dequantized weights as
  frozen dense layers. They are not trained.
- **Order is unchanged.** Blocks are still processed front to back, so a staged block is
  fitted against upstream activations that already contain the standard units' real error.
  This is what a post-hoc swap cannot do.
- **A QKV group in standard form** becomes three plain layers. The shared-QKV path needs a
  branch for that case.
- Bump `RESIDENT_ALGORITHM_VERSION`, since commits from a mixed plan are not interchangeable
  with uniform ones.

### Export

- The converter writes staged sidecars for staged units, and full-precision tensors for
  standard units and the embedding.
- llama-quantize then applies the plan through `--tensor-type-file`. `gguf_export.py` already
  drives the quantizer with per-tensor types for the embedding, in `_quantizer_tensor_policy`.
- **Gate:** every standard tensor in the final GGUF must dequantize to exactly the weights the
  resident flow used. Same quantizer, source and imatrix make this achievable, and a hash
  check makes it enforced.
- **Gate:** exported bytes per unit must equal the plan's predicted bytes.

### Runtime

The Q8_0 hybrid shows a GGUF can hold both staged and standard tensors, but it was scored
through the dense PyTorch replay only. Native execution of a mixed file in the modified
llama.cpp, including the IQ formats, needs its own check.

## 10. Order of work

Each step is a gate for the next.

1. **Validate the estimate on 1B (GPU, hours).** Measure joint KL for 080 and for the Q8_0
   hybrid on the sweep's text, and isolated KL for a sample of menu options. Decide the
   family multipliers from that, or decide the estimate needs a different objective.
2. **Extend the 1B sweep downward in FP32,** to 1% and 2.5% error, so the 4-bit and higher
   options are inside the measured range. The BF16 sweep on 270M showed a noise floor near
   0.003 nats/token at these levels, so FP32 is required.
3. **Measure the embedding** at Q8_0, Q5_1, Q5_0 and IQ4_NL.
4. **Post-hoc mixed file at 547.4 MB,** built like the Q8_0 hybrid and scored on the same
   protocol. It costs one evaluation and gives a floor for the gain.
5. **Plan-time mixed compression at 547.4 MB,** the real test. Success is lower measured KL
   and perplexity than 080 at the same bytes.
6. **Third-stage probe** on MLP gate and up for a few blocks, only if step 5 leaves those
   units as the main remaining error.

## 11. Open questions

- Should the objective be the additive estimate at all, or a measured block-level KL once
  step 1 is in? The solver does not care; only the menu's KL column changes.
- Does llama.cpp's quantization of a standard unit improve if it is given the same row
  weighting the staged recipe can use? Today it uses columns only.
- Can a staged unit be tuned to compensate for a frozen standard unit in the same block? The
  block order above gives this for later blocks, not within one.
- Are sub-full-rank single stages worth adding below one stage for the most tolerant blocks?
  The allocation datasets already hold rank-response curves that could price them.

## Reproduction

```powershell
.\.venv\Scripts\python.exe tools\probe_byte_budget_allocation.py `
  Datasets\gemma-3-1b-it-fp16-sensitivity-16x256\measurements\022fc4f64542b546eebd3bbbe21ee50cf36d36946dcd0a871d20ae5e24abfc6f\results.json `
  --imatrix Datasets\gemma-3-1b-it-imatrix-unsloth\imatrix_unsloth.dat `
  --staged Results\080\gemma-3-1b-it-nanoquant.gguf `
  --standard <each GGUF in Datasets\gemma-3-1b-it-gguf-local-quantize> `
  --budget 547391648 --output outputs\byte-budget-allocation-pilot\gemma-3-1b-it-080.md
```

- Quantized inputs and the commands that made them: `Datasets/gemma-3-1b-it-gguf-local-quantize/source.json`
  (17.6 GB, regenerable in a few minutes on the CPU, llama.cpp commit `2927569`).
- Menu, allocations and report: `outputs/byte-budget-allocation-pilot/gemma-3-1b-it-080.{menu.json,json,md}`.
- The two-stage column reproduces the published 080 attribution to within 0.1%. The small
  difference is the below-range rule in section 2.
