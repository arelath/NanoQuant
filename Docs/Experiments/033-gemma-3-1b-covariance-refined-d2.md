# Experiment 033: Gemma 3 1B covariance-refined D2 compression

## Status

**Completed and rejected on 2026-07-29.** The complete candidate passed
integrity, resume, export, and BPW gates, but its protocol-matched WikiText
perplexity was 19.26% worse than Experiment 022. The covariance refinement
must remain experimental rather than become the default recipe.

- Model: `google/gemma-3-1b-it`
- Launcher:
  `experiments/033-covariance-refined-d2-compress-and-benchmark-gemma-3-1b-it.py`
- Baseline: Experiment 022

## Question

Does the same-format covariance-aware binary refinement promoted by Documents
43–49 improve the complete Experiment 022 D2 result after resident tuning,
global distillation, packing, GGUF export, and retained quality evaluation?

## Controlled change

Experiment 033 is structurally identical to Experiment 022 except for the
calibration objective that explicitly enables the resident refinement:

| Setting | Experiment 022 | Experiment 033 |
| --- | --- | --- |
| Objective kind | `diagonal` | **`dense_hessian`** |
| Covariance fit rows per block | none | **8,192** |
| Refinement | none | **32 left-coordinate steps, 16 right batches** |

The selected refinement settings are fixed production constants. They were
chosen before this run from the independent sample-size and depth screens,
not tuned against Experiment 033 quality.

All other recipe settings remain matched to Experiment 022: fused QKV,
0.6-shrunk diagonal Fisher importance, fresh same-campaign exact-unit KL
measurement, calibration-weighted measured rank responses, D2 allocation,
per-layer and factorized tuning, post-block refit, global distillation, export,
and the 64-by-128 retained WikiText protocol.

The refinement changes neither rank nor factor storage. It updates the binary
signs and three diagonal scales of the already accepted factorization before
factorized tuning. Fused QKV, O, gate, and up projections are eligible.
Gemma's 6,912-wide down projections remain unchanged because that dense solve
was outside the validated memory and numerical screen.

## Prior evidence

The final independent 26-block static screen at exactly 0.999472370 factor
BPW reported:

- held-out covariance error −24.06%;
- joint KL −12.85%;
- NLL −0.455772 nats/token;
- 104/104 refined groups and all 26 block aggregates improved;
- all three retained block-output comparisons improved;
- 20.74 seconds of refinement versus 684.23 seconds of ADMM.

Those results justify this complete run but do not predict its outcome.
Experiment 032 showed that a strong static functional result can reverse
after allocation, tuning, and distillation.

## Success criteria

- the fresh uniform control, exact-unit KL profile, and D2 rank-response
  profile complete under the covariance-refined recipe;
- all 26 candidate blocks and the complete transitive artifact graph pass
  strict validation;
- logical, packed, checkpoint, GGUF, and export-summary outputs complete;
- the retained WikiText protocol and token identity match Experiment 022;
- effective BPW does not exceed Experiment 022's 1.024494712;
- candidate perplexity improves on Experiment 022's 228.550618;
- task accuracy, ranks, block losses, runtime, peak memory, and artifact bytes
  are reported regardless of the primary quality outcome.

## Interpretation guard

The same-campaign uniform control is also covariance-refined, so the exact-unit
KL anchors compare projection removals at the candidate's operating point.
Measured rank responses remain the ordinary calibration-weighted response
model used by Experiment 022; covariance refinement is deliberately treated
as a post-factorization optimizer, not silently substituted for the D2
allocation objective.

No promotion decision will be made from block losses alone. The decisive gate
is protocol-matched retained quality after the complete compression lifecycle.

## Result

### Recovery and integrity

The first process completed the 26-block uniform control, all 162 exact-unit
KL arms, and candidate blocks 0–4 before the computer lost power during block
5. The ordinary zero-argument launcher then:

- reused the completed control and KL profile;
- loaded the five durable candidate blocks and block-4 activation boundary;
- replayed the incomplete block deterministically;
- completed all remaining blocks and 2,048 global-distillation steps;
- exported logical, packed, checkpoint, and GGUF representations;
- ran the retained quality suite.

The fresh strict audit is
`evidence/033/033-covariance-refined-d2-compress-and-benchmark-gemma-3-1b-it/strict-validation.json`
with SHA-256
`f29df9f034103dc37ce7d3b4e69b286c6fa526aa3bbd5f6dc29babd89d3ccecd`.
It validates:

- one active commit identity and no inactive records;
- 156 journal records, 26 contiguous block records, and 130 unit records;
- 708 transitive artifacts and 10,182,211,031 artifact bytes;
- 104 covariance-refinement artifacts;
- exact complete logical and packed representations.

The completed GGUF is 417,334,656 bytes with SHA-256
`3c0733573f72bbe8374ac076f61ea28cc41cdc4b4f26bf8118f7322429d6cb63`.

### Storage gate passes

| Metric | Experiment 022 | Experiment 033 | Change |
| --- | ---: | ---: | ---: |
| Effective BPW | 1.024494712 | **1.024427205** | −0.000067507 |
| Total unit rank | 111,776 | 111,904 | +128 |
| Packed quantized-layer bytes | 89,480,656 | **89,474,768** | −5,888 |
| GGUF bytes | 417,340,544 | **417,334,656** | −5,888 |

The higher total rank and lower physical size are not contradictory. The D2
plan moved ranks among differently shaped units, reducing binary-factor bits
by 49,152 while adding 2,048 scale bits.

The fresh covariance-aware KL profile changed 39/130 ranks relative to
Experiment 022:

| Unit type | Changed units | Rank delta |
| --- | ---: | ---: |
| `mlp.down_proj` | 7 | −352 |
| `mlp.gate_proj` | 14 | +512 |
| `mlp.up_proj` | 12 | −224 |
| fused QKV | 4 | +96 |
| `self_attn.o_proj` | 2 | +96 |
| **Total** | **39** | **+128** |

### Local covariance objective succeeds

The resident implementation did what it was designed to do before tuning.
All 104 eligible groups reduced their captured fit-covariance error:

| Unit type | Groups | Mean reduction | Range |
| --- | ---: | ---: | ---: |
| `mlp.gate_proj` | 26 | 35% | 30–43% |
| `mlp.up_proj` | 26 | 37% | 32–44% |
| fused QKV | 26 | 32% | 27–38% |
| `self_attn.o_proj` | 26 | 54% | 37–69% |
| **All** | **104** | **39.25%** | **27.08–69.16%** |

The refinement itself took 30.55 seconds. This confirms the domain algorithm,
resident wiring, and persistence contract. It does not establish language
quality.

### Decisive retained quality gate fails

The final comparison uses the same pinned model and revision, tokenizer hash,
64-by-128 WikiText protocol, and token hash
`sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`.

| Metric | Experiment 022 | Experiment 033 | Change |
| --- | ---: | ---: | ---: |
| Mean NLL after KD | **5.431758** | 5.607859 | +0.176102 |
| Perplexity after KD | **228.550618** | 272.560153 | **+44.009535 (+19.26%)** |
| Mean limited-task primary score | 0.4692 | **0.4717** | +0.0025 |

The limited tasks are mixed:

| Task | Experiment 022 | Experiment 033 | Change |
| --- | ---: | ---: | ---: |
| PIQA `acc_norm` | 0.605 | 0.620 | +0.015 |
| ARC Easy `acc_norm` | 0.380 | 0.415 | +0.035 |
| ARC Challenge `acc_norm` | 0.215 | 0.190 | −0.025 |
| HellaSwag `acc_norm` | 0.460 | 0.445 | −0.015 |
| WinoGrande `acc` | 0.520 | 0.520 | 0.000 |
| BoolQ `acc` | 0.635 | 0.640 | +0.005 |

The 0.0025 mean task increase over six 200-row point estimates cannot
overturn the dense WikiText regression.

### Pre-KD evaluation rules out distillation as the sole cause

Both immutable resident states were evaluated again with
`tools/evaluate_wikitext.py --ignore-global-tuning` on the same token hash:

| Pre-KD metric | Experiment 022 | Experiment 033 | Change |
| --- | ---: | ---: | ---: |
| Mean NLL | **5.611360** | 5.810259 | +0.198899 |
| Perplexity | **273.516089** | 333.705584 | **+60.189496 (+22.01%)** |

The retained receipts are `Results/022/022-pre-kd-wikitext.json` and
`Results/033/033-pre-kd-wikitext.json`. Global distillation improves both
models and narrows the relative gap slightly; it does not create the
regression.

The matched distillation block-snapshot proxy is therefore misleading for
this decision. Before KD it favors Experiment 033 in 25/26 blocks by 7.90% on
average; after KD it favors Experiment 033 in 17/26 blocks by 1.42% on
average. Yet the actual pre- and post-KD WikiText evaluations both favor
Experiment 022. Experiment 033's final epoch top-k loss is also 0.83% worse
(1.956931 versus 1.940916). Neither local block snapshots nor the training
loss is a sufficient promotion metric.

### Same-rank prefix localizes a pre-allocation interaction

Blocks 0–6 use the same 35 unit ranks in both experiments. Their committed
resident boundary losses are all worse in Experiment 033, by 2.67% to 8.51%
(mean 5.05%). Rank redistribution therefore cannot explain the earliest
regression. The likely interaction is between the covariance-refined
initialization and the existing non-factorized/factorized/post-block tuning
objectives. Allocation changes after block 6 add a second confound but are not
the original cause.

The result also explains why the successful static screen did not transfer.
That screen measured fixed-rank untuned splices directly on held-out
sequences. The complete recipe subsequently optimizes a different sequence of
teacher-reconstruction objectives, and the locally improved binary state does
not survive that composition as a language-quality gain.

### Runtime, memory, and artifact cost

| Measurement | Experiment 022 | Experiment 033 | Change |
| --- | ---: | ---: | ---: |
| Sum of committed block wall time | 7,284.68 s | 8,176.79 s | +12.25% |
| Strictly reachable artifact bytes | 9,276,915,157 | 10,182,211,031 | +9.76% |
| Peak CUDA allocator bytes | 9,196,011,520 | 7,784,628,224 | −15.35% |
| Peak process host bytes | 15,339,757,568 | 12,052,140,032 | −21.43% |

The shutdown and resume make launcher wall time unsuitable for a direct speed
comparison. The committed block-time sum excludes the abandoned partial block
but is the least ambiguous compute measure. Memory differences are
descriptive because the runs occurred under different workstation
conditions. The extra artifact storage is expected from 104 persisted
refinement states.

## Verdict

Reject pre-tuning covariance-aware binary refinement for the production D2
recipe. Keep the implementation behind its explicit dense-Hessian option for
research and resume compatibility, but do not enable it by default.

The next bounded probe must hold Experiment 022 ranks and outliers fixed and
compare refinement placement:

1. ordinary tuned state;
2. refinement before factorized tuning, as Experiment 033 did;
3. refinement after factorized tuning;
4. refinement after post-block refit.

It should begin on representative same-rank blocks and require held-out block
output plus direct WikiText/KL improvement before another complete numbered
run. This isolates tuning placement from D2 reallocation and avoids spending
another full campaign on the already-falsified pre-tuning configuration.
