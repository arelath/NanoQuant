# Experiment 040: Low-Pressure Correction with a Minimal Confidence Fold

## Status

Complete and accepted. The selected candidate passed the untouched WikiText,
task, C4, resident-validation, packed-reload, GGUF, and effective-BPW gates.
It reused retained Experiment 038 training evidence; no new factorization or
KD training was needed.

## Rationale

Experiment 039 met the mass floor with trained parameters alone and nearly
preserved task quality, but the high one-sided coefficient spent enough
capacity that pinned C4 NLL became neutral rather than confidently better.
Experiment 038's lower-pressure weight-2.0 checkpoint had substantially more
WikiText NLL headroom, but missed absolute mass by only 0.0111 on its former
confirmation slice.

The original 1.06 fold had to repair a much larger raw-tail mass deficit and
significantly harmed C4 NLL. A fold near 1.01 should cost much less. This
experiment tests whether combining the task-friendlier, lower-pressure
32-step correction with the smallest sufficient fold dominates both earlier
endpoints.

## Fixed protocol

- trained checkpoint: Experiment 038 ratio-0.80, weight-2.0, epoch 1,
  `sha256-56dc4b560ed2c73965a02b5a4bb50945aa05fa6f62763171e2f486e207bbfa24`;
- development slice: WikiText validation offset 104, now released by the
  rejected Experiment 038;
- folded final-RMSNorm scales screened once: 1.005, 1.010, 1.015, and 1.020;
- selection: lowest scale reaching mass at least 0.75 while retaining NLL and
  KL improvements over conditional KD;
- untouched WikiText confirmation: validation offset 300, 48x512, token hash
  `sha256:9ee31088130e314637ed3607ddf7903890438c0aa7e62b1cd261b971479a4aef`;
- the scale and checkpoint freeze before offset 300 is evaluated.

If no screened scale passes development, the experiment is rejected. If a
scale passes offset 300, proceed to the complete task benchmark and pinned C4
gate. Final acceptance still requires statistically supported C4 NLL and KL
improvement, no established task regression, unchanged effective BPW, normal
materialization, validation, and the complete export contract.

## Development selection

The four predeclared scales produced:

| Final-norm scale | Development mass | NLL | Full KL |
| ---: | ---: | ---: | ---: |
| 1.005 | 0.74324 | **4.34992** | 1.28213 |
| 1.010 | 0.74613 | 4.35306 | 1.28148 |
| **1.015** | **0.75016** | 4.35787 | 1.28053 |
| 1.020 | 0.75436 | 4.36319 | **1.27998** |

Scale 1.015 was the lowest passing value and was frozen before the offset-300
slice was opened. This is much smaller than Experiment 037's 1.06 fold because
the short one-sided correction supplies most of the missing selected mass.

## Untouched WikiText confirmation

On validation offset 300, the folded candidate reached mass 0.76308. Against
conditional KD it improved NLL from 4.49452 to 4.40061, full KL from 1.58588
to 1.22031, and top-k-plus-tail KL from 1.51235 to 1.14943. The evaluated token
hash exactly matched the predeclared
`sha256:9ee31088130e314637ed3607ddf7903890438c0aa7e62b1cd261b971479a4aef`.

## Quality and transfer

The factorized 64x128 WikiText plus six-task/200-example result improved PPL
from conditional KD's 187.5229 to 172.6285. Its task mean was 0.47167 versus
0.47583. On the larger 1,000-example inventory the means were 0.45283 and
0.45317, a delta of -0.00033 with a paired task-stratified 95% interval
[-0.00650, +0.00583]. No task regression was established.

Pinned C4 supplied the decisive transfer gate:

| Metric | Conditional KD | Candidate | Candidate delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| NLL | 4.95102 | **4.89349** | -0.05753 | [-0.07685, -0.03830] |
| KL | 1.28124 | **1.05489** | -0.22634 | [-0.24586, -0.20838] |

Both improvements are supported with confidence. This fixes Experiment 039's
failure: lower trained pressure preserves NLL headroom, while the minimal fold
repairs the remaining mass deficit cheaply.

## Materialization, packed quality, and export

The selected checkpoint was materialized and then folded as active global
tuning artifact
`sha256-8f1d413a7ed4ebe8fefacb9b9326b4c201dec4e6f81d6fbcdf12ae52ce8eb914`.
Fresh validation covered 708 transitive artifacts, all 26 blocks, and 130
committed layers. Effective BPW remained 1.024494712.

The complete compression workflow streamed that derived state into logical
format, validated an exact logical-to-packed conversion, built the llama.cpp
checkpoint, and exported the GGUF. The packed quantized-layer payload is
89,480,664 bytes. The GGUF is 417,340,544 bytes with SHA-256
`ff29b0e1960638c6b542d98b0b81cd505a4b72d676b970205278c1da27457196`.
After packed quality completed, the validated regenerable logical artifact
(2,739,803,993 filesystem bytes, descriptor SHA-256
`ea44f0cefcf371e30ae6f3e781db8413be7e1ef39fff34892e014a3f13d23436`)
was removed with the guarded cleanup tool. The resident state, packed artifact,
llama.cpp checkpoint, GGUF, receipts, and quality reports remain.

Quality was then rerun from `outputs/040/packed`, not merely from the research
factorized state. Packed WikiText PPL was 172.7052, only 0.0767 above the
factorized result. On six tasks at 1,000 examples each, packed mean was 0.45183
versus conditional KD's 0.45317. The paired delta was -0.00133 with interval
[-0.00750, +0.00483]. Packed versus factorized delta was -0.00100 with interval
[-0.00417, +0.00217]. Neither comparison establishes a task regression.

## Conclusion

Experiment 040 is the first candidate in this sequence to satisfy all gates:
absolute selected mass, broad NLL/KL, strict paired C4 NLL/KL, no established
task regression, unchanged BPW, complete resident validation, exact packed
reload, compressed-model quality, and validated GGUF export. The useful design
lesson is to spend most optimization capacity on conditional shape and apply
only enough inequality pressure plus the smallest confidence fold needed to
cross the deployment mass floor.

Retained evidence:

- `evidence/040/experiment040-weight2-epoch1-fold-sweep-validation104-48x512-kl.json`
- `evidence/040/experiment040-weight2-epoch1-fold1p015-validation300-48x512-kl.json`
- `evidence/040/experiment040-conditional-epoch8-validation300-48x512-kl.json`
- `evidence/040/experiment040-weight2-epoch1-fold1p015-standard-quality.json`
- `evidence/040/experiment040-weight2-epoch1-fold1p015-tasklimit1000-quality.json`
- `evidence/040/experiment040-matched-c4-validation104-48x512.json`
- `evidence/040/experiment040-materialized-validation.json`
- `Results/040/040-low-pressure-correction-minimal-fold-gemma-3-1b-it-quality.json`
- `Results/040/040-low-pressure-correction-minimal-fold-gemma-3-1b-it-tasklimit1000-quality.json`
- `Results/040/gemma-3-1b-it-nanoquant.export-summary.json`
