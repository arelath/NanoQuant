# Nine-Block Even-Depth Composition

## Question

The five-block composition establishes a large zero-bit scale gain, but it
does not yet cover enough depth to support a model-quality claim. This
experiment screens the remaining even-depth representatives through block
24, confirms every passing local policy, and composes nine accepted blocks
on two fresh evaluation inventories.

## Additional confirmed blocks

All block-local runs retain the pinned model, equal-bit rank-970 free and
rank-1,344 mixed factors, 800 factorization iterations, and the established
fit, validation, and bootstrap protocol.

| Block | Accepted mixed scale policy | Combined 36-sequence change | Representation result |
| ---: | --- | ---: | --- |
| 2 | Gate/up operator only | **-8.409%** | Free point lower; tied |
| 6 | Gate/up plus joint down | **-2.989%** after gate/up | Mixed point lower; tied |
| 10 | Gate/up plus down input | **-9.390%** after gate/up | Mixed wins by 7.788% |
| 14 | Gate/up operator only | **-3.688%** | Free/mixed tied |

At block 10, gate/up itself also improves by 7.821%. Its final mixed input
arm has KL 0.100880 versus 0.109400 for free, with paired interval
[-0.013478, -0.003693].

Blocks 18 and 22 reject all scale policies:

- block 18 gate/up is neutral and joint refitting regresses by 5.541%;
- block 22 gate/up regresses by 5.166%, and downstream stages regress
  further.

Together with earlier results, the accepted even-depth set is:

```text
0:joint, 2:operator, 4:joint, 6:joint, 8:joint,
10:input, 12:input, 14:operator, 16:operator
```

Blocks 18, 20, 22, and 24 are excluded from this composition.

## Predeclared representation mapping

The composition screen declares both scale and factor choices before
evaluation:

```text
0:mixed, 2:free, 4:free, 6:mixed, 8:free,
10:mixed, 12:mixed, 14:free, 16:mixed
```

This mapping follows each block's confirmed or lower point-estimate
representation. Uniform free, uniform mixed, uniform input, uniform joint,
and all-free/all-mixed tailored policies remain comparison arms.

## Fresh composition screen

The first nine-block inventory uses sequences 116-127, disjoint from all
fit, validation, local-selection, local-confirmation, and earlier
composition windows.

| Arm | KL |
| --- | ---: |
| Free unrefitted | 1.697915 |
| Free joint | 1.357750 |
| Free tailored | 1.326058 |
| Mixed tailored | 1.298639 |
| Mixed joint | 1.297509 |
| **Predeclared hybrid** | **1.229251** |

The hybrid beats:

- free unrefitted by 27.602%;
- free joint by 9.464%;
- mixed joint by 5.261%;
- free tailored by 7.300%;
- mixed tailored by 5.343%.

Every paired interval excludes zero.

## Fresh composition confirmation

The exact mapping is rerun on sequences 128-151:

| Arm | KL |
| --- | ---: |
| Free unrefitted | 1.608442 |
| Free joint | 1.236710 |
| Free tailored | 1.223567 |
| Mixed tailored | 1.212502 |
| **Hybrid** | **1.202622** |
| Mixed joint | 1.198367 |

Hybrid remains below both tailored policies, although those direct
confirmation-window margins are inconclusive. Uniform mixed joint is lower
by 0.35% on this window. The combined evidence determines which policies
remain viable.

## Combined 36-sequence evidence

| Comparison | Before KL | Hybrid KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Free unrefitted | 1.638266 | **1.211498** | **-26.050%** | **[-0.466194, -0.386786]** |
| Mixed unrefitted | 1.542572 | **1.211498** | **-21.462%** | **[-0.364914, -0.298159]** |
| Free joint | 1.277057 | **1.211498** | **-5.134%** | **[-0.094069, -0.038160]** |
| Free tailored | 1.257731 | **1.211498** | **-3.676%** | **[-0.070635, -0.022141]** |
| Mixed tailored | 1.241215 | **1.211498** | **-2.394%** | **[-0.050954, -0.008670]** |
| Mixed joint | 1.231414 | 1.211498 | -1.617% | [-0.041438, +0.001192] |

The predeclared hybrid is the best combined point and beats every tailored
or free representation arm with confidence. Uniform mixed joint remains a
viable simpler alternative because their direct interval touches zero.

Uniform mixed joint independently improves 24.834% over the free unrefitted
baseline, with interval [-0.443687, -0.368659].

## Decision

Accept the nine-block hybrid as the current quality-first MLP composition
candidate, while retaining uniform mixed joint as the simpler tied option.

- The scale and representation mapping is fixed before each evaluation.
- The 26.05% improvement is measured on inventories never used for local
  selection or scale fitting.
- The result still covers only nine MLP blocks, not the full compressed
  model, so the project goal remains open.
- Repeated confirmation currently refactorizes identical matrices. Durable
  reconstruction caching is the next implementation priority before
  expanding across all remaining depths.

[Probe Reconstruction Cache](65-probe-reconstruction-cache.md) implements
that content-keyed reuse with strict model/calibration/setting identity,
atomic safetensors persistence, corruption rejection, and exact real-model
KL replay.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/block2-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block2-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block6-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block6-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block10-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block10-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block14-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block14-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block18-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block22-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-2-4-6-8-10-12-14-16-nine-policy-800-fit48-val52-kl116-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-2-4-6-8-10-12-14-16-nine-policy-800-fit48-val52-kl128-24.json`
