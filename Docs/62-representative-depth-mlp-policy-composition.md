# Representative-Depth MLP Policy Composition

## Question

Block 12 confirms two zero-bit scale stages for the mixed MLP
representation:

1. coupled gate/up output-scale refitting;
2. student-activation-aware down input-scale refitting.

This experiment tests whether those stages transfer across depth and whether
individually selected block policies compose on evaluation sequences that
were not used for either block selection.

## Protocol

The block-local representation and fitting settings are unchanged:

- pinned `google/gemma-3-1b-it`;
- all three MLP projections at rank 1,344 with k10, two corrections, and
  256 free dominant-factor rows;
- rank-970 free-word control;
- 800 factorization iterations;
- gate/up multiplier bounds 0.25-2 and 0.1-8;
- down input/output multiplier bounds 0.25-4;
- 50 down input iterations;
- operator fit on sequences 48 through 51;
- operator validation on sequences 52 through 55.

Representative block selection uses KL sequences 0 through 11 and
confirmation sequences 24 through 47. The selected multi-block policy is
then evaluated on two additional disjoint inventories:

- composition screen: sequences 12 through 23;
- composition confirmation: sequences 56 through 79.

No composition-evaluation sequence participates in scale fitting,
validation, or per-block policy selection.

## Representative-depth transfer

### Block 0

At block 0, the down input-scale-only stage is inconclusive. The joint
input/output stage is instead large and repeatable:

| Window | Gate/up KL | Joint KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Selection 0-11 | 0.235495 | 0.207853 | **-11.738%** | **[-0.039267, -0.017948]** |
| Confirmation 24-47 | 0.305904 | 0.277458 | **-9.299%** | **[-0.037324, -0.019879]** |
| Combined 36 | 0.282434 | 0.254256 | **-9.977%** | **[-0.035100, -0.021508]** |

The selected block-0 policy is therefore joint down input/output scale
refitting.

### Block 12

Block 12 retains the simpler input-only policy from the previous experiment:

- disjoint change: -2.254%, interval [-0.003185, -0.000617];
- combined change: -2.412%, interval [-0.002971, -0.000897];
- adding output scales does not beat input-only directly.

### Block 24

Block 24 is a hard rejection:

| Down policy | Change from gate/up arm | Paired 95% delta interval |
| --- | ---: | ---: |
| Input only | **+3.568%** | **[+0.000682, +0.003819]** |
| Input plus output | **+4.495%** | **[+0.001010, +0.004666]** |

This happens even though held-out block-output NRMSE improves by 3.856%
for input-only and 5.782% for joint fitting. Block-output reconstruction is
therefore not a safe cross-depth policy selector.

## Explicit per-block policy

The splice probe now supports a declared mapping such as:

```text
0:joint,12:input
```

It builds one reconstruction set by selecting the requested downstream arm
for every layer in each block. Both free-word and mixed representations use
the identical block policy. Missing arms, incomplete block coverage, and
unsupported choices are rejected before factorization.

This allows exact evaluation of heterogeneous policies without treating a
uniform downstream rule as representative.

## Multi-block composition

The selected block-0 joint plus block-12 input policy passes both new
inventories:

| Window | Gate/up-only mixed KL | Policy mixed KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| 12-23 | 0.377709 | 0.351262 | **-7.002%** | **[-0.040458, -0.012889]** |
| 56-79 | 0.297075 | 0.262802 | **-11.537%** | **[-0.041636, -0.026700]** |
| Combined 36 | 0.323953 | 0.292289 | **-9.774%** | **[-0.038499, -0.024625]** |

On the combined composition inventories, the mixed policy also beats the
identically refitted free-word policy:

| Free-word policy KL | Mixed policy KL | Change | Paired 95% delta interval |
| ---: | ---: | ---: | ---: |
| 0.323101 | **0.292289** | **-9.536%** | **[-0.043229, -0.018396]** |

The block-local improvements therefore compose rather than cancel. This is
the first confirmed heterogeneous multi-block scale policy.

## Decision

Accept the following research policy:

- block 0: gate/up refit plus joint down input/output refit;
- block 12: gate/up refit plus down input-only refit;
- block 24: no operator-scale refit.

All accepted stages change only scale values already present in the equal-bit
factor representation.

Do not infer the remaining blocks from operator validation NRMSE. The next
search must use a statistically gated per-block KL screen, confirm passing
arms on an independent inventory, and reserve fresh sequences for each
multi-block composition test.

[Five-Block Composition-Scope MLP Policy](63-five-block-composition-scope-mlp-policy.md)
expands that search through blocks 4, 8, 16, and 20. It finds that
composition-scope uniform joint refitting beats the independently selected
hybrid and confirms a 14.34% KL reduction for the five-block free-word arm.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/block0-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block0-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block24-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-12-policy-joint-input-mixed-k10-r1344-free256-800-fit48-val52-kl12-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-12-policy-joint-input-mixed-k10-r1344-free256-800-fit48-val52-kl56-24.json`
