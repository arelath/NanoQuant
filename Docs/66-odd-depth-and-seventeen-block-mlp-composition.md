# Odd-Depth and Seventeen-Block MLP Composition

## Question

The nine-block even-depth composition established a 26.05% held-out KL
improvement, but it left the intervening blocks untested. This experiment
screens every odd-depth MLP, confirms passing local policies, and then composes
the accepted odd and even candidates. It also tests whether an unrefitted
representation can coexist with refitted blocks without forcing a harmful
scale stage.

All runs use the pinned Gemma model, equal-bit rank-970 free and rank-1,344
mixed factors, 800 factorization iterations, and the established disjoint
fit, validation, screen, and confirmation protocol.

## Base policy support

Block 11 selects the mixed representation but rejects every scale refit. The
probe's per-block downstream policy previously required one of `operator`,
`output`, `input`, or `joint`, so it could not express this valid result in a
composition.

The policy now also accepts `base`. It selects the unrefitted free or mixed
reconstruction for that block, while other blocks can still use any refit
stage. The result schema is version 12.

## Odd-depth sweep

The accepted policies after a 12-sequence screen and independent 24-sequence
confirmation are:

| Block | Representation | Scale policy | Confirmed result |
| ---: | --- | --- | ---: |
| 3 | Free | Operator | **-11.91%** versus free base |
| 5 | Mixed | Operator | **-6.25%** versus mixed base |
| 7 | Free | Output | **-12.10%** versus free base |
| 9 | Mixed | Joint | **-11.01%** versus free base |
| 11 | Mixed | Base | **-12.91%** versus free base |
| 13 | Mixed | Operator | **-6.35%** versus mixed base |
| 17 | Mixed | Operator | **-5.29%** versus mixed base |
| 25 | Mixed | Output | **-32.30%** versus free base |

The block-25 downstream increment alone is 27.02%, and mixed beats free under
the output policy by 17.35%. Blocks 1, 15, 19, 21, and 23 reject all special
policies. Block 7's mixed-joint arm remains an unconfirmed alternative; the
composition keeps the predeclared and independently confirmed free-output
winner.

The fixed odd-depth mapping is:

```text
downstream:
3:operator, 5:operator, 7:output, 9:joint,
11:base, 13:operator, 17:operator, 25:output

representation:
3:free, 5:mixed, 7:free, 9:mixed,
11:mixed, 13:mixed, 17:mixed, 25:mixed
```

On the combined 36 fresh sequences 152-187, the odd hybrid has KL 1.007241.
It improves:

| Baseline | Baseline KL | Hybrid change | Paired 95% delta interval |
| --- | ---: | ---: | ---: |
| Free unrefitted | 1.144236 | **-11.973%** | **[-0.162865, -0.110953]** |
| Uniform mixed joint | 1.036999 | **-2.870%** | **[-0.049427, -0.010444]** |
| All-free tailored | 1.031795 | **-2.380%** | **[-0.043038, -0.005502]** |
| All-mixed tailored | 1.065160 | **-5.438%** | **[-0.078737, -0.037303]** |

## Seventeen-block composition

The accepted odd mapping was combined without modification with the
predeclared nine-block even mapping:

```text
blocks:
0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 25

downstream:
0:joint, 2:operator, 3:operator, 4:joint, 5:operator,
6:joint, 7:output, 8:joint, 9:joint, 10:input, 11:base,
12:input, 13:operator, 14:operator, 16:operator,
17:operator, 25:output

representation:
0:mixed, 2:free, 3:free, 4:free, 5:mixed, 6:mixed,
7:free, 8:free, 9:mixed, 10:mixed, 11:mixed, 12:mixed,
13:mixed, 14:free, 16:mixed, 17:mixed, 25:mixed
```

The screen uses sequences 188-199. Confirmation uses sequences 200-223.
Neither inventory overlaps factor fitting, local selection, local
confirmation, or earlier composition evaluation.

| Arm | Screen KL | Confirmation KL |
| --- | ---: | ---: |
| Free unrefitted | 3.242199 | 3.399402 |
| Mixed unrefitted | 2.976986 | 2.947789 |
| Uniform free joint | 2.194947 | 2.223726 |
| Uniform mixed joint | 2.282274 | 2.202235 |
| All-free tailored | 2.177157 | 2.213273 |
| All-mixed tailored | 2.317220 | 2.242625 |
| **Hybrid** | **2.121005** | **2.095910** |

The exact hybrid wins both inventories. On the 24-sequence confirmation it
beats all-free tailored by 5.30%, interval [-0.140639, -0.094472], and
all-mixed tailored by 6.54%, interval [-0.193371, -0.101600].

## Combined 36-sequence evidence

| Baseline | Baseline KL | Hybrid KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Free unrefitted | 3.347001 | **2.104275** | **-37.130%** | **[-1.300220, -1.185164]** |
| Mixed unrefitted | 2.957521 | **2.104275** | **-28.850%** | **[-0.905961, -0.804329]** |
| Uniform free joint | 2.214133 | **2.104275** | **-4.962%** | **[-0.132356, -0.087359]** |
| Uniform mixed joint | 2.228915 | **2.104275** | **-5.592%** | **[-0.157912, -0.091680]** |
| All-free tailored | 2.201234 | **2.104275** | **-4.405%** | **[-0.119581, -0.074164]** |
| All-mixed tailored | 2.267490 | **2.104275** | **-7.198%** | **[-0.198053, -0.129522]** |

Every comparison passes the paired bootstrap gate. The hybrid improves the
previous nine-block scope by covering eight additional independently selected
MLP blocks, while leaving nine rejected blocks dense.

## Decision

Accept the 17-block hybrid as the current quality-first MLP composition
candidate.

- Per-block representation and scale choices are fixed before each
  composition inventory.
- The 37.13% improvement is measured on 36 sequences unused by fitting or
  selection.
- `base` is necessary: forcing a scale stage on block 11 would discard a
  confirmed representation-only gain.
- This remains a partial-model splice result. It does not yet establish the
  quality or BPW of a complete compressed model.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/block{1,3,5,7,9,11,13,15,17,19,21,23,25}-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block{3,5,7,9,11,13,17,25}-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks3-5-7-9-11-13-17-25-eight-odd-policy-800-fit48-val52-kl152-12-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks3-5-7-9-11-13-17-25-eight-odd-policy-800-fit48-val52-kl164-24-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-2-3-4-5-6-7-8-9-10-11-12-13-14-16-17-25-seventeen-policy-800-fit48-val52-kl188-12-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-2-3-4-5-6-7-8-9-10-11-12-13-14-16-17-25-seventeen-policy-800-fit48-val52-kl200-24-cache.json`
