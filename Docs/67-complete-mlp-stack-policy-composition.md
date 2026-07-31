# Complete MLP-Stack Policy Composition

## Question

[Odd-Depth and Seventeen-Block MLP Composition](66-odd-depth-and-seventeen-block-mlp-composition.md)
leaves nine locally rejected blocks dense. This experiment asks whether the
accepted policy still wins when every `gate_proj`, `up_proj`, and `down_proj`
in all 26 Gemma blocks is replaced.

Rejected blocks use `base/free`: the equal-budget rank-970 free-word
reconstruction with no scale refit. This makes the splice a complete
MLP-stack compression test instead of a selected-block test. Attention,
embeddings, normalization, and the language-model head remain dense.

## Fixed policy

The 17 accepted blocks retain their independently selected representation and
scale policies. The nine rejected blocks use base/free:

```text
base/free blocks:
1, 15, 18, 19, 20, 21, 22, 23, 24
```

All 78 MLP matrices use approximately one-bit-per-weight factor budgets.
Factorization uses 800 outer iterations. Scale fitting uses sequences 48-55.
The composition screen uses sequences 224-235 and confirmation uses
sequences 236-259.

## Fresh screen and confirmation

| Arm | Screen KL | Confirmation KL |
| --- | ---: | ---: |
| Free unrefitted | 4.037956 | 4.100610 |
| Mixed unrefitted | 3.458744 | 3.523346 |
| Uniform free joint | 3.060105 | 2.886927 |
| Uniform mixed joint | 3.007654 | 2.939661 |
| All-free tailored | 3.022608 | 2.926290 |
| All-mixed tailored | **2.956747** | 2.939272 |
| **Hybrid** | 2.957275 | **2.824623** |

The screen leaves hybrid and all-mixed tailored exactly tied. The fixed hybrid
wins the larger confirmation:

- 3.474% below all-free tailored, interval [-0.147682, -0.053708];
- 3.901% below all-mixed tailored, interval [-0.165323, -0.060940].

## Combined 36-sequence evidence

| Baseline | Baseline KL | Hybrid KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| Free unrefitted | 4.079726 | **2.868840** | **-29.681%** | **[-1.297328, -1.127343]** |
| Mixed unrefitted | 3.501812 | **2.868840** | **-18.076%** | **[-0.696176, -0.566810]** |
| Uniform free joint | 2.944653 | **2.868840** | **-2.575%** | **[-0.120396, -0.033104]** |
| Uniform mixed joint | 2.962325 | **2.868840** | **-3.156%** | **[-0.138631, -0.047407]** |
| All-free tailored | 2.958396 | **2.868840** | **-3.027%** | **[-0.136983, -0.041717]** |
| All-mixed tailored | 2.945097 | **2.868840** | **-2.589%** | **[-0.122379, -0.028003]** |

Every comparison passes the paired bootstrap gate.

## Interpretation

Local rejection does not mean a block can stay dense in a complete
compression candidate. Supplying a conservative base/free reconstruction for
those blocks preserves the composition gain. The mixed representation on
locally rejected blocks is competitive on the screen, but the predeclared
hybrid wins confirmation and combined evidence.

The improvement from 17 selected blocks to all 26 MLP blocks is not directly
comparable as an absolute KL reduction because the baseline and number of
replaced matrices change. The important result is that the policy advantage
survives complete MLP replacement.

## Decision

Accept the hybrid as the quality-first policy for a complete compressed MLP
stack within the untuned sign-word reconstruction family.

- It covers all 78 MLP matrices across all 26 blocks.
- Its representation and scale choices were fixed before both inventories.
- It improves 29.68% over the equal-budget free-word MLP stack.
- It remains a partial-model result because all attention projections are
  still dense. Model-level BPW and perplexity claims require integrating the
  policy into a complete compression run.

The subsequent
[Complete MLP Frozen-Model Transfer Gate](68-complete-mlp-frozen-transfer-gate.md)
rejects direct installation: the untuned logical overlay regresses retained
Experiment 022 pre-KD perplexity from 273.87 to 1904.75. The policy remains
useful format and initialization evidence, not a model-quality candidate
until it receives resident tuning.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-25-complete-mlp-policy-800-fit48-val52-kl224-12-cache.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-25-complete-mlp-policy-800-fit48-val52-kl236-24-cache.json`
