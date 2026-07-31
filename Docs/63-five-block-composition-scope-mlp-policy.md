# Five-Block Composition-Scope MLP Policy

## Question

The two-block experiment proves that heterogeneous scale policies can
compose. The next question is whether independently selected refit and
representation choices remain optimal as more block errors interact.

This experiment expands the representative set to blocks 0, 4, 8, 12, 16,
20, and 24, confirms new block-local candidates, and tests a five-block
composition on evaluation inventories unused by fitting or block selection.

## Additional block screens

All screens use the same pinned model, equal-bit rank-970 free-word and
rank-1,344 mixed representations, 800 factorization iterations, scale
bounds, and fit/validation inventories as the preceding experiments.

### Blocks 4 and 8

Both blocks strongly prefer joint down input/output refitting:

| Block | Gate/up refit change | Joint change after gate/up | Combined 36-sequence joint change |
| ---: | ---: | ---: | ---: |
| 4 | -13.543% | -11.056% | **-12.869%** |
| 8 | -30.146% | -9.367% | **-10.480%** |

The disjoint confirmations pass for every reported stage. Joint also beats
input-only directly on the initial screen:

- block 4: -2.185%, interval [-0.005001, -0.000958];
- block 8: -3.291%, interval [-0.004404, -0.002041].

Representation choice differs:

- block 4 mixed versus free joint is tied across 36 sequences: +0.240%,
  interval [-0.006032, +0.006574];
- block 8 mixed is 6.825% worse than free joint, interval
  [+0.002143, +0.011885].

### Blocks 16 and 20

Block 16 accepts only the gate/up operator stage:

- combined operator gain: **-9.402%**, interval
  [-0.019770, -0.014906];
- combined mixed versus free operator: **-3.468%**, interval
  [-0.011172, -0.001009].

Downstream input and joint stages do not pass the initial block-16 screen.

Block 20 rejects the entire operator-scale path. Gate/up is neutral, while
down input and joint refits regress by 11.497% and 11.800% with confidence.

Block 24 remains rejected from the preceding experiment.

## Hybrid representation policy

The splice probe now accepts a second declared mapping for factor
representation. For example:

```text
downstream:     0:joint,4:joint,8:joint,12:input,16:operator
representation: 0:mixed,4:free,8:free,12:mixed,16:mixed
```

The representation policy chooses complete layer reconstructions from the
already constructed free or mixed downstream-policy arm. It requires
complete block coverage and rejects unavailable policies before evaluation.

The initial five-block hypothesis followed the independently confirmed
choices:

- block 0: mixed joint;
- block 4: free joint, choosing the cheaper tied representation;
- block 8: free joint;
- block 12: mixed input;
- block 16: mixed operator.

## Fresh composition screen

Scale fitting still uses sequences 48-51 and validation uses 52-55. The
five-block screen uses new sequences 80-91.

The independently selected hybrid does not win:

| Arm | KL |
| --- | ---: |
| Uniform mixed joint | **0.687147** |
| Uniform free joint | 0.694086 |
| All-free tailored downstream policy | 0.696251 |
| All-mixed tailored downstream policy | 0.701225 |
| Per-block representation/refit hybrid | 0.714222 |

Uniform mixed joint improves 14.825% over mixed gate/up-only, interval
[-0.143664, -0.096017]. The uniform joint result shows that a downstream
stage rejected or weakened in isolation can become beneficial once upstream
block errors change its input distribution.

The hybrid's point estimate is worse than both uniform representation arms.
It is rejected and uniform joint advances to confirmation.

## Fresh composition confirmation

The second inventory uses sequences 92-115. Uniform joint refitting passes
again:

| Representation | Gate/up-only KL | Joint KL | Change |
| --- | ---: | ---: | ---: |
| Free words | 0.774691 | **0.672491** | **-13.192%** |
| Mixed | 0.762607 | 0.692827 | **-9.150%** |

The mixed joint paired interval is [-0.082016, -0.055728]. On this larger
confirmation, free joint is 3.024% better than mixed joint with interval
[+0.001559, +0.040067] when expressed as mixed minus free.

## Combined 36-sequence evidence

| Comparison | Before KL | After KL | Change | Paired 95% delta interval |
| --- | ---: | ---: | ---: | ---: |
| **Free joint minus free gate/up** | 0.793451 | **0.679689** | **-14.338%** | **[-0.127518, -0.100340]** |
| Mixed joint minus mixed gate/up | 0.777322 | 0.690934 | -11.114% | [-0.100726, -0.072330] |
| Mixed joint minus free joint | 0.679689 | 0.690934 | +1.654% | [-0.005823, +0.028098] |
| Hybrid minus free joint | 0.679689 | 0.695851 | **+2.378%** | **[+0.001307, +0.031019]** |

The combined mixed/free comparison is inconclusive, but free has the lower
KL point estimate, lower factorized work, and wins the larger confirmation.
It is therefore the quality-first five-block candidate.

## Decision

Accept uniform joint MLP scale refitting across blocks 0, 4, 8, 12, and 16
as the current five-block quality candidate.

- Prefer rank-970 free-word factors at this composition scope.
- Retain mixed factors as tied rather than disproven; they remain valuable
  where greater rank is needed, but do not improve this five-block result.
- Reject the independently selected hybrid representation/refit policy.
- Select future scale policies at composition scope, not solely by isolated
  block behavior.
- Continue block expansion in bounded groups and reserve fresh sequence
  inventories for every composition decision.

The 14.34% confirmed reduction is obtained by changing existing factor scale
values only; it adds no bits or runtime operations to the rank-970 format.

## Evidence

- `evidence/m4/sign-word-codebook-probe/downstream-refit/block4-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block4-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block8-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block8-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block16-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block16-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl24-24.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/block20-mlp-mixed-k10-r1344-free256-800-gatewide-downinput50-fit48-val52-kl0-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-4-8-12-16-hybrid-policy-mixed-free-800-fit48-val52-kl80-12.json`
- `evidence/m4/sign-word-codebook-probe/downstream-refit/blocks0-4-8-12-16-hybrid-policy-mixed-free-800-fit48-val52-kl92-24.json`
