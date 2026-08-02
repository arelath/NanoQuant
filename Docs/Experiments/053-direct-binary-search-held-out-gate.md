# Experiment 053: Direct Binary Search Held-Out Gate

## Question

Can the bounded direct search from Experiment 052 be made more effective with
shared sign-pattern transfer and variable-depth moves, and do its static
objective gains survive a held-out functional test on a real Gemma projection?

This experiment follows the additional suggestions in
`Docs/ImprovementSuggestions/BinomialFactorizationOptimizations-PossibleAdditions.md`.
It remains an opt-in diagnostic. It does not change the resident compression
recipe or packed representation.

## Added search tiers

The direct search now includes three additional bounded mechanisms:

1. **Shared-Q codebook transfer.** Sign rows already present in a factor form a
   deduplicated codebook. A matrix multiply scores those patterns for every
   row against the exact scale-profiled objective. The codebook is capped and
   selected using frequency plus deterministic diversity.
2. **Variable-depth locked-bit chains.** Each row follows an uphill-capable
   sequence of distinct bit flips and commits only its best improving prefix.
   This crosses barriers that ordinary one-bit descent cannot cross without
   accepting a worse final state.
3. **Capped one-bit refinement.** One-bit candidates are ranked by their exact
   objective gain times row or column importance. Each pass may edit only a
   configured fraction and absolute maximum of vectors.

Every tier retains exact objective acceptance and outer-pass rollback. The
rank, signs stored, scale arrays, and physical bit cost are unchanged.

## Search-tier checks

The codebook mechanism has a deterministic transfer test: a better sign row
present elsewhere in the shared factor is copied to a row that benefits from
it. On the retained 32x32/rank-3 ladder it made no change to the sampled
Gaussian or real Gemma crop, but accepted five transfers on the represented
control. That control was already at a numerical floor: repeated scale fitting
around the initial tiers reduced squared error by 36.80% without an accepted
one-bit flip, and codebook transfer extended the displayed reduction to 44.81%.
Those percentages exercise the implementation but are not evidence of useful
natural-model headroom. The codebook stage took 0.09-0.17 seconds per case.

The variable-depth mechanism has a deterministic rank-6 test in which one-bit
search is at a local optimum and the best locked chain improves it with two bit
changes. In the 3x3 ladder it accepted ten chains on the Gaussian case, reducing
weighted error from 0.0361831 to 0.0361341 after the preceding tiers. It did
not accept a move on the difficult real crop. Experiment 052 showed that crop's
remaining headroom requires a coupled left/right move, so a stronger
single-factor neighborhood does not remove the identified barrier.

These checks establish that both mechanisms work, but not that natural ADMM
solutions commonly contain the corresponding opportunity.

## Held-out protocol

The functional screen uses the pinned `google/gemma-3-1b-it` revision and
block 12. It refactorizes all five projection groups with the ordinary
diagonal/Fisher ADMM baseline and changes only the fused QKV group in the
candidate. QKV remains rank 638; all five groups remain at exactly 26,822,832
bits (0.999472370 BPW).

The comparison uses:

- 2,048 fit-covariance tokens and a disjoint 2,048 held-out tokens;
- 12 fixed WikiText-2 sequences of 512 tokens for paired full-splice KL;
- four fixed samples for isolated block-output reconstruction;
- the same baseline factorization, covariance slice, functional slice, and
  bit/rank allocation for every arm.

Three direct-search arms were tested:

| Arm | Search budget | One-bit updates | Codebook transfers |
|---|---|---:|---:|
| One-bit + codebook | 2 outer, 8 one-bit, 2 codebook passes | 5,773 | 749 |
| One-bit only | 2 outer, 8 one-bit passes | 5,780 | 0 |
| Capped one-bit | 1 outer, 1 one-bit pass, top 5% of vectors | 135 | 0 |

## Static and held-out reconstruction results

All arms improve the exact factor objective and both covariance measurements,
including the held-out covariance slice:

| Arm | Factor-error reduction | QKV fit-covariance reduction | QKV held-covariance reduction | Aggregate held-covariance reduction | Search time |
|---|---:|---:|---:|---:|---:|
| One-bit + codebook | 0.3318% | 0.4228% | 0.2069% | 0.0273% | 0.154 s |
| One-bit only | 0.3322% | 0.4292% | 0.2196% | 0.0290% | 0.091 s |
| Capped one-bit | 0.0721% | 0.1248% | 0.2281% | 0.0301% | 0.068 s |

The aggregate reduction is small because four of five groups are deliberately
held tensor-identical. The capped arm generalizes at least as well on the local
held-out covariance metric despite making only 135 edits.

## Functional result

The local gains do not survive composition into the block or model:

| Arm | Full-splice KL change | Relative KL change | Paired 95% interval | Block-output NRMSE change |
|---|---:|---:|---:|---:|
| One-bit + codebook | +0.0009241 | **+0.7235% worse** | `[+0.0002037, +0.0017083]` | +0.5203% worse |
| One-bit only | +0.0007668 | **+0.6004% worse** | `[+0.0002197, +0.0013686]` | +0.4732% worse |
| Capped one-bit | +0.0002452 | **+0.1920% worse** | `[+0.0000069, +0.0004775]` | +0.2454% worse |

The baseline full-splice KL is identical in all three runs at 0.1277180 nats
per token. Every paired interval is entirely above zero, so all three
candidates are rejected. Capping edits reduces the damage by about three
quarters but does not change its direction.

## Interpretation

The failure is not codebook-specific. One-bit-only search already produces the
functional regression, and the much smaller capped arm has the same sign. Nor
is this a simple fit/held-out overfitting result: the held covariance metric
improves in every arm. The mismatch is between the separable projection-local
objective and the projection's role inside the nonlinear block and model.

Variable-depth and joint windows are stronger ways to optimize the same local
objective. Running them on this QKV splice after the three consistently harmful
results would increase search strength without addressing the measured
objective-placement mismatch. They are therefore not promoted to another
model run here.

## Decision

1. Keep codebook transfer, variable-depth chains, and capped one-bit search as
   bounded, opt-in diagnostic tools.
2. Keep all new tiers disabled by default. Do not add them to the resident
   compression recipe or claim a compressed-model quality gain.
3. Require held-out block-output and paired language-KL gates for any future
   direct sign-search placement. Static factor error and held covariance are
   insufficient promotion criteria.
4. If direct search is revisited, change the optimization signal or placement
   first—for example, use a block-output-aware residual/Jacobian proxy or apply
   the search jointly with functional tuning—rather than spending more compute
   on the same diagonal objective.
5. Preserve the exhaustive and planted controls: they remain useful for
   determining whether a future solver can cross a known combinatorial basin,
   independently of whether that solver should be deployed.

## Evidence

- `evidence/053/ladder32-rank3-codebook.json`
- `evidence/053/ladder3-variable-depth-cpu.json`
- `evidence/053/block12-qkv-direct-splice.json`
- `evidence/053/block12-qkv-onebit-splice.json`
- `evidence/053/block12-qkv-capped-onebit-splice.json`

The evidence directory is intentionally ignored. The reproducible probes,
tests, and this report are the durable record.
