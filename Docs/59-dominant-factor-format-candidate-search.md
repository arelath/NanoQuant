# Dominant-Factor Format Candidate Search

**Date:** 2026-07-30

**Status:** k10 plus two corrections remains the quality optimum; k11 is a
lower-compute Pareto point; banking, correction tiers, and source-free-U
alternatives rejected

## Question

[Mixed Dominant-Factor Projection Screen](58-mixed-dominant-factor-projection-screen.md)
showed that the selective mixed representation transfers to a jointly
evaluated gate/up pair. This follow-up searches for a better per-component
format before paying for a complete 26-block screen.

The search asks:

1. Can cheaper one-correction words buy enough extra rank?
2. Can a larger dictionary replace rank without losing quality?
3. Can implicit table banks specialize by channel region or component?
4. Can only the difficult component banks pay for corrections?
5. For tall gate/up matrices, is it better to leave source U free even though
   coding the smaller source V buys much less rank?

Every arm uses the pinned Gemma block-12 corrected-CCE Fisher objective,
0.6 shrinkage, production scale fitting, and the rank-970 free-word control.
All reported candidates are at or below the control's approximately 1-BPW
storage.

## Width and correction-count frontier

The accepted block-12 gate/up control for this search is:

- transposed factorization, so the source matrix's large left factor is coded;
- rank 1,344 with 256 free dominant-factor rows;
- k10 table and two corrections;
- 400-iteration weighted-RMSE changes of -1.773% for gate and -0.784% for up.

Representative alternatives are:

| Format | Rank / free rows | Gate change | Up change |
| --- | --- | ---: | ---: |
| k8, one correction | 1,632 / 320 | -0.292% | +0.551% |
| k9, one correction | 1,568 / 320 | -1.053% | -0.187% |
| k10, one correction | 1,504 / 320 | -1.411% | -0.506% |
| k11, one correction | 1,472 / 256 | -0.895% | +0.031% |
| k9, two corrections | 1,376 / 256 | -0.877% | +0.055% |
| **k10, two corrections** | **1,344 / 256** | **-1.773%** | **-0.784%** |
| k11, two corrections | 1,280 / 288 | -1.771% | -0.782% |
| k9, three corrections | 1,248 / 192 | -0.560% | not continued |

k11/two-correction is an extremely close reconstruction tie, but uses 1,280
components instead of 1,344. Its factorized work is 1.296x dense versus
1.361x, 4.8% lower on the affected projection. It is therefore a
latency/quality Pareto point, not a quality improvement.

Three corrections are decisively uneconomical in the current joint
assignment. One gate fit took 2,796 seconds versus 20–34 seconds for the
one/two-correction arms and still lost most of the reconstruction gain. That
branch was stopped before the up fit.

The result also answers whether much wider offline assignment search is the
missing lever. The prior 16-to-64 shortlist experiment roughly doubled
candidate time for only 0.02–0.08 percentage points. Increasing correction
combinatorics is substantially worse.

## Implicit word-position banks

A word-position bank is selected from the 32-channel word offset, so no bank
ID is stored per component. More tables cost only table bytes, but fitting the
extra tables requires reducing the free prefix.

| Banks | Rank / free rows | Gate change | Up change |
| ---: | --- | ---: | ---: |
| 1, accepted control | 1,344 / 256 | **-1.773%** | **-0.784%** |
| 2 | 1,344 / 224 | -1.130% | -0.139% |
| 4 | 1,344 / 192 | -0.475% | +0.521% |
| 8 | 1,344 / 160 | +0.149% | +1.195% |

Regional channel specialization is not worth displacing fully free anchors.
The monotonic degradation is strong enough to reject this bank axis.

## Implicit component banks

Component banking partitions coded factor rows. Each bank has its own smaller
table, and the row range supplies the bank ID. For example, two k9 tables
contain the same total number of entries as one k10 table, while every coded
word stores one fewer index bit.

| Tables | Local width | Rank / free rows | Gate change | Up change |
| ---: | ---: | --- | ---: | ---: |
| 2 | k9 | 1,408 / 224 | -1.449% | -0.505% |
| 4 | k8 | 1,440 / 256 | -1.254% | -0.351% |
| 8 | k7 | 1,472 / 288 | -1.083% | -0.215% |

Factor permutation gives the solver freedom to specialize component groups,
but local-table restrictions still cost more than the extra rank recovers.
All three are dominated by the single k10 table.

## Tiered corrections

The next arms use row banks but pay the 9-bit two-correction stream only on a
prefix of coded component banks. Uncorrected banks are much cheaper, allowing
rank up to 1,632.

| Corrected share | Rank / free rows | Gate change | Up change |
| ---: | --- | ---: | ---: |
| 25% | 1,632 / 320 | +0.588% | +1.472% |
| 50% | 1,536 / 288 | -0.585% | +0.311% |
| 75% | 1,408 / 288 | -0.908% | +0.032% |
| 100%, accepted control | 1,344 / 256 | **-1.773%** | **-0.784%** |

The breadth/precision curve is monotonic: every coded component needs the
repair stream more than the factorization needs additional components.
Correction tiers are rejected.

## Leaving source U free

The transposed gate/up representation codes the storage-dominant source U.
The structural counterfactual fits the original 6,912 by 1,152 orientation,
leaves source U fully free, and codes only V.

Because U dominates the factor bits, coding V can increase rank only from 970
to 992–1,024. Five allocations spanning k8 through k12, one/two corrections,
and free-V shares from zero to 52% all regress:

| Projection | Best measured arm | Weighted RMSE change |
| --- | --- | ---: |
| Gate | k10/two corrections, rank 992, 512 free V rows | +3.200% |
| Up | k10/two corrections, rank 992, 512 free V rows | +2.903% |

Other arms regress by 4.5–5.8%. Per-output mixture freedom is useful only when
the coded factor is large enough to fund substantial breadth. For these tall
matrices, coding the smaller source V cannot do that.

## Down-projection transfer

The surviving variants were also tested on block-12 `down_proj`:

| Format | 400-iteration change |
| --- | ---: |
| k10/two corrections, rank 1,344 | **-0.441%** |
| k11/two corrections, rank 1,280 | -0.451% |
| k10/one correction, rank 1,504 | -0.127% |
| two component banks, local k9 | -0.171% |
| two word-position banks | +0.241% |
| 50% corrected tier | +0.708% |
| 75% corrected tier | +0.382% |

The k11 point is slightly ahead before convergence, so both were repeated at
800 iterations:

| Format | Weighted RMSE change | Factor work / dense |
| --- | ---: | ---: |
| **k10, rank 1,344** | **-0.962%** | 1.361x |
| k11, rank 1,280 | -0.858% | 1.296x |

The accepted k10 representation regains the quality lead at convergence.
No new down-projection quality candidate advances.

## Analysis implementation

The analysis code now supports:

- odd index widths for unconstrained full tables;
- multiple implicit full-codebook banks selected by word position or
  component row;
- exact table-count accounting;
- a corrected component-bank prefix followed by uncorrected coded banks;
- exact decode and export of partial correction prefixes.

These are research-only capabilities. The packed mixed-V overlay remains
fixed to one k10 table and two corrections on every coded row. Rejected arms
must not be emitted through the runtime schema.

## Decision

The k10/two-correction mixed dominant-factor representation remains the
quality optimum across the measured format family.

- Keep k11/two-correction only as a possible 4.8%-lower-factor-work option
  when affected-layer latency matters more than about 0.1 percentage point of
  local reconstruction gain.
- Reject one correction, three corrections, word banks, component banks,
  partial correction tiers, and source-free-U gate/up fitting.
- Do not spend more GPU time on nearby index-width/free-row allocations.

The remaining promising search axis is not another component encoding. It is
operator-scope optimization: jointly refit gate and up against the
post-SiLU product error so their errors cancel deliberately, then optionally
refit down against the resulting student MLP activation. The block-12 joint
splice gain in the previous screen is direct evidence that this coupling is
real, while this format search shows that independent matrix reconstruction
has reached a local capacity optimum.

[Coupled MLP Operator-Scale Refit](60-coupled-mlp-operator-scale-refit.md)
tests that next axis. The mixed block-12 gate/up pair gains a confirmed
7.7-8.0% relative KL reduction at zero additional format bits, while the
same refit does not pass disjoint confirmation on free-word factors.
