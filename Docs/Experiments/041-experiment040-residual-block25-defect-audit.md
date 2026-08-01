# Experiment 041: Residual Block-25 Defect Audit after Experiment 040

## Status

Complete. The refit is rejected because it is significantly harmful on both
the screen and untouched confirmation. The Experiment 035 block-25-class
pathology is absent from the accepted Experiment 040 endpoint.

## Why this is next

The recommendations in
[72-experiment-035-036-review.md](../72-experiment-035-036-review.md) have now
split into resolved and unresolved parts:

1. The two retained cross-artifact tests are complete in Document 73.
2. The block-25 anomaly was traced to conditional top-64 KD's probability-mass
   blind spot in Documents 74 and 75.
3. The transplanted Experiment 036 seed, including its winsorized variant, is
   rejected and remains paused.
4. A pinned C4 NLL/KL gate is now mandatory and passed by Experiment 040.
5. Experiment 040 found a better objective policy: a short, one-sided
   mass-floor correction plus a minimal final-norm fold.

The winning correction is still analysis-only. Before productionizing it, the
review's central refit question must be answered on the accepted endpoint: did
the correction remove the block-25-class separable defect, or merely reduce
the same symptom? A large residual refit gain would mean the objective remains
incomplete and a per-campaign post-KD refit stage is still required. A neutral
or harmful refit would support productionizing the correction without that
stage.

## Frozen source

- run:
  `evidence/040/040-low-pressure-weight2-epoch1-fold1p015-gemma-3-1b-it`;
- active global tuning:
  `sha256-8f1d413a7ed4ebe8fefacb9b9326b4c201dec4e6f81d6fbcdf12ae52ce8eb914`;
- frozen identity: config
  `sha256:08d2e59056ddc6c4878d847b0a8802fd2b3b194bbc7b6567052a83cfd96def0b`,
  model
  `sha256:32d5b5d041e98027bc7415107bc79b580f9cce407535b4e30134e8f8aed3b130`,
  plan
  `sha256-2eaebd95597d7603267ba7f21a12aa777648d370f1111f2786b1a4e75b5563cc`;
- effective BPW: 1.024494712.

The active global tuning and folded final RMSNorm are part of the baseline.
The experiment changes only the existing block-25 MLP factor components.

## Fixed fitting policy

Fit one teacher-context `joint` refit for block 25's gate, up, and down
projections using the existing scale-refit implementation and its established
bounds. The dense target is then recovered into the current campaign's own
factorized `scale_pre`, `scale_post`, outlier, and patch terms using the
covariant equal-size encoding. No new tensor or represented byte is allowed.

The WikiText-2 test windows are fixed at 512 tokens:

| Role | Offset | Samples | Token hash |
| --- | ---: | ---: | --- |
| fit | 380 | 4 | `sha256:c21daa95a645876c986334db00f173c9319edbcc772312d56f7b858f9fb0eac2` |
| local validation | 384 | 4 | `sha256:a8f0d4b7ec1dd764e8768a4ab3999a59d63edbb21d829aed5ae5e63f39d1322d` |
| composed screen | 460 | 24 | `sha256:4489afec2135f92ba7d12c1a66ecd2ff7a25d578f76f640ebba0f933f16523da` |
| untouched confirmation | 484 | 48 | `sha256:55b6a116151c5968e4bb002836ee0ec7f77aece95e950db8a060bd22cfb18e07` |

The pinned dataset fingerprint is `a29ea8a573703a32`; BOS token 2 is prepended
to every independent window. The confirmation is not used to choose axes,
bounds, iterations, damping, or any other fit parameter.

## Gates

1. Dense fitting must complete with finite metrics and the exact source
   identity and active global-tuning reference.
2. Factor-compatible recovery must have zero payload-byte delta, exact tensor
   inventory validation, and agree with the dense refit closely enough that
   the sign of the screen NLL result cannot change.
3. The 48x512 confirmation reports causal NLL, full-vocabulary teacher KL,
   top-64-plus-tail KL, selected mass, and paired 10,000-resample intervals.
4. A residual **block-25-class defect** exists only if factor-compatible NLL
   and full KL each improve by at least 0.02 nats and both paired upper bounds
   are below zero on confirmation. This threshold is over 20 times smaller
   than the Experiment 035 block-25 effect while remaining above the review's
   marginal-power floor.
5. Any smaller, uncertain, or harmful result rejects a production block-25
   refit for this endpoint. It does not prove every possible refit is useless;
   it proves the large pathology motivating the stage is absent.

No quality benchmark or export is needed for a rejected refit because the
unchanged Experiment 040 baseline already completed both. If the refit passes
the residual-defect gate, it must proceed through an independent C4 gate,
packed quality, and complete export before it can alter the final-run design.

## Result

Fresh validation first rechecked all 708 transitive resident artifacts and 26
blocks at effective BPW 1.024494712. The dense teacher-context fit did improve
block-local validation normalized RMSE by 4.13%, so the procedure found a real
local direction. Its composed behavior moved the wrong way:

| 24x512 screen | Baseline | Dense block 25 | Delta |
| --- | ---: | ---: | ---: |
| NLL | 4.05519 | 4.14889 | **+0.09370** |
| Full KL | 1.26552 | 1.28607 | **+0.02055** |

The full-KL paired 95% interval was entirely harmful at
[+0.01422, +0.02750]. This reproduces the review's warning that a local output
fit is not evidence of a global model improvement.

The dense target was recovered into nine existing factor components. The
replacement and replaced payloads are both 119,808 bytes, so payload-byte
delta is exactly zero. On the same screen, factor-compatible NLL was 4.17354
versus dense 4.17365; factor-minus-dense delta was -0.00011 with interval
[-0.00058, +0.00037]. Encoding therefore preserves the decision.

On the untouched confirmation:

| 48x512 confirmation | Baseline | Factor block 25 | Delta | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| NLL | 4.28886 | 4.40065 | **+0.11178** | [+0.10407, +0.11978] |
| Full KL | 1.22652 | 1.25426 | **+0.02774** | [+0.02304, +0.03241] |
| Top-64 + tail KL | 1.15761 | 1.17925 | +0.02164 | - |
| Student top-64 mass | 0.75960 | 0.81528 | +0.05568 | - |

The refit increases confidence mass but spends substantially more conditional
shape and causal likelihood than it gains in tail calibration. Experiment
040's one-sided correction plus minimal final-norm fold already occupies the
useful tradeoff; block 25 is no longer a beneficial nonlinear compensator.

## Decision

The predeclared residual-defect gate fails by a wide margin. Do not
productionize a post-KD block-25 refit for this endpoint and do not resume
Experiment 036. The surviving final-run direction is to productionize the
one-sided correction itself, prove interrupted/resumed equivalence on the
retained state, and only then spend a fresh factorization campaign.

Retained evidence:

- `evidence/041-source-validation.json`
- `evidence/041/experiment041-block25-teacher-context-fit380-val384-screen460-24x512.json`
- `evidence/041/experiment041-block25-factor-compatible-screen460-24x512.json`
- `evidence/041/experiment041-block25-factor-confirmation484-48x512-kl.json`
- `evidence/041/experiment041-block25-factor-confirmation484-48x512-tail-mass.json`
