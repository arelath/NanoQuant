# Fisher Importance Power-Exponent Probe

**Date:** 2026-07-29
**Status:** completed; no exponent promoted
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Question

Raw diagonal Fisher importance decisively beat the recipe's linear 0.6
shrinkage in the held-out static gate documented in
[Document 42](42-fisher-importance-shrinkage-probe.md). That result does not
fully answer whether a nonlinear transformation can retain Fisher's channel
ordering while reducing only its dynamic range.

The next controlled family is:

`I_alpha = mean(I) * I^alpha / mean(I^alpha)`.

The mean-preserving normalization matters. It prevents the fixed ADMM
regularization from being confounded by an arbitrary change in objective
scale. The endpoints have explicit meanings:

- `alpha=1` is an exact clone of raw Fisher and is the baseline;
- `alpha=0` is uniform importance at the original vector mean;
- `0 < alpha < 1` preserves ordering while compressing the dynamic range.

This is a fit-time-only change. Like linear shrinkage, it adds no runtime
storage or arithmetic.

## Harness

`tools/probe_importance_power.py` implements the analysis gate. It reuses the
same corrected CCE Fisher state, fused-QKV topology, fixed physical budget,
rank calculation, scale representation, ADMM settings, scale fitting, and
held-out WikiText evaluator used by the shrinkage probe.

The implementation has CPU unit coverage for:

- exact `alpha=1` identity without aliasing the input tensor;
- mean preservation for an interior exponent;
- the uniform `alpha=0` endpoint;
- rejection of invalid exponents and importance values;
- stable parsing and complete fused-QKV block topology.

The focused validation is:

```text
7 passed
Ruff: clean
mypy: clean
```

The pinned snapshot is a single safetensors file and exposes the expected
`model.layers.*` tensor names. The queued representative command is:

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools\probe_importance_power.py `
  --model "$snapshot\model.safetensors" `
  --snapshot $snapshot `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\importance-power-probe\blocks-0-12-24.json `
  --local-files-only
```

The measurement ran after Experiment 032 released the CUDA device. No
overlapping NanoQuant worker was active, and the probe acquired the
cross-process CUDA lease before loading model tensors.

## Pre-registered screen

The pre-registered screen used:

- exponents `{0.5, 0.75, 1.0}`;
- complete projection inventories for blocks `{0, 12, 24}`;
- fused Q/K/V plus separate O, gate, up, and down groups;
- target 1.0 BPW with identical ranks and seeds across arms;
- corrected raw CCE Fisher state from
  `evidence/m4/gemma-cce-fisher-state`;
- 400 outer and 5 inner ADMM iterations, cubic penalty schedule,
  regularization 0.03, and two scale-fit passes;
- 12 held-out WikiText sequences of 512 tokens;
- isolated block-output error on four held-out sequences;
- paired 10,000-resample sequence-bootstrap KL intervals.

The primary gate is joint three-block splice KL against the `alpha=1` raw
endpoint. An interior exponent is promoted only if its candidate-minus-raw
95% interval is entirely below zero at unchanged physical BPW. Isolated
block-output error is supporting evidence, not a substitute for the joint KL
gate.

If an interior exponent passes, it must next beat raw Fisher in a complete
26-block static confirmation before any resident compression recipe changes.
If neither arm passes, power tempering is recorded as a failure and raw Fisher
remains the selected importance transform.

## Result

The three-arm probe completed on 2026-07-29. The retained JSON artifact is:

`evidence/m4/importance-power-probe/blocks-0-12-24.json`

Its SHA-256 is
`f38b05a66a319ee028e076b7cefe4f4b6f6f0a72f1ef237f4327a711f1fe440b`.
All arms used exactly 80,468,496 physical bits over 80,510,976 source
elements, or 0.999472370 BPW. They used identical ranks, seeds, block
topology, and held-out tokens.

### Primary joint-splice gate

| Exponent | Joint KL, nats/token | Delta vs raw | Relative delta | Paired 95% interval | Pass |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1.0 raw baseline | 0.370512 | — | — | — | baseline |
| 0.75 | 0.362486 | -0.008027 | -2.17% | [-0.035181, +0.014686] | **no** |
| 0.5 | 0.429138 | +0.058625 | +15.82% | [+0.032420, +0.082241] | **no; worse** |

`alpha=0.5` is decisively harmful: its entire interval is above zero. The
`alpha=0.75` point estimate is modestly better than raw Fisher, but its
interval crosses zero. It therefore fails the pre-registered requirement
that the complete interval be below zero.

The isolated block KL comparisons agree with that decision:

| Exponent | Block 0 delta vs raw | Block 12 delta vs raw | Block 24 delta vs raw |
| ---: | ---: | ---: | ---: |
| 0.75 | -0.84%, interval crosses zero | -1.78%, interval crosses zero | -0.82%, interval crosses zero |
| 0.5 | **+30.34%, confidently worse** | +5.18%, interval crosses zero | +0.14%, interval crosses zero |

All three `alpha=0.75` block point estimates favor tempering, but none is
individually significant and the pre-registered joint gate remains the
promotion criterion. Running a complete 26-block confirmation after this
failed screen would weaken the protocol by selecting on noise.

### Supporting reconstruction evidence

| Exponent | Original-weight normalized RMSE | Isolated block-output RMSE delta vs raw: block 0 / 12 / 24 |
| ---: | ---: | --- |
| 1.0 | 0.619068 | baseline |
| 0.75 | 0.591424 | +2.78% / -4.50% / +0.28% |
| 0.5 | 0.569630 | +22.56% / -1.95% / +4.34% |

Tempering monotonically improves unweighted original-weight RMSE while
`alpha=0.5` decisively worsens functional KL. This independently reinforces
the Experiment 032 conclusion that plain Frobenius reconstruction error is
not a valid promotion metric. The mean-preserving transform did remove the
intended dynamic range, but preserving more unweighted mass did not preserve
model behavior.

### Verdict

Reject power tempering at `alpha=0.5` and do not promote `alpha=0.75`.
Within the nonlinear raw-Fisher family, `alpha=1` remains the statistically
supported endpoint. This does **not** promote raw Fisher into the compression
recipe: Experiment 032's complete retained evaluation already rejected raw
Fisher against Experiment 022's 0.6 linear shrinkage. The production baseline
therefore remains Experiment 022, and the next independent direction is the
pre-registered covariance-headroom probe.
