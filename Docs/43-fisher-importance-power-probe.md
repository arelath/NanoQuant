# Fisher Importance Power-Exponent Probe

**Date:** 2026-07-29
**Status:** harness validated; GPU measurement pending Experiment 032
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

No exponent arm has run yet. Experiment 032 currently owns the CUDA device,
and overlapping a second reconstruction probe would invalidate runtime and
memory evidence.

## Pre-registered screen

The first screen will use:

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

Pending GPU availability after Experiment 032.
