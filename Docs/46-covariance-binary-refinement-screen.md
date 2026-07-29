# Covariance-Aware Binary Refinement Screen

**Date:** 2026-07-29
**Status:** all full-model gates passed with 8,192 covariance fit rows
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Question

The same-rank real-valued covariance bound in
[Document 44](44-covariance-headroom-probe.md) reduces held-out error energy
by 41.53%, while the input-only Hadamard attempt in
[Document 45](45-input-hadamard-covariance-screen.md) moves strongly in the
wrong direction. Can the existing runtime representation realize useful
covariance headroom when its scales and binary signs are optimized directly
under the dense activation metric?

The representation remains:

`diag(post) @ B_left @ diag(mid) @ B_right @ diag(pre)`.

Ranks, binary payloads, scale counts, and runtime algebra are unchanged. No
whitening matrix, rotation metadata, sparse side path, or new persisted field
is introduced.

## Solver

`tools/probe_covariance_binary.py` starts both arms from the same ordinary
diagonal-objective ADMM and scale fit. The candidate then performs:

1. two alternating dense-covariance scale passes;
2. up to eight exact left-sign coordinate steps, with at most one accepted
   bit per output row in each step;
3. up to eight right-sign batches of at most 128 proposed bits, accepted only
   when the exact joint quadratic objective change is negative;
4. two final dense-covariance scale passes with rollback.

The three scale subproblems are solved in closed form. Input covariance makes
the pre-scale solve dense rather than column-independent. The middle-scale
system is the Hadamard product of the output-weighted left Gram and
covariance-weighted right Gram. Post scales remain independent by output row.

For sign refinement, the harness calculates exact one-bit objective deltas.
Left rows are independent because output Fisher is diagonal. Right bits are
coupled by both the rank Gram and input covariance, so proposed batches use
the full pairwise quadratic correction and back off until the batch is a true
descent step.

This is a bounded warm-start optimizer, not yet the generalized ADMM solve
described in `NextQualityLevers.md` §13. A failure would measure how much
covariance headroom is accessible from the production diagonal solution; it
would not prove that a covariance-aware initialization or continuous ADMM
path cannot reach a different basin.

## Pre-registered protocol

The screen uses:

- complete blocks `{0, 12, 24}`;
- refined fused QKV, O, gate, and up groups;
- down projection reconstructed once and held tensor-identical between arms;
- raw corrected CCE output Fisher importance;
- 2,048 fit and 2,048 disjoint held-out covariance rows;
- 1% mean-diagonal covariance damping for optimization;
- target 1.0 BPW with identical ranks, factor seeds, scale costs, and 400 by 5
  baseline ADMM settings;
- 12 additional disjoint WikiText sequences of 512 tokens for joint-splice KL;
- isolated block-output error on four of those sequences;
- paired 10,000-resample sequence-bootstrap KL intervals.

The primary functional gate requires at least a 5% joint-KL reduction with
the paired 95% interval entirely below zero. The supporting covariance gate
requires at least a 10% aggregate held-out error-energy reduction. Both gates
must pass before productionizing any covariance-aware factorizer.

If the candidate misses the primary gate but captures at least 10% held-out
covariance error, the result still justifies a true generalized-ADMM
initialization screen. If it cannot improve the held-out covariance metric,
inspect the fit/held gap and accepted flip counts before deciding whether the
warm-start basin or the binary format is binding.

## Validation

Focused CPU tests prove:

- every left-bit delta equals direct dense-covariance recomputation;
- every right-bit delta equals direct dense-covariance recomputation;
- covariance scale fitting is monotone with rollback and recovers a known
  factorized target;
- accepted sign refinement strictly respects the measured objective;
- invalid shapes and settings are rejected.

The five new tests pass together with the nine covariance and Hadamard
regressions. Focused Ruff, bytecode compilation, and CLI help validation also
pass.

## Retained command

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools\probe_covariance_binary.py `
  --model "$snapshot\model.safetensors" `
  --snapshot $snapshot `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\covariance-binary-probe\blocks-0-12-24.json `
  --local-files-only
```

## Result

The pre-registered run and two bounded depth checks completed on 2026-07-29
without an overlapping NanoQuant worker. All arms use exactly 80,468,496
factor bits, or 0.999472370 BPW, with identical ranks and tensor-identical
down projections.

### Retained artifacts

| Left steps / right batches | Artifact | SHA-256 |
| ---: | --- | --- |
| 8 / 8 | `evidence/m4/covariance-binary-probe/blocks-0-12-24.json` | `cae84a589d6d6e3e6edf48e954f4b152cf4ae3075b2416c3877efd6e426caac1` |
| 32 / 16 | `evidence/m4/covariance-binary-probe/blocks-0-12-24-depth32.json` | `b752b77acbd80cd5fda5241e4a36a6d94c0da94e8c04cc357bc82101693d8576` |
| 128 / 32 | `evidence/m4/covariance-binary-probe/blocks-0-12-24-depth128.json` | `7bd2d5a4759da6e10f7ba099c0dea7ecd2fc98a94c6621e1b21e32563b47b33e` |

### Depth selection

| Left steps / right batches | Held-out covariance error reduction | Joint KL reduction | Absolute joint-KL 95% interval |
| ---: | ---: | ---: | ---: |
| 8 / 8 | 22.28% | 6.88% | `[-0.03837, -0.01702]` |
| **32 / 16** | **30.94%** | **10.47%** | **`[-0.06022, -0.02112]`** |
| 128 / 32 | 32.11% | 8.78% | `[-0.05149, -0.01774]` |

The pre-registered 8-step arm passes both gates. Extending to 32 steps improves
both metrics substantially. At 128 steps, the local covariance metric gains
only another 1.18 percentage points while joint KL gives back 1.69 percentage
points. This is the expected signature of over-optimizing the local proxy, so
32 steps is the promoted bounded setting rather than “run to convergence.”

### Selected 32-step result

| Aggregate metric | Plain diagonal | Covariance refined | Change |
| --- | ---: | ---: | ---: |
| Original-space normalized RMSE | 0.620157 | 0.665852 | +7.37% |
| Fit-covariance normalized RMSE | 0.140635 | 0.097462 | error energy −51.96% |
| Held-out covariance normalized RMSE | 0.150204 | 0.124825 | error energy **−30.94%** |
| Joint-splice KL | 0.398949 | 0.357165 | **−10.47%** |
| Joint-splice NLL | 4.058265 | 4.059764 | +0.001499 |

The original-space regression is not treated as a contradiction. This solver
is deliberately trading parameter-space fidelity for directions observed in
the activation covariance. The nearly neutral joint NLL and strongly improved
teacher KL show why an unweighted Frobenius gate would reject a functionally
better reconstruction.

All projection families improve on held-out covariance:

| Projection | Mean error-energy reduction | Range over blocks 0/12/24 |
| --- | ---: | ---: |
| O | **35.29%** | 28.07% to 47.11% |
| Up | 27.25% | 22.75% to 32.02% |
| Gate | 26.83% | 21.49% to 31.44% |
| Fused QKV | 25.46% | 21.93% to 29.25% |

Isolated block-output normalized RMSE improves by 10.58% at block 0, 16.08%
at block 12, and 5.57% at block 24. The isolated KL intervals for all three
blocks are also entirely below zero. This is broad evidence rather than one
late-block outlier.

The scale-only first pass reduces the aggregate regularized fit objective by
4.45%. Signs account for most of the accessible gain: after 32 left steps and
16 right batches, the fit objective is 49.43% lower, and the final scale pass
takes it to 50.09% lower. This identifies covariance-guided binary selection,
not merely a better scale solve, as the operative mechanism.

The incremental refinement took 2.53 seconds over the twelve refined groups
in this representative run, with a maximum allocated device footprint of
521,184,768 bytes. These are screen measurements rather than production
benchmarks, but they show that a direct bounded refinement is much cheaper
than a second 400-iteration ADMM pass.

### Verdict

Promote covariance-aware binary refinement with 32 left-sign steps and 16
bounded right-sign batches. It captures 74.5% of the 41.53% real-valued
held-out covariance headroom while preserving the exact format and bit count,
and it passes the pre-registered functional gate with margin.

The next gate is a complete 26-block splice screen using the same bounded
setting, still leaving down projection identical. That run must establish
whether local improvements compose across the whole model before this math is
routed into resident quantization. Down projection needs a low-rank,
block-diagonal, or sample-space covariance solve and remains a separate
follow-up rather than silently using a 6,912-wide dense pre-scale system.

## Pre-registered full-model composition follow-up

The generalized harness accepts any inventory of complete blocks. The
full-model follow-up uses all 26 blocks, retains the selected 32-left-step and
16-right-batch setting, evaluates only the complete-model splice, and limits
isolated output captures to blocks 0, 12, and 24. This avoids 26 redundant
single-block language-model evaluations while retaining early/middle/late
diagnostics.

The primary gate remains at least a 5% full-splice joint-KL reduction with the
paired 95% interval entirely below zero. Supporting gates are:

- at least 20% aggregate held-out covariance error-energy reduction;
- candidate NLL no more than 0.01 nats/token above the diagonal baseline on
  the exact same retained sequences;
- exact equality of ranks, factor bits, token hashes, and down reconstructions;
- no early/middle/late isolated block-output regression.

Passing authorizes integration behind an explicit resident option and a
complete compression experiment. Failing stops production work until the
composition failure is localized.

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
$blocks = (0..25) -join ','
.\.venv\Scripts\python.exe tools\probe_covariance_binary.py `
  --model "$snapshot\model.safetensors" `
  --snapshot $snapshot `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\covariance-binary-probe\blocks-0-25-depth32.json `
  --blocks $blocks `
  --block-output-blocks 0,12,24 `
  --full-only `
  --left-flip-steps 32 `
  --right-flip-batches 16 `
  --local-files-only
```

### Full-model result

The 26-block run completed in 13.2 minutes and retained:

`evidence/m4/covariance-binary-probe/blocks-0-25-depth32.json`

Its SHA-256 is
`b8f9832015e403aea67cfa3430663dd9182894b06da7d4fd8f08234ee4096823`.
Both arms use exactly 697,393,632 physical bits over 697,761,792 source
weights, or 0.999472370 BPW.

| Full-model metric | Plain diagonal | Covariance refined | Change |
| --- | ---: | ---: | ---: |
| Original-space normalized RMSE | 0.612809 | 0.657627 | +7.31% |
| Fit-covariance normalized RMSE | 0.189562 | 0.141932 | error energy −43.94% |
| Held-out covariance normalized RMSE | 0.193750 | 0.175330 | error energy **−18.11%** |
| Joint-splice KL | 4.086728 | 3.214561 | **−21.34%** |
| Joint-splice NLL | 7.069564 | 6.194533 | **−0.875030** |

The absolute joint-KL change is −0.872167 nats/token with paired 95% interval
`[-0.936826, -0.806091]`. This decisively passes the primary gate. NLL improves
by far more than the allowed 0.01 regression, and isolated block-output RMSE
again improves at blocks 0, 12, and 24 by 10.58%, 16.08%, and 5.57%.

The generic payload promotion field uses the harness's original 10% covariance
threshold and is therefore `true`. Against the stricter 20% threshold
pre-registered specifically for this full-model follow-up, the 18.11% result
is a 1.89-point supporting-gate miss. It is recorded as such rather than
post-hoc relabeled as a pass.

The miss is not a localized composition failure:

- all 104 refined block/groups improve held-out covariance error;
- all 26 block aggregates improve, ranging from 11.20% to 31.12%;
- mean group reductions are 30.78% for O, 20.42% for up, 19.61% for gate,
  and 17.63% for fused QKV;
- the three retained isolated block outputs and the full language-model
  metrics all improve.

Instead, the fit/held gap points to covariance-estimation noise. Across all
refined groups, the regularized fit objective falls 42.44%, while the disjoint
held-out objective falls 18.11%. The bounded refinement itself takes 22.49
seconds across 104 groups and peaks at 520,607,744 allocated device bytes;
the baseline ADMM work takes 730.36 seconds.

### Full-model verdict

The functional composition question passes strongly, but production
integration remains paused under the literal pre-registered rule because the
supporting covariance threshold missed. The failure has been localized to
fit-to-held generalization rather than a block or projection regression. The
next bounded experiment therefore shrinks only the off-diagonal fit
covariance on the three weakest blocks. Its diagonal, baseline reconstruction,
rank inventory, bits, and downstream protocol remain fixed. An independently
selected shrinkage must then rerun the 26-block screen before resident
integration.

### Final independent composition gate

The shrinkage screen rejected nonzero diagonal blending, while the
sample-size screen selected 8,192 fit rows. The independently reserved
20-sequence full-model rerun is:

`evidence/m4/covariance-binary-probe/blocks-0-25-fit-8192-depth32.json`

Its SHA-256 is
`ef2160e67185f089912100f8218fdc53ab74f8f99d251a2222917b9655f129f6`.

| Final full-model metric | Plain diagonal | Covariance refined | Change |
| --- | ---: | ---: | ---: |
| Original-space normalized RMSE | 0.612011 | 0.647000 | +5.72% |
| Fit-covariance normalized RMSE | 0.191669 | 0.158113 | error energy −31.95% |
| Held-out covariance normalized RMSE | 0.192925 | 0.168122 | error energy **−24.06%** |
| Joint-splice KL | 3.663641 | 3.192687 | **−12.85%** |
| Joint-splice NLL | 5.986675 | 5.530903 | **−0.455772** |

The absolute joint-KL delta is −0.470954 nats/token with paired 95% interval
`[-0.559022, -0.391311]`. Isolated block-output RMSE improves by 11.86%,
10.46%, and 3.91% at blocks 0, 12, and 24. All 104 refined groups and all 26
block aggregates improve held-out covariance error. The weakest block is
still positive at 19.47%; projection means are 35.94% for O, 24.77% for up,
23.77% for gate, and 21.75% for fused QKV.

Both arms retain exactly 697,393,632 physical factor bits and
0.999472370 BPW. The functional-slice hash is
`sha256:3170679947482c658dfc542028a143d81f371d9c51b8e036050b21cd5e5ca07a`.
The refinement takes 20.74 seconds over 104 groups, versus 684.23 seconds of
baseline ADMM work, and peaks at 520,601,088 allocated device bytes.

This rerun clears every pre-registered composition gate, including the
previously missed 20% held-out threshold. Covariance-aware binary refinement
is therefore promoted for resident integration behind an explicit option,
followed by a complete compression run and normal quality/export validation.
