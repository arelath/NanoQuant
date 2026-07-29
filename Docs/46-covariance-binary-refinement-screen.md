# Covariance-Aware Binary Refinement Screen

**Date:** 2026-07-29
**Status:** harness validated; pinned GPU measurement pending
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

Pending the pinned GPU run.
