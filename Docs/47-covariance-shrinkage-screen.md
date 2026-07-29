# Dense-Covariance Shrinkage Screen

**Date:** 2026-07-29
**Status:** pre-registered; GPU measurement pending
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Motivation

The full 26-block covariance-aware binary refinement run improves every
refined group and reduces joint KL by 21.34%, but its 18.11% aggregate
held-out covariance reduction narrowly misses the pre-registered 20%
supporting threshold. The regularized fit objective falls 42.44%, identifying
fit-covariance estimation noise as the remaining gap.

The weakest block aggregates are blocks 7, 3, and 4 at 11.20%, 12.35%, and
12.53% held-out reduction. This screen uses blocks 3, 7, and 9; block 9 is the
next low region at 15.18% and avoids selecting three adjacent blocks that may
share the same local covariance regime.

## Candidate

For fit covariance `C`, test:

`C_alpha = (1 - alpha) C + alpha diag(C)`

at `alpha ∈ {0.25, 0.50, 0.75}` against the retained `alpha = 0` result.
The existing 1% mean-diagonal damping remains fixed.

Diagonal blending leaves `diag(C)` exactly unchanged. Consequently every arm
starts from the same diagonal-objective ADMM reconstruction with the same
seed, ranks, scale counts, and bits. Only the candidate's off-diagonal
covariance refinement metric changes.

## Protocol and decision

- complete blocks `{3, 7, 9}`;
- 32 left-sign steps and 16 right-sign batches;
- 2,048 fit plus 2,048 disjoint held-out covariance rows;
- the exact retained 12 functional sequences and four block-output sequences;
- fused QKV, O, gate, and up refined; down held tensor-identical;
- exact 0.999472370 BPW factor payload.

Select a nonzero blend only if it improves the aggregate held-out covariance
reduction by at least two absolute percentage points over `alpha = 0` and
does not weaken joint-KL gain by more than one percentage point. Ties choose
the larger held-out reduction, then the smaller blend. This is a selection
screen; the chosen value must pass the already pre-registered 26-block gates
on a rerun before production integration.

The retained commands differ only in output and blend:

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
foreach ($blend in 0, 0.25, 0.5, 0.75) {
  .\.venv\Scripts\python.exe tools\probe_covariance_binary.py `
    --model "$snapshot\model.safetensors" `
    --snapshot $snapshot `
    --calibration-state evidence\m4\gemma-cce-fisher-state `
    --output "evidence\m4\covariance-binary-probe\blocks-3-7-9-blend-$blend.json" `
    --blocks 3,7,9 `
    --left-flip-steps 32 `
    --right-flip-batches 16 `
    --covariance-diagonal-blend $blend `
    --local-files-only
}
```

## Result

Pending.
