# Dense-Covariance Sample-Size Screen

**Date:** 2026-07-29
**Status:** pre-registered; GPU measurement pending
**Model:** pinned `google/gemma-3-1b-it` revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752`

## Motivation

The complete 26-block covariance-aware binary refinement improves joint KL by
21.34% and improves every refined group, but fit covariance error falls 42.44%
while held-out error falls 18.11%. Off-diagonal shrinkage on weak blocks
improves the held proxy only by sacrificing too much functional gain. This
screen tests the direct alternative: estimate the dense covariance from more
activation rows without weakening its correlation structure.

## Protocol

Compare fit-row counts `{2,048, 4,096, 8,192}` with:

- 2,048 disjoint held-out rows after each fit window;
- complete blocks `{3, 7, 9}`, selected before the shrinkage screen;
- zero diagonal blend and fixed 1% mean-diagonal damping;
- 32 left-sign steps and 16 right-sign batches;
- identical target BPW, rank formula, factor seed, and scale counts;
- down projection held tensor-identical within each matched arm;
- the same functional sequences for every fit-row count.

The last condition is enforced by reserving the first 20 WikiText sequences
for covariance data in every arm. Each arm consumes only the prefix needed for
its fit plus held-out rows; functional evaluation always begins at sequence
20. Thus functional hashes are identical even though the covariance-fit
prefix grows.

The diagonal baseline is matched within each arm because its estimated
diagonal changes with sample count. Selection compares each covariance
candidate with its own same-sample diagonal baseline.

## Decision rule

Select a larger sample count if either:

1. held-out covariance reduction improves by at least two absolute percentage
   points over 2,048 rows while joint-KL gain weakens by no more than one
   percentage point; or
2. joint-KL gain improves by at least one percentage point while held-out
   covariance reduction does not regress.

Ties prefer functional gain, then fewer rows. A selected count must rerun the
complete 26-block gate with the same 20-sequence reservation before resident
integration.

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
foreach ($fitRows in 2048, 4096, 8192) {
  .\.venv\Scripts\python.exe tools\probe_covariance_binary.py `
    --model "$snapshot\model.safetensors" `
    --snapshot $snapshot `
    --calibration-state evidence\m4\gemma-cce-fisher-state `
    --output "evidence\m4\covariance-binary-probe\blocks-3-7-9-fit-$fitRows.json" `
    --blocks 3,7,9 `
    --fit-tokens $fitRows `
    --held-out-tokens 2048 `
    --covariance-reserved-samples 20 `
    --left-flip-steps 32 `
    --right-flip-batches 16 `
    --local-files-only
}
```

## Result

Pending.
