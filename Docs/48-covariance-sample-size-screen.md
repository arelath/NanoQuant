# Dense-Covariance Sample-Size Screen

**Date:** 2026-07-29
**Status:** completed; 8,192 fit rows selected
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

All three arms completed on 2026-07-29. Their functional-slice hash is
identical:

`sha256:3170679947482c658dfc542028a143d81f371d9c51b8e036050b21cd5e5ca07a`

| Fit rows | Held-out covariance reduction | Joint-KL reduction | Absolute joint-KL 95% interval | NLL change |
| ---: | ---: | ---: | ---: | ---: |
| 2,048 | 12.35% | 3.60% | `[-0.05271, -0.00367]` | −0.01339 |
| 4,096 | 17.87% | 9.83% | `[-0.09670, -0.05571]` | −0.03549 |
| **8,192** | **19.71%** | **9.99%** | **`[-0.11234, -0.04318]`** | **−0.04470** |

Retained artifacts:

| Fit rows | Artifact suffix | SHA-256 |
| ---: | --- | --- |
| 2,048 | `blocks-3-7-9-fit-2048.json` | `4d27c98fcc3f090f59cac4c0d677f01718ec277995337757ce80b254ccd8092c` |
| 4,096 | `blocks-3-7-9-fit-4096.json` | `83bf5731f20b6807541331475e8d33f03f9868a7b789bad72cb316a1f5a161d7` |
| 8,192 | `blocks-3-7-9-fit-8192.json` | `6c79fd6f45d69270b0cf6271e7483ccb0c30f4835cabf8d505daad8a7497c89b` |

Both larger sample counts pass both branches of the decision rule relative to
2,048 rows. At 8,192 rows the held-out gain improves by 7.37 absolute
percentage points and joint-KL gain improves by 6.39 points. Relative to 4,096
rows, 8,192 adds another 1.84 held-out points and a small 0.16 functional
point, so the tie-break selects 8,192.

The decreasing fit-objective reductions—39.63%, 31.76%, and 27.05%—are also
healthy. More rows make the fit metric harder to exploit while making its
solution generalize better. This is the opposite of a larger optimizer merely
overfitting a fixed sample.

### Verdict

Select 8,192 covariance fit rows, retain zero diagonal blend, and rerun the
complete 26-block gate with the same 20-sequence reservation. Production
integration remains unauthorized until that independent composition rerun
clears the previously missed 20% held-out threshold while retaining the
functional and NLL gains.

```powershell
$snapshot = 'C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752'
$blocks = (0..25) -join ','
.\.venv\Scripts\python.exe tools\probe_covariance_binary.py `
  --model "$snapshot\model.safetensors" `
  --snapshot $snapshot `
  --calibration-state evidence\m4\gemma-cce-fisher-state `
  --output evidence\m4\covariance-binary-probe\blocks-0-25-fit-8192-depth32.json `
  --blocks $blocks `
  --block-output-blocks 0,12,24 `
  --full-only `
  --fit-tokens 8192 `
  --held-out-tokens 2048 `
  --covariance-reserved-samples 20 `
  --left-flip-steps 32 `
  --right-flip-batches 16 `
  --local-files-only
```

### Independent 26-block confirmation

The selected 8,192-row setting passed the complete rerun:

`evidence/m4/covariance-binary-probe/blocks-0-25-fit-8192-depth32.json`

SHA-256:
`ef2160e67185f089912100f8218fdc53ab74f8f99d251a2222917b9655f129f6`.

It reduces aggregate held-out covariance error by 24.06%, joint-splice KL by
12.85%, and NLL by 0.455772 nats/token. The joint-KL interval is wholly
negative at `[-0.559022, -0.391311]`. All 104 refined groups, all 26 block
aggregates, and all three retained block-output checks improve. Bits remain
exactly 697,393,632, or 0.999472370 BPW.

The selected sample count therefore clears the previously missed support gate
without sacrificing the functional result. Promote explicit resident
integration and a complete compression experiment.
