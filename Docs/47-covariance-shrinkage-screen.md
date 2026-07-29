# Dense-Covariance Shrinkage Screen

**Date:** 2026-07-29
**Status:** completed; nonzero shrinkage rejected
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

All four arms completed on 2026-07-29 with identical ranks, bits, diagonal
baselines, token hashes, and down reconstructions.

| Diagonal blend | Held-out covariance reduction | Joint-KL reduction | NLL change | Block-output changes (3 / 7 / 9) |
| ---: | ---: | ---: | ---: | ---: |
| **0.00** | 12.35% | **21.37%** | −0.01670 | −5.64% / −7.40% / −34.82% |
| 0.25 | **14.89%** | 17.52% | +0.00349 | −5.29% / −5.16% / −29.95% |
| 0.50 | **15.23%** | 16.61% | −0.00080 | −2.23% / −1.25% / −27.42% |
| 0.75 | 11.60% | 11.64% | +0.01718 | +1.02% / +0.67% / −21.89% |

Retained artifacts and SHA-256 values:

| Blend | Artifact suffix | SHA-256 |
| ---: | --- | --- |
| 0.00 | `blocks-3-7-9-blend-0.json` | `5f9b17d7c62b637bf2459bad2cbe6385c88cf09619416ae05ef30b0379be962e` |
| 0.25 | `blocks-3-7-9-blend-0.25.json` | `de9c8eb0e0d4dfbbd7daa60b709c30b65c9f42d4f4886c92363140db453d4e1e` |
| 0.50 | `blocks-3-7-9-blend-0.5.json` | `0a425e85e0d196a2f5f3df59a87f5c789621d4261e0bf009bba170986719a90d` |
| 0.75 | `blocks-3-7-9-blend-0.75.json` | `505020099befcd7e6a77c0bb6831ca880015521e5fea2f8a9e359f7122ab537b` |

Blends 0.25 and 0.50 clear the required two-point held-out improvement, but
they weaken joint-KL gain by 3.85 and 4.76 percentage points, respectively,
well beyond the allowed one point. Blend 0.75 fails both metrics and regresses
two isolated block outputs. No nonzero blend passes the pre-registered
selection rule.

### Verdict

Retain the unshrunk off-diagonal covariance. The correlation signal is
functionally valuable: weakening it monotonically gives back joint-KL and
block-output gains even where the held covariance metric initially improves.
The remaining fit/held gap should be tested with more fit rows, which improves
covariance estimation without deliberately discarding correlations.
