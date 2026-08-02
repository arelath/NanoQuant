# Claim Status

This file separates paper-ready evidence from assumptions and missing data.

## Main claims

| Claim | Current status | Primary project evidence |
| --- | --- | --- |
| Exact-unit KL-calibrated rank allocation improves Gemma 3 1B at effectively equal BPW | Supported | `Docs/Experiments/022-gemma-3-1b-d2-kl.md` |
| D2 transfers across Gemma 270M and 1B | Partially supported | Experiments 021 and 022; same model family only |
| Selected post-KD MLP scale folding improves quality at identical packed bytes/BPW | Supported on retained Gemma state | `Docs/69-tuned-frozen-mlp-scale-placement.md` |
| Global composed-context folding is accepted | Assumed by the requested paper draft | Replace with fresh campaign evidence before submission |
| Tail-aware 256-step KD beats matched conditional KD | Supported on one frozen fresh factorization | Experiment 043 |
| Tail-aware KD is production complete | Not yet supported | Experiment 044 is predeclared |
| Mixed free/coded V is an equal-bit representation improvement | Supported in reconstruction, splice, seed, and runtime screens | Documents 54-57 |
| Mixed V improves a complete compressed/exported model | Not yet supported | Full campaign placeholder |
| Complete method generalizes across model families | Not supported | Llama/Qwen factorial campaigns required |

## Required before arXiv submission

1. Fill author, affiliation, and contact metadata.
2. Verify every bibliography entry marked `PLACEHOLDER`.
3. Complete the fresh global foldable campaign assumed by the draft.
4. Complete Experiment 044 or replace its claim with a clearly limited analysis result.
5. Run the full factorial design on Gemma from matched states.
6. Add at least two non-Gemma model families and preferably two model scales.
7. Produce the overview and Pareto figures from retained machine-readable evidence.
8. Fill the end-to-end BPW, quality, compression time, memory, artifact-size, and runtime table.
9. Recheck every number against its authoritative JSON rather than copying rounded documentation values.
10. Perform a literature search immediately before submission for concurrent sub-1-bit PTQ, sparse KD, scale-folding, and codebook-factor work.

## Claims deliberately excluded

- Covariance refinement is not presented as a production improvement; its complete transfer failed.
- Shared QKV or reciprocal attention grouping is not presented as a quality improvement.
- The selected-mass floor is not presented as a capability metric.
- Engineering parity, resume, and export do not substitute for model-quality evidence.
- Historical Experiment 018 is not used as a bitwise numerical oracle.
