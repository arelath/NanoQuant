# Resident Covariance-Aware Binary Refinement

**Date:** 2026-07-29
**Status:** implemented behind explicit dense-Hessian objective; complete compression pending

## Promotion evidence

The analysis campaign in Documents 44–48 established:

- 41.53% same-rank real-valued covariance headroom;
- input Hadamard decisively rejected;
- bounded direct sign/scale refinement selected at 32 left steps and 16 right
  batches;
- 8,192 covariance fit rows selected over 2,048 and 4,096;
- final 26-block held-out covariance error −24.06%;
- final 26-block joint KL −12.85%, NLL −0.455772 nats/token;
- 104/104 refined groups, 26/26 block aggregates, and all retained block
  outputs improved;
- exact same 0.999472370 BPW factor payload.

This evidence authorizes production integration but does not replace the
required complete compression, export, and quality gate.

## Resident design

Resident algorithm version is now 50. The path is enabled only when
`calibration.objective.kind` is `dense_hessian`; the ordinary diagonal recipe
is unchanged.

For each pending block, the resident engine:

1. attaches input hooks to fused QKV, O, gate, and up owners;
2. runs the original working block once on teacher inputs until 8,192 rows are
   accumulated;
3. regularizes and persists the dense fit covariance;
4. runs ordinary outlier selection, ADMM, retry, and diagonal scale fit;
5. after retry acceptance, refines the accepted residual's binary signs and
   scales under covariance;
6. uses the refined signs for both export binaries and factorized-tuning
   latent initialization;
7. continues through existing tuning, freezing, commits, resume, export, and
   evaluation.

Owners wider than 2,048 input features are skipped. This deliberately leaves
Gemma down projection unchanged because its 6,912-wide dense pre-scale solve
was not part of the validated screen. Empty or selected outlier columns remain
protected by forcing their factor pre-scales to zero.

The option and all objective sampling/regularization settings participate in
resident commit identity. The algorithm-version increment prevents discovery
from adopting version-49 commits into a covariance-refined run.

## Validation

Focused tests cover:

- exact objective improvement while preserving binary factors;
- protected outlier pre-scales;
- dimension validation;
- dense-objective mapping from canonical `RunConfig`;
- covariance option invalidation of resident commit identity;
- a complete tiny Gemma resident run that captures one block, emits seven
  refinement events, commits, and reloads under the same identity;
- pinned launcher defaults and explicit covariance CLI mapping.

Focused tests, Ruff, and full-source mypy pass. The next required gate is a
numbered complete compression experiment using
`execute_complete_compression`, followed by strict run validation, retained
WikiText quality, BPW, memory, artifact, and resume comparisons.
