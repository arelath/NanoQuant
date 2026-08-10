# Adaptive free-row and functional codebook-tuning gate

## Scope

This staged analysis resumed the half-word product-codebook experiment after
the residual-product representation failed. It tested two zero-rate and
fixed-rate changes without modifying the resident algorithm or packed format:

1. choose arbitrary free rank components before activating the product
   constraint, then permute them into the stored free prefix;
2. update product assignments and shared half-table bits against disjoint
   functional linear-output fit and validation sets.

The candidate remained the Experiment 057-style `down_proj` representation:
rank 1,152, a 16-bit product selector, exact Experiment 056 outliers, 1,200
ADMM steps, 100 warmup steps, and the retained 8+8 binary search.

## Stage B: zero-bit adaptive free components

The pre-constraint selector estimates each component's weighted coding cost,
keeps the highest-cost components free, permutes them into the physical
prefix, and refits the product tables on the coded suffix. The permutation is
absorbed into the factor components and therefore stores no metadata.

Representative reconstruction screens selected roughly 280 different free
components per layer. Reconstruction changes were very small and mixed, so
the held-out gate compared only the policy-relevant winners:

| block | free rows | prefix KL | adaptive KL | adaptive minus prefix | paired 95% interval | decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 672 | 0.173897 | 0.174169 | +0.000272 (+0.16%) | [-0.005564, +0.006142] | null |
| 24 | 704 | 0.023779 | 0.022885 | -0.000895 (-3.76%) | [-0.001490, -0.000322] | pass |

Each comparison contains 96 sequences in two 48-sequence windows. Block 12
did not advance because its adaptive reconstruction candidate regressed its
matched prefix control. The block-24 prefix window was rerun with the exact
warmup literal after an interrupted PowerShell launch rounded the value enough
to move constraint activation one iteration earlier.

Adaptive row selection is therefore layer-specific. It is not justified as a
blanket down-projection policy, but block 24 demonstrates a statistically
supported zero-bit gain.

## Stage C1: mutable product assignments

The adaptive block-24 candidate received two exact product-payload passes.
Candidate words were ranked by the isolated weight objective, then accepted
only when they improved both functional fit and validation linear-output
errors. The functional windows (WikiText offsets 96--99 and 100--103) were
disjoint from the final KL windows (offsets 0--47 and 48--95).

The tuner accepted 76 words and changed 198 decoded signs:

- functional fit error: 805.5582 to 804.9991;
- functional validation error: 1005.3073 to 1004.7534;
- final 96-sequence KL versus adaptive-only: +0.0000143 (+0.062%);
- paired 95% interval: [-0.0000208, +0.0000497].

The model-level result is null despite improvements on both functional proxy
sets.

## Stage C2: mutable shared table bits

A bounded diagonal Gauss--Newton screen aggregated functional sign pressure
over every use of each shared half-table bit. Proposed bits were evaluated
exactly and accepted only if both functional fit and validation errors
improved. Four passes accepted four table-bit flips, changing 1,569 decoded
signs in addition to the assignment updates:

- functional fit error: 805.5582 to 803.9667;
- functional validation error: 1005.3073 to 1003.9091;
- weighted reconstruction error: 0.9873650 to 0.9889969;
- final 96-sequence KL versus adaptive-only: +0.0000542 (+0.237%);
- paired 95% interval: [+0.0000111, +0.0000978].

The shared-table update significantly regresses logits even though its fit and
validation block-output objectives both improve. This is direct evidence that
the local linear-output proxy is not suitable for codebook tuning or global
allocation.

## Evidence

- adaptive/prefix final gates:
  `evidence/m4/product-codebook-adaptive-functional`;
- assignment-tuned final gates:
  `evidence/m4/product-codebook-adaptive-functional-tuning`;
- table-bit-tuned final gates:
  `evidence/m4/product-codebook-adaptive-table-tuning`;
- reconstruction screens:
  `evidence/m4/product-codebook-warmup-prefix-block0-24`,
  `evidence/m4/product-codebook-warmup-prefix-block12`, and
  `evidence/m4/product-codebook-preconstraint-adaptive-block0-24`.

## Decision

Do not promote functional assignment or shared-table tuning under the current
block-output objective. Do not wire either path into resident tuning.

Retain zero-bit adaptive free-row permutation as an analysis candidate because
block 24 has a robust held-out improvement. Any broader use must be selected
per layer using direct model-logit KL on disjoint selection and final windows.
The next allocation experiment should measure candidate-specific logit deltas
directly and compose accepted blocks sequentially; another local matrix or
block-output multiplier is not supported by this evidence.
