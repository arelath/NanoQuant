# Complete MLP Frozen-Model Transfer Gate

## Question

[Complete MLP-Stack Policy Composition](67-complete-mlp-stack-policy-composition.md)
shows a 29.68% relative KL improvement over equal-budget free-word MLP
reconstructions. This gate asks the more important absolute question: does
that logical MLP stack improve the retained Experiment 022 compressed model?

## Exact reconstruction export

The splice probe now optionally exports any complete reconstruction arm as an
atomic, hashed safetensors directory. The complete hybrid export contains:

- arm: `hybrid_operator_policy_refit`;
- 78 MLP weights across all 26 blocks;
- logical BF16 bytes: 1,242,178,096;
- SHA-256:
  `d431ec9b4ff471f19ae8b8df91dee3e70cdb80bd5fef106296cebcdf0232ca83`.

The dense export is an analysis artifact, not the proposed packed storage.
Its purpose is to transfer the exact refitted logical weights without
reimplementing the policy.

## Frozen-run transfer

`tools/probe_mlp_policy_frozen_transfer.py` loads a retained frozen run,
strictly verifies the overlay manifest and tensor hash, evaluates the
unchanged model, replaces only its MLP modules with the exported logical
weights, and evaluates again.

The first gate uses:

- retained Experiment 022;
- factorized backend;
- pre-global-KD state;
- exact retained 64×128 WikiText protocol;
- token hash
  `sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`.

## Result

| Metric | Experiment 022 pre-KD | Hybrid MLP transfer | Change |
| --- | ---: | ---: | ---: |
| Mean NLL | 5.612664 | 7.552107 | +1.939443 |
| Perplexity | 273.872886 | 1904.751818 | +1630.878932 |
| Relative perplexity | — | — | **+595.49%** |

The transfer fails decisively.

## Interpretation

The composition experiments establish a valid relative result: representation
and zero-bit scale policy substantially improve the untuned sign-word
reconstruction family. They do not establish that this family has reached the
absolute quality of the tuned resident factors.

Experiment 022 includes block-local factorized tuning, non-factorized tuning,
post-block refit, outliers, and other retained additions. The exported
codebook weights were fit from scratch to the original dense weights and did
not receive those model-level optimization stages. Replacing tuned MLP
modules wholesale discards too much functional recovery.

## Decision

Reject direct installation of the untuned complete MLP overlay.

- Do not promote the splice-relative gain as a higher-quality compressed
  model.
- Preserve the codebook allocation and heterogeneous scale policy as
  promising initialization/format evidence.
- Next test the zero-bit coupled MLP scale policies on Experiment 022's
  already tuned factors, where existing `scale_pre` and `scale_post` vectors
  can represent the changes without new bits.
- If codebooks advance further, integrate their constraints into resident
  factorized tuning rather than bypassing tuning.

## Evidence

- `evidence/m4/sign-word-codebook-probe/complete-mlp-hybrid-overlay/manifest.json`
- `evidence/m4/sign-word-codebook-probe/complete-mlp-hybrid-overlay/weights.safetensors`
- `evidence/m4/sign-word-codebook-probe/complete-mlp-hybrid-experiment022-prekd-transfer.json`
