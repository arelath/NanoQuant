# Experiment 037: Matched Tail-Aware D2

## Status

Running. Experiment 036 remains paused and is not reused. The common-state
worker uses two-new-block process slices so Windows releases retired rolling
activation file handles between resumes; each slice is validated before the
next one starts.

## Question

Does the calibrated top-k-plus-tail policy reproduce its retained Experiment
035 improvement on a fresh Gemma D2 factorization when compared with
conditional KD from the exact same pre-KD state?

## Common state

The numbered launcher
`experiments/037-matched-tail-aware-d2-base-gemma-3-1b-it.py` creates one fresh
resident state with global distillation disabled. It uses the immutable
Experiment 035 exact-unit KL profile solely to keep allocation fixed and avoid
building another multi-gigabyte uniform control. It does not import ranks,
factors, tuning values, or post-KD weights.

The common-state run must complete all 26 blocks and pass
`tools/validate_resident_run.py --require-complete` before branching. Both arms
are zero-copy hard-link forks of that validated directory.

## Matched arms

Both arms use 256 calibration samples, eight epochs, at most 32 batches per
epoch, batch size 1, learning rate 1e-5, top 64, at most 512 selected tokens,
8,192-token vocabulary chunks, and 128-token chunks. They differ only in the
loss contract:

| Arm | Loss | Tail mass weight | Final RMSNorm calibration |
| --- | --- | ---: | ---: |
| Control | conditional top-k | n/a | none |
| Candidate | top-k plus aggregated tail | 0.5 | 1.06, after checkpoint selection |

The candidate's 1.06 fold is model-specific and identity-bound. It is applied
only after the uncalibrated tail-aware checkpoint passes NLL/full-KL direction;
the final broad mass gate must be checked on the serialized calibrated model.

## Execution sequence

After the common state completes:

1. validate all resident artifacts and the 26-block prefix;
2. create `conditional` and `tail-aware` hard-link forks with
   `tools/fork_resident_run.py`;
3. run `tools/run_gemma_global_distillation.py` on the control with
   `--objective top_k --maximum-batches-per-epoch 32`;
4. run it on the candidate with
   `--objective top_k_tail --maximum-batches-per-epoch 32
   --tail-mass-weight 0.5`;
5. screen durable checkpoints on the predeclared WikiText-validation 48x512
   mass/NLL/full-KL/tail-KL gate;
6. fold scale 1.06 into the selected candidate using
   `tools/fold_global_tuning_final_norm_scale.py` and validate ordinary reload;
7. run the complete 64x128 WikiText and six-task/200-example benchmark on the
   common state, conditional control, uncalibrated candidate, and calibrated
   candidate;
8. run the pinned C4 48x512 paired gate;
9. complete logical/packed reload, GGUF export, byte/BPW accounting, and the
   numbered export summary for the accepted arm.

## Acceptance

The candidate must retain the common state's effective BPW and represented
payload bytes, keep teacher-top-64 mass at least 0.75 on the broad gate, and
improve NLL, full KL, and tail KL over conditional KD. It must improve
WikiText perplexity over the common pre-KD state without regressing the
six-task mean versus conditional KD, pass C4 with paired confidence, and
complete all artifact/export contracts. Training loss, a short monitor, or a
retained-state result alone cannot accept Experiment 037.
