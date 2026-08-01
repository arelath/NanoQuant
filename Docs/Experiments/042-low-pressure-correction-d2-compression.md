# Experiment 042: Fresh Low-Pressure-Correction D2 Compression

## Status

Predeclared and not yet started. This is the fresh complete campaign required
after the accepted Experiment 040 retained result and the byte-exact production
replay in [Docs/78-experiment040-production-integration-replay.md](../78-experiment040-production-integration-replay.md).

Experiment 036 remains paused. The rejected block-25 refit and transplanted
foldable-MLP seed are not included.

## Fixed candidate

Experiment 042 uses the pinned `google/gemma-3-1b-it` revision and the exact D2
allocation/factorization recipe used for the matched Experiment 037 frozen
state. It creates a new factorization; it does not fork retained tensors.

After ordinary eight-epoch conditional top-64 KD, it applies the production
Experiment 040 continuation:

- one executed epoch and 32 batches;
- learning rate `1e-5`;
- cosine horizon 128 steps, matching checkpoint epoch 1 of the original
  four-epoch analysis schedule;
- minimum teacher selected-mass ratio `0.8`;
- one-sided mass coefficient `2.0`;
- Gemma final-RMSNorm effective-weight scale `1.015`.

All correction cache, checkpoint, initializer, result, and calibration
identities must be explicit and resumable. The fold must remain byte-neutral.

## Acceptance gates

The experiment is not complete until all gates pass:

1. Strictly validate all 26 blocks, 130 committed layers, journals, descriptors,
   hashes, and transitive artifacts before evaluation/export.
2. Retain and report the primary conditional KD artifact, correction checkpoint,
   corrected artifact, and final calibrated artifact.
3. Confirm effective BPW and represented payload bytes are unchanged by the
   correction and fold.
4. Complete `execute_complete_compression`, exact logical-to-packed conversion,
   packed reload, llama.cpp checkpoint construction, GGUF export, and export
   summary.
5. Run the ordinary quality workflow from the packed artifact: 64x128 WikiText
   and six tasks at 200 examples each.
6. Run the same prepared quality inputs through the exported GGUF in llama.cpp.
7. Run the predeclared WikiText 48x512 selected-mass/NLL/full-KL confirmation
   and the pinned C4 48x512 paired NLL/KL gate after the main workflow.
8. Run the 1,000-example six-task confirmation if the 200-example task mean
   does not establish a clear regression.
9. Check block snapshots for a new block-25-class defect; do not fit a refit
   unless a fresh, held-out defect is demonstrated.
10. Report ranks, BPW, logical/packed/GGUF bytes, stage wall time, peak GPU and
    host memory, artifact bytes, resume behavior, and comparisons with
    Experiments 022, 035, 037, and 040.

Training loss, a fixture, or retained Experiment 040 quality cannot satisfy
these gates for the fresh run.

## Storage and execution safety

The D: volume does not have sufficient free space for a new resident store plus
the complete export lifecycle. The run's ignored `evidence/042`, `outputs/042`,
and `Results/042` roots will be directory junctions to one dedicated C: staging
root before launch. Keeping all three roots on the same volume preserves the
required hard-link publication behavior. No existing evidence is removed.

Only one CUDA worker may run. Before launch or resume, inspect the process
command line, device lease, journal tail, and `nvidia-smi`. Poll long-running
stages sparsely; journals and committed artifacts are authoritative.
