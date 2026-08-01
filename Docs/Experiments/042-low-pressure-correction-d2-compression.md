# Experiment 042: Fresh Low-Pressure-Correction D2 Compression

## Status

Complete; the compression and deployment workflow passed, but the fixed
candidate is rejected because its untouched selected mass is `0.73052`, below
both the established `0.75` floor and the correction's held-out implied target
of `0.74870`. This is the fresh complete campaign required after the accepted
Experiment 040 retained result and the byte-exact production replay in
[Docs/78-experiment040-production-integration-replay.md](../78-experiment040-production-integration-replay.md).

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

## Result

### Immutable state and complete export

The fresh run committed one identity and passed a fresh strict audit of 708
transitive artifacts, 26 blocks, and 130 factor owners. The rank sum is
111,776 and effective BPW is `1.0244947118`, exactly matching the retained
Experiment 040 production replay. Strictly validated resident artifacts occupy
9,276,915,496 bytes; the complete workflow reports 12,169,508,720 artifact
bytes including its export lifecycle.

The retained tuning chain is explicit:

- primary conditional checkpoint
  `sha256-d5c6c5dab72e02d2c01cb2697e2cacc9c7693db336dde41f939451cbcdec2c1f`
  and result
  `sha256-9e4f7d1d52f664e40df32cb56ac32ec2d37a169b42086f41b0666f558bd109f9`;
- correction checkpoint
  `sha256-70c70a6dc20e648e1a4848763476a2768934b730d2f25eb8ff20a966244ec323`
  and result
  `sha256-0d5fe322f97b99e1d5bf3c0a1842cd8dad37cd7653b9c4edb5f24b52e9ff149d`;
- calibrated final result
  `sha256-727cb13d3ac753f2ca0c91830d032f25a2dd72f7874848eebd326cd1a68aab76`.

The correction and final-norm fold add no represented payload. Logical weight
bytes are 2,739,492,464 and the exact packed payload is 89,480,664 bytes. The
GGUF is 417,340,544 bytes with SHA-256
`fedddc139e55044b8c5dd2911db08e4cc142e6897b5d044be2f8ef1674db2738`.
Packed reload and llama.cpp evaluation both completed.

The complete workflow took 12,622.13 seconds, including 10,227.69 seconds in
resident quantization and 8,058.07 seconds in committed block work. Peak GPU
and host allocations were 9,149,874,176 and 10,778,648,576 bytes. The initial
launcher attempt stopped before quantization on the publication-root guard;
after the junction-aware guard fix, the retry retained valid pre-quantization
inputs and produced all 130 commits fresh (`reused_commit_count = 0`). Primary
and correction caches/checkpoints remain independently namespaced and
resumable; byte-exact interrupted correction replay was established before
this campaign in Document 78.

### Same-factorization WikiText gate

The predeclared validation slice is offset 104, 48x512, token hash
`sha256:983ca15101666bef50ef4c1ccd44670a032e865e8f85230f942b08acc01e1b3d`.

| State | NLL | Full KL | Top-64 + tail KL | Student mass on teacher top 64 |
| --- | ---: | ---: | ---: | ---: |
| Pre-KD | 4.82511 | 1.56356 | 1.48079 | 0.82895 |
| Conditional epoch 8 | 4.67584 | 1.89501 | 1.80395 | 0.37417 |
| Corrected + 1.015 final | **4.37537** | **1.26287** | **1.18691** | 0.73052 |

The final state improves conditional NLL by 0.30047 and full KL by 0.63214,
and it also improves both over pre-KD. It nevertheless fails the absolute mass
gate. Teacher selected mass is 0.93587, so the configured 0.8 ratio implies
0.74870 on this slice; the observed final state misses it by 0.01818 and misses
the established 0.75 deployment floor by 0.01948. This failure is sufficient
to reject the fixed candidate without a post-hoc scale sweep.

### Packed, GGUF, tasks, and transfer

The standard packed result has WikiText PPL 171.87090; the exported GGUF gives
170.18196 on the same protocol, versus BF16 96.45961. This is substantially
better than Experiments 022 and 035 (228.55062 and 220.87905), and slightly
better than Experiment 040's packed 172.7052. It also improves dramatically
over this run's conditional checkpoint at 236.5.

The six-task 200-example mean is 0.46417. Because its comparison with the
same-run conditional state was ambiguous, the 1,000-example gate was run. The
independent packed comparison gives 0.45850 versus conditional 0.46367, delta
`-0.00517`, with paired task-stratified 95% interval
`[-0.01183, +0.00133]`. A regression is not established, although the direction
is unfavorable. For context, Experiment 037's calibrated tail-aware candidate
scored 0.44633 at 1,000 examples and Experiment 040's accepted packed endpoint
scored 0.45183; task inventories and frozen identities must still be compared
pairwise rather than by absolute mean alone.

Pinned C4 used the retained 48x512 token hash
`sha256:e34b788d48b021857df1130779e98d3936d4275bb17553e079a027e496bc2bef`.
Final versus conditional improved NLL from 5.14534 to 4.82922, delta
`-0.31612` with 95% interval `[-0.33965, -0.29297]`, and improved full KL from
1.56116 to 1.08733, delta `-0.47383` with interval
`[-0.50147, -0.44664]`. This is a decisive transfer pass and is stronger than
Experiment 040's corresponding marginal improvements, but it cannot override
the failed mass gate.

### Block 25 audit

Conditional KD again creates the distinctive last-block snapshot movement:
block-25 loss rises from 12,647 to 105,194, and the correction reduces it to
48,530. That large number is not itself evidence that a block-25 refit is
useful because the last block is measured directly against the logit-facing
teacher context and its scale is not comparable with earlier blocks.

The exact Experiment 041 teacher-context protocol was therefore screened once.
The joint block-25 refit improved local validation normalized RMSE by 7.63%,
but on the separate offset-460 24x512 screen it worsened NLL by 0.25190 and
full KL by 0.05754. The paired KL interval is entirely harmful at
`[+0.04418, +0.07009]`. No residual block-25-class exploitable defect is
present, so factor recovery and the untouched 48x512 confirmation were not
run.

## Decision

The production implementation is correct and the low-pressure correction
generalizes strongly for NLL/KL, C4 transfer, packed reload, and GGUF runtime.
The fixed `1.015` confidence fold does not generalize far enough across fresh
D2 factorizations to guarantee the selected-mass floor. Experiment 042 is
therefore a completed rejected experiment, not a new production default.

The next experiment should predeclare a calibration rule that derives the
smallest final-norm scale from a calibration-only mass statistic, then freezes
it before the existing WikiText and C4 gates. It must not tune a scalar on the
failed offset-104 gate, and it should preserve the current 32-step correction,
which produced the large transferable NLL/KL gains.

Retained evidence:

- `evidence/042/042-low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it`;
- `evidence/042/experiment042-final-vs-conditional-validation104-48x512-tail-mass.json`;
- `evidence/042/experiment042-matched-c4-validation104-48x512.json`;
- `evidence/042-analysis/experiment042-conditional-tasklimit1000-quality.json`;
- `evidence/042-analysis/experiment042-candidate-packed-tasklimit1000-quality.json`;
- `evidence/042/experiment042-block25-teacher-context-fit380-val384-screen460-24x512.json`;
- `Results/042/042-low-pressure-correction-d2-compress-and-benchmark-gemma-3-1b-it-summary.json`;
- `Results/042/gemma-3-1b-it-nanoquant.export-summary.json`.
