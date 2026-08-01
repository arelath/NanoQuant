# Experiment 040 Production Integration Replay

## Decision

The accepted Experiment 040 correction is now a normal canonical workflow
stage rather than an analysis-only checkpoint materialization. The production
path exactly reproduces the accepted trainable tensors, resumes after an
intentional process boundary, folds the selected 1.015 Gemma final-RMSNorm
scale into an immutable global-tuning artifact, and passes strict resident
validation at unchanged effective BPW.

This closes the productionization gate. It does not replace the still-required
fresh full campaign, compressed-model quality benchmark, packed reload, and
GGUF export.

## Implemented path

The canonical order is now:

1. primary conditional top-k KD;
2. a warm-started one-sided selected-mass correction;
3. an immutable final-RMSNorm effective-weight calibration;
4. normal frozen-model assembly and export from the final active tuning.

The correction binds the exact initializer global-tuning reference into its
protocol and checkpoint identity. Its cache, training checkpoint, and completed
result use stage-specific pointers, so an interruption cannot overwrite the
primary KD state. Completed stages are idempotent: source blocks, protocol, and
tokens are validated before the retained result is reactivated.

The disabled-by-default Experiment 040 policy is explicit in canonical config:

- one executed correction epoch;
- 32 batches and 32 optimizer steps;
- learning rate `1e-5`;
- teacher selected-mass ratio `0.8`;
- one-sided mass-loss weight `2.0`;
- cosine scheduler horizon `128` steps;
- final Gemma RMSNorm effective-weight scale `1.015`.

## Scheduler-horizon defect found by replay

The first production replay used a 32-step cosine horizon because it described
the retained endpoint as a one-epoch run. Its teacher sample order, token
positions, top-64 indices, selected logits, and full-vocabulary normalizers were
all byte-exact, but its epoch loss was `2.192406` rather than the retained
`2.182741`.

Experiment 038 had selected checkpoint epoch 1 from a four-epoch run, so that
checkpoint was trained under a 128-step cosine horizon. Executing only 32 steps
is not equivalent to scheduling only 32 steps. The production configuration now
separates executed steps from scheduler horizon and validates that the horizon
covers the configured training steps.

With the horizon fixed at 128, the interrupted production checkpoint loss was
exactly:

```text
2.182741157710552
```

This equals the accepted Experiment 038/040 checkpoint without tolerance.

## Interrupted/resumed retained replay

The replay is retained at
`evidence/041/experiment040-production-replay`. It is a hard-link fork of the
audited Experiment 037 conditional-KD initializer.

The first process cached 32 normalizer-bearing teacher batches, completed the
32 correction steps, committed checkpoint
`sha256-2504284c3f4b9e175923fb12238a9a91830547dc4276455590d676bc652e5321`,
and then stopped at the requested interruption. A separate process loaded that
checkpoint at step 32/32, froze the tensors, and committed production correction
artifact
`sha256-f04b5678938efeac73b5c10c68d63d3c9412358a71bbfe1635badd4f6366f392`.

Compared with accepted raw Experiment 040 artifact
`sha256-b6a86b0640fce35d031d393c7899f236e58c52e8059c427be56071109db599e3`:

- every tuned block state and tensor reference is identical;
- every auxiliary parameter reference is identical;
- epoch loss and completed-step count are identical.

The production artifact identity differs appropriately because its protocol,
cache-byte accounting, resource measurements, and block-snapshot evidence are
stronger than the analysis materialization.

The production 1.015 fold committed artifact
`sha256-28462b9f17e9846512a09987c88b78406c157916a6e88017eada9981ad5b6768`.
Its tuned blocks and auxiliary references exactly match accepted folded
Experiment 040 artifact
`sha256-8f1d413a7ed4ebe8fefacb9b9326b4c201dec4e6f81d6fbcdf12ae52ce8eb914`.

## Validation

Strict validation is retained in
`evidence/041/experiment040-production-replay-validation.json`:

- 708 transitive artifacts validated;
- 26 blocks and 130 layers complete;
- effective BPW `1.0244947117998688`;
- no inactive journal records;
- active final tuning reloads through the normal factorized path.

The replay also exposed that this PyTorch build rejects a raw `"cuda:0"` string
in allocator instrumentation even though module placement accepts it. Global
distillation resource instrumentation now resolves CUDA strings to
`torch.device`, with a regression test for indexed CUDA devices.

## Next gate

Run one fresh complete Gemma campaign with the correction and final-norm policy
enabled from canonical config. It must finish validation, WikiText and task
quality, the pinned C4 confirmation, effective-BPW and byte accounting, logical
and packed reload, compressed-model quality, llama.cpp checkpoint, and GGUF
export. Experiment 036 and the rejected block-25 refit remain out of scope.
