# Experiment 036: Composed-Context-Initialized Foldable MLP D2

## Status

Implementation and preflight validation in progress. The numbered launcher is
`experiments/036-composed-context-initialized-foldable-mlp-d2-gemma-3-1b-it.py`.

## Purpose

Experiment 036 is the controlled successor to Experiment 035. It preserves the
same complete D2 compression, global top-k distillation, 64-step conservative
foldable continuation, export, and retained quality protocol. Its only numerical
change is the initialization of the foldable stage.

Experiment 035 started the continuation at identity and failed its untouched
held-out NLL gate. Experiment 036 starts from the accepted six-block
composed-context policy:

```text
0:output, 17:joint, 18:joint, 23:joint, 24:joint, 25:joint
```

## Portable initializer contract

The accepted component overlay from Experiment 022 cannot be installed directly
on a fresh run: it is correctly bound to Experiment 022's frozen and global-KD
identities. Experiment 036 therefore uses a model-specific log-multiplier seed.
The seed records 23 semantic row/column axes and 124,416 FP32 fitting values, but
these are training inputs rather than deployed payload.

At runtime the stage:

1. validates the seed's model source, revision, tensor inventory, and pinned
   SHA-256;
2. applies the seed covariantly to the fresh run's own scales, outliers, and
   correction patches;
3. folds the seed into BF16 components without changing shapes, dtypes, or bytes;
4. reinstalls identity continuation multipliers, so the identity penalty is
   centered on the composed-context seed rather than pulling back toward the
   unrefitted post-KD state;
5. trains, checkpoints, folds, exports, and benchmarks through the existing
   complete workflow.

The initializer artifact is
`experiments/assets/gemma-3-1b-it-composed-context-six-block-initializer/`.
Its multiplier SHA-256 is
`718b6d79f37fb1181b64be4faf08141045e302ca7931862836d2a5d5b80fc73c`.
Regeneration from the retained dense and component evidence exactly replays all
54 accepted BF16 component tensors with zero maximum error.

## Acceptance gates

The experiment must:

1. complete and freshly validate all 26 resident blocks and transitive artifacts;
2. prove the seeded and final folds preserve component shapes, dtypes, and bytes;
3. prove exact folded replay and exact logical-to-packed conversion;
4. keep effective BPW unchanged and report any physical packed-byte variation;
5. improve untouched 48x512 paired NLL over the same-run post-KD state with a
   95% upper interval below zero and no supported KL regression;
6. complete the retained WikiText 64x128 and six-task benchmark;
7. compare protocol identities and quality against Experiments 022, 035, the
   same-run post-KD ablation, and the earlier Phase C result.

No result is accepted from the training loss, initializer replay, or WikiText
perplexity alone.
