# Experiment 036: Composed-Context-Initialized Foldable MLP D2

## Status

Paused after uniform-control block 10. The numbered launcher is
`experiments/036-composed-context-initialized-foldable-mlp-d2-gemma-3-1b-it.py`.

The original full six-block-seed candidate is rejected before its candidate
campaign. Review probes on retained Experiment 035 show that block 25 transfers
strongly, but blocks 0 and 18 are harmful and the full seed is worse than block
25 alone. A fresh block-25 refit on the active campaign is significantly better
than the transplanted value and reaches retained perplexity `148.747976` at
zero byte delta. Experiment 036 must not resume unchanged. See
[73-experiment-035-review-probe-results.md](../73-experiment-035-review-probe-results.md).

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

## Paused-run state and revision decision

The resumable uniform-control run is retained under
`evidence/036/036-d2-uniform-control-gemma-3-1b-it/`. Its journal contains 66
records and 11 complete blocks through block 10. No Experiment 036 candidate
block was started.

The review probes demonstrate that the retained top-k objective actively
penalizes the correction preferred by full-vocabulary NLL/KL. The original
64-step continuation is therefore also removed from the revised candidate
direction. The next candidate must fit and screen teacher-context MLP scales on
its own post-KD factors, initially block 25 only, then complete the ordinary
zero-byte folding, export, and quality gates.

The deeper audit in
[74-block25-anomaly-and-topk-tail-mass-audit.md](../74-block25-anomaly-and-topk-tail-mass-audit.md)
supersedes that immediate next step. Conditional top-64 KD collapses the
student probability mass on the teacher's selected entries from 0.8977 pre-KD
to 0.5133 post-KD while improving only the renormalized conditional objective.
Block 25 is the highest-leverage compensator for that model-wide error, not an
independent block-local repair. Experiment 036 remains paused while a
top-64-plus-tail objective is tested on the retained Experiment 035 pre-KD
state. Only the remaining block-25 marginal after that repair can justify a
revised production candidate.

That objective ablation is complete. At 256 matched steps the tail-aware arm
kept selected-token mass at 0.826 and improved full KL to 1.163, while the
conditional control collapsed mass to 0.580 and worsened full KL to 1.378.
Experiment 036 remains paused pending factor-compatible checkpoint export,
the remaining block-25 marginal test, and the complete quality gate. See
[75-topk-tail-kd-objective-ablation.md](../75-topk-tail-kd-objective-ablation.md).

Those gates are now complete. Tail-aware KD improved WikiText perplexity from
257.49 pre-KD to 188.72, but did not improve the retained six-task mean. More
importantly, the fresh block-25 refit after tail-aware KD worsened untouched
48x512 NLL by 0.05971 and full KL by 0.02022. This confirms that the earlier
block-25 gain was compensation for the conditional-KD defect, not reusable
initialization evidence. Experiment 036 remains paused; it should not inherit
that compensator.

The selected 0.5 tail-mass coefficient has now also passed a pinned 48x512 C4
gate against the exact 1.0 tail objective: paired NLL delta `-0.044465`
(`[-0.050693, -0.038023]`) and full-KL delta `-0.018483`
(`[-0.021759, -0.015215]`). This strengthens the global-KD direction but does
not revive Experiment 036. The next full campaign must receive a new number
and compare a production-integrated tail-aware global KD branch with a matched
conditional branch from one fresh frozen factor state. The required staged
gates are in
[76-tail-aware-global-kd-final-experiment-plan.md](../76-tail-aware-global-kd-final-experiment-plan.md).
