# Experiment 080: conditional replacements and fixed-sign refit

## Measured outcome

Both 26-block continuations completed. **The fixed-sign scale refit did not recover
the predicted mixed-format benefit.** The conservative mix got worse on held-out
projection KL and dense WikiText perplexity, despite improving local held-out
activation error in all 25 blocks that retained trainable NanoQuant factors.

| Endpoint | GGUF bytes | Projection KL | Dense WikiText perplexity | Native WikiText perplexity |
| --- | ---: | ---: | ---: | ---: |
| Original 080 before | 547,391,648 | 1.369504 | 130.312663 | 130.481960 |
| Original 080 after | 547,391,648 | 1.378487 | 132.377926 | 132.311188 |
| Conservative mix before | 547,349,056 | 1.609000 | 140.056854 | 140.145062 |
| Conservative mix after | 547,349,056 | 1.704683 | 180.001714 | 180.100274 |

Native llama.cpp evaluation confirms the regression. All four evaluations complete
with the same retained tokens and runtime identity. Native mixed perplexity rises
28.51%, versus 1.40% for the control. These are compatible diagnostic exports, not
accepted replacement models. Full evidence is linked in the
[result report](../../Results/080/080-mixed-format-refit.md).

Mixed KL rises by 0.095683, versus 0.008983 for the original control. The mixed
penalty relative to that control grows by 0.086699, from 0.239496 to 0.326196.
The mixed model regresses on all 16 KL windows; the paired window standard error
of its before/after change is 0.014927. These windows are one text sample and do
not establish generalization across independent corpora.

The original control improves local held-out activation MSE in 13/26 blocks;
the mixed arm improves it in all 25 trainable blocks (block 0 is entirely standard
and remains unchanged). At block 25, mixed normalized validation MSE falls from
0.0740412 to 0.0656938 when refitting that block with its updated prefix fixed.
This is a local before/after comparison, not the total network improvement versus
the untouched original prefix. It demonstrates why local activation MSE alone is
not an acceptance criterion for this continuation.

Original/mixed refits change 488/252 BF16 scale tensors respectively. Both exports
pass byte-exact checks on embeddings, signs, standard weights and all other
non-scale payloads, and preserve the source file size. Summed recorded block work
is 704.6/533.0 seconds, excluding startup/export; these are diagnostic timings,
not matched production performance claims.

**Decision:** keep 080 as the control and reject this refitted mixed candidate.
Do not promote the additive allocation or expand the production mixed-plan
pipeline from these results. Conditional block replacements are more informative
than another Gaussian sweep: only blocks 0–4 improve the compressed model before
refitting. A subsequent candidate must select conditional benefits subject to
the byte budget; merely retaining the favorable swaps need not fit that budget.

This result rejects this bounded production-style scale-refit schedule, not all
possible mixed compression. The preferred next adaptation test is output-logit
distillation on these same two allocations, keeping signs fixed and retaining the
original-only training control and a separate held-out model-quality gate. This
isolates the objective change. Binary-factor changes or fresh factorization are
subsequent capacity tests if needed. Conditional reallocation is a separate
experiment, rather than changing the allocation and training objective together.
Keep these diagnostics bounded before a full resident mixed-plan implementation.

## Question and fixed protocol

Does refitting the retained NanoQuant factors recover the quality lost by the
equal-budget conservative mixed-format allocation? The two-stage-minimum mix is
used so truncating a jointly fitted residual stage is not a confound.

This first continuation uses **fixed-sign factor-scale refitting**, matching the
trainable surface of production `post_block_refit`. It is not a new binary-factor
search or a full mixed resident compression run. The binary signs, ranks, standard
quantized weights, embeddings and norms stay fixed. This limitation applies to
both positive and negative conclusions.

The original 080 artifact graph passed a fresh `--require-complete` audit: all 26
blocks, 836 artifacts, with the rolling-retention exception for 25 retired
activation generations. Only the last original activation generation remains;
the pinned calibration tokens remain available. Every teacher and student stream
is therefore recomputed. No retired activation or stale compressed-prefix input
is used for refitting.

## Conditional replacement screen

The [conditional plan](080-mixed-refit-conditional-plan.json) measures 28 cases:
original 080, the full conservative mix, and each of its 26 block-level replacement
sets installed separately into otherwise-original 080. Shared QKV switches as a
whole group. These are the exact previously allocated format choices, not a new
selection search.

On the fixed projection KL protocol, only the first five block-level replacement
sets improve 080. The other 21 worsen it. The sum of conditional changes is
+0.204779 nats/token; applying all changes together raises KL by +0.239496, from
1.369504 to 1.609000. The largest individual increases are blocks 17 (+0.042954),
9 (+0.041210), 25 (+0.032491), 8 (+0.027297) and 10 (+0.026080).

## Matched refit controls

The [refit plan](080-mixed-refit-plan.json) fixes two arms:

- Original 080, with all 130 binary storage units retained.
- Conservative mixed 080, with 74 binary units retained and 56 standard units
  frozen. Its exact file size is 547,349,056 bytes, versus 547,391,648 for 080.

Both get the same sequential, blockwise schedule and data. The original-only arm
controls for benefits of extra training unrelated to replacing formats. A block
with no remaining binary owners is propagated unchanged, without an optimizer.

Use the original run's pinned 50/50 UltraChat `train_sft` and WikiText-2 `train`
calibration. Training uses rows 0, 2, ..., 254, truncated to 512 tokens: 128
sequences. Activation validation uses rows 1, 9, ..., 249 at the same length: 32
different sequences. They are held out from this continuation, though originally
part of 080's calibration. Validation is measured before/after, not used for
checkpoint or hyperparameter selection.

For each block, evaluate the clean teacher on its clean preceding activations.
Install the arm's actual standard weights and binary factors. Refit the remaining
factor scales against the clean block outputs, using inputs propagated through
that arm's already-refitted preceding blocks. Retain production output-importance
weights and `post_block_refit` parameter selection, including its exclusion of
coupled shared-QKV pre/mid scales; QKV post scales remain trainable.

The fixed schedule is eight epochs, learning rate 1e-5, batch four, microbatch one,
seed zero, zero weight decay, and the existing fused AdamW configuration with
FP32 master scales and BF16 forward scale rounding. There are 256 optimizer updates
per trainable block. Use the final endpoint, without early stopping or best-epoch
selection. Signs are immutable throughout. This is a bounded continuation using
shorter calibration windows than 080, not a replay of its original full recipe.

Each completed block stores its BF16 scales and a hash-bound receipt. Resume
reconstructs prior blocks and propagates their saved state, without repeating
completed tuning. Training executes factorized linears; propagation uses the exact
CPU-reconstructed, BF16 dense weights represented by the persisted scales. Record
both arithmetic paths' activation errors, rather than attributing their difference
to training. Source norms are checked against the fixed clean model.

## Evaluation and artifact gates

Score original and mixed files before and after refit on the same sensitivity
teacher and retained WikiText-2 test protocol. The sensitivity measurements and
refit keep the clean embedding fixed; all exported files and WikiText evaluations
keep original 080's Q8_0 embedding payload fixed. No embedding parameter is trained
or optimized. The pinned WikiText test split differs from calibration's train split.

The research GGUF writer changes only existing BF16 scale payloads. It validates
every other tensor's raw bytes, all tensor types/shapes, the tensor inventory and
exact source file size. Binary signs, standard weights, norms and embeddings must
remain byte-identical. Native evaluation provides an additional compatibility and
quality check. This does not alter resident schemas or the mandatory complete-run
export lifecycle for numbered compression experiments.

Report both mixed-after minus mixed-before and mixed-after minus original-after.
An improvement versus untreated mixed alone does not establish a mixed-format gain
over a similarly refitted NanoQuant control. Compare KL and WikiText NLL/perplexity
separately; improvements in activation MSE do not establish model-quality recovery.

## Reproduction

From the repository root, with the GPU free:

```powershell
.\.venv\Scripts\python.exe tools/refit_mixed_gguf.py `
  Docs/Experiments/080-mixed-refit-plan.json --arm original `
  --output outputs/mixed-refit-1b/original

.\.venv\Scripts\python.exe tools/refit_mixed_gguf.py `
  Docs/Experiments/080-mixed-refit-plan.json --arm mixed `
  --output outputs/mixed-refit-1b/mixed
```

The original arm first completed a one-block feasibility invocation with
`--stop-after-blocks 1`, then resumed the same eight-epoch-per-block schedule.
The source, plan and tool identities must remain unchanged to resume.

The exact token-window audit is reproducible with:

```powershell
.\.venv\Scripts\python.exe tools/audit_mixed_refit_data.py `
  Docs/Experiments/080-mixed-refit-plan.json `
  --teacher-root Datasets/gemma-3-1b-it-fp16-sensitivity-16x256 `
  --quality Results/080/080-fast-imatrix-no-kd-compress-and-benchmark-gemma-3-1b-it-quality.json `
  --output outputs/mixed-refit-1b/data-audit.json
```

It found no exact held-out content windows in the continuation training tokens,
and reproduced the retained WikiText token/mask digest. This is an exact-window
check, not a claim that the broader source corpora contain no related text.

The [quality plan](080-mixed-refit-quality-plan.json) compares all four endpoints
using `tools/measure_projection_interventions.py` on the expanded FP16 sensitivity
measurement `307b4cf1f11bc153222ddc75e79e9da66462b497ed7f4ce131f3f3ef1fc66114`.
`tools/benchmark_mixed_gguf_wikitext.py` scores the same four files with the retained
080 quality JSON and `--llama-cpp-root D:/dev/research/llama.cpp`. Use distinct
hardlink basenames for the two refitted files so native report names are unique.

## Software validation

Five focused CPU tests pass, covering multi-stage scale export, unchanged payloads,
shared-QKV parameter selection, invalid scales and token-window overlap detection.
Repository Ruff and mypy pass. The full default suite reports 1,871 passed,
54 deselected and three existing Experiment 079 launcher/identity failures in
`test_architecture.py` and `test_promoted_recipes.py`; those unrelated files were
not changed. The full log is retained with the experiment evidence.
