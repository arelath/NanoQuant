# Residual stage quality experiment plan

Date: 2026-11-10

## Objective

Test whether additional binary factors fitted to reconstruction residuals improve model quality at the same stored-byte budget. Prioritize this test, then a controlled distillation comparison, before implementing a new deployment format or dynamic scale experts.

This plan follows [the second opinion](ImprovementSuggestions/11-10-26-SecondOpinion-OneBitAndFullRank.md), [full-rank storage accounting](108-full-rank-nanoquant-bits-per-parameter.md), and the negative functional-proxy result in [Doc 100](100-adaptive-free-row-functional-tuning-gate.md).

## Ordered work and acceptance gates

### 1. Equal-byte residual factor gate

- [x] Audit retained inputs and baseline availability; freeze identities and storage accounting for the initial pilot.
- [x] Reproduce the block-12 reconstruction screen at an exact affordable rank with complete payload accounting, retaining candidate tensors.
- [x] Extend to blocks 0, 12, and 25 with several fitting seeds.
- [x] Evaluate initial block-12 candidates on fresh paired model-logit KL and NLL sequences.
- [x] Extend the functional comparison to the new multi-block fits.
- [x] Compose promising replacements in the compressed model and confirm on an independent slice.

Compare the current factor plus INT8-column correction, a full-rank factor plus an independently scaled residual factor, and a factor plus a partial sign-plane correction. Use the same original target, importance weights, protected columns, and common first-stage initialization where applicable. Distinguish a controlled reconstruction screen from a direct comparison to the deployed Experiment 060 state; the retained Experiment 067 single-matrix optimum is not that state.

Charge packed binary factors, FP32 scales, protected-column values, indices, and alignment. Record both used bytes and unused budget. Equal budget means no candidate may exceed the common cap; rank granularity can leave slack. Charge a sign plane's row/column scales and selected-column indices too. No model-level claim may use factor-sign-only BPW.

First reproduce the retained block-12 fixture on CPU without resuming the cancelled search. Use 800-step production ADMM and two-pass scale fitting, fitting the residual of the frozen first stage. Save candidate tensors, source hashes, settings, weighted NMSE, and byte breakdowns. This is a diagnostic milestone only. The earlier near-2-BPW NMSE of 0.089 was interpolated and must not be used as a measured endpoint.

For the functional gate, adapt the existing dense-overlay/splice evaluator. Materialize the persisted candidate precision before scoring. Keep fit, selection, and confirmation data disjoint and register evaluation slices through the existing mechanism. An initial 48-by-512 sequence selection slice is a screen, not a final capability result. Report sequence-paired confidence intervals for KL and NLL; expand confirmation if uncertainty remains. Require a supported KL improvement without a material NLL regression, then confirm the composed candidate rather than assuming independent layer gains add together. Predeclare any practical noninferiority margin before inspecting results.

No packed-runtime changes are needed for the dense screen. Separate stages with independent outer scales cannot generally be collapsed exactly into the current single sign-binary unit. Production stage support is conditional on functional success.

### 2. Controlled distillation comparison

- [x] Verify and preserve a common checkpoint; explicitly document matched continuation when the historical pre-distillation state is unavailable.
- [x] Compare the current objective, full-distribution KL, and full KL plus normalized hidden-state anchoring at matched tokens, trainable parameters, and optimization settings.
- [ ] Extend only promising short-run arms to longer training.

Keep the untreated checkpoint as a control. Evaluate held-out KL/NLL, out-of-domain perplexity, and downstream tasks. Introduce sign updates and checkpoint averaging as later ablations, rather than changing all variables together. Measure memory before choosing teacher/student co-residency; full KL does not itself guarantee stable long training.

### 3. External baseline and embedding precision

- [ ] Download a published low-bit Llama 3.2 1B model and compare with the retained NanoQuant run using matching scoring inputs and runtime where possible.
- [ ] Audit actual tensor types and decoder BPW, model provenance, and embedding precision before calling the comparison matched.
- [ ] Test changing only Gemma's embedding precision and report whole-file savings separately from decoder BPW.

Keep external baselines downloaded, following the user's preference. An unavailable genuinely matched artifact should be reported as a limitation rather than silently replaced by a self-quantized baseline.

### 4. Complete and transfer the winning recipe

- [ ] Compare a complete residual-stage model against Experiment 060 re-distilled with the same winning objective.
- [ ] Report per-task and average accuracy, KL/NLL, out-of-domain perplexity, actual bytes, inference speed, and memory.
- [ ] Implement packed/GGUF/runtime support only after the functional result justifies it.
- [ ] Complete a Gemma 3 4B run before prolonged further optimization of 1B.

Complete compression runs must use `execute_complete_compression` and the repository's validation/export lifecycle. A resident numerical-path change requires an algorithm-version increment and resume-compatibility validation.

### 5. Conditional follow-ups

- [ ] Test residual-stage allocation under a fixed 1-BPW global budget, funding additions with measured low-cost rank reductions elsewhere.
- [ ] Test token-adaptive scale experts only after the training objective is validated; charge their parameters and runtime overhead.

## Interpretation and stopping rules

If the residual-stage improvement fails to transfer, evaluate the sign-plane candidate through the same gate and stop deployment work on stages. Do not resume hour-long fixed-format fit searches based solely on a reconstruction improvement. Full rank does not imply an exact binary-factor fit, but the previous searches justify assigning additional search low priority.

Near-Gaussian kurtosis alone does not establish white residuals. The Gaussian rate-distortion reference does not prove optimality for these weights. Published results on larger models and different training budgets provide context rather than a quality ceiling. Above-chance retention may be a supplementary metric; retain raw per-task accuracy and task-average accuracy as directly interpretable endpoints.

## Execution log

- 2026-11-10: Plan recorded. Initial process inspection found no NanoQuant worker. An unrelated ComfyUI process owns a CUDA context; initial fixture work will use CPU. Existing dirty evidence and user documents are preserved.
- 2026-11-10: Experiment 060's run/artifact directory is absent. Its GGUF remains and matches SHA-256 `5e445cab6c93f03492ca0c9cea4ce7977fc780b9dbd75ad60578aef609dad02c`. Existing GGUF reconstruction code successfully materialized finite block 0, 12, and 25 `down_proj` weights. Their NanoQuant tensor payloads each occupy 1,886,088 bytes. Audit: `outputs/residual-stage-gate-20261110/baseline060-audit.json`.
- 2026-11-10: Completed the first CPU pilot, seed 1, 800 ADMM steps and two-pass scale fitting. Inputs are the hash-verified Experiment 066 fixture and Experiment 067 first-stage fit. This is the retained Experiment 041 diagonal-Fisher objective; Experiment 060's calibration state is unavailable. The INT8 arm below is a residual-column correction on the common first stage, not a reproduction of Experiment 060's fitting recipe.

### Initial measured result

All candidates share the first stage and seven FP32 protected columns. The byte cap is 1,990,656 bytes, or 2.0 bits per original matrix parameter. Accounting includes a conservative diagnostic 4,096-byte header, packed sign bits, FP32 scales, INT32 indices, and 32-byte array alignment. Candidate data was written, read back, and reconstructed before measuring error. This diagnostic container is not a deployable GGUF format.

| Candidate | Stored bytes | BPW | Weighted NMSE |
|---|---:|---:|---:|
| First stage only | 1,234,464 | 1.240258 | 0.227502 |
| First stage + 651 INT8 residual columns | 1,989,664 | 1.999003 | 0.194923 |
| First stage + partial sign plane | 1,990,560 | 1.999904 | 0.114772 |
| First stage + rank-704 residual factor | 1,979,168 | 1.988458 | **0.105030** |

The residual factor fit took 125.6 seconds. At the common cap, its weighted error is approximately 46.1% lower than the INT8 residual-column arm and 8.5% lower than the sign-plane arm. Its 11,488 bytes of unused budget cannot fund the next 32-rank increment. This confirms a reconstruction advantage on one matrix and one seed; the functional and multi-block gates remain open.

Raw results and round-tripped candidate payloads: `outputs/residual-stage-gate-20261110/block12-seed1/`. Reproduce into a new directory:

```powershell
.\.venv\Scripts\python.exe tools/probe_residual_stage_budget.py --output outputs/residual-stage-gate-20261110/block12-seed1-repeat --iterations 800 --seed 1 --threads 12
```

Next implementation milestone: recover the full 060 compressed-model reconstruction set from GGUF, collect fresh fitting objectives for blocks 0 and 25, and adapt the paired dense-splice driver. The current overlay runner requires a resumable run directory and therefore cannot directly load 060. No functional evaluation slices have been consumed by this pilot.

Validation: six focused payload/reconstruction tests passed; the full suite passed with 1,522 tests and 49 deselected. Repository Ruff and mypy checks passed (217 source files checked by mypy). The pilot completed successfully; no experiment worker remains running.

### First functional selection result

The [block-12 functional gate](110-residual-stage-block12-functional-gate.md) completed on a fresh C4 48-by-512 selection slice. The residual factor improves paired KL by 0.53% versus the equal-budget INT8 pilot, with a 95% interval excluding zero, and improves NLL. It nevertheless regresses KL by 0.88% versus unchanged 060, also with an interval excluding zero. Keep the incumbent; this is a controlled representation gain, not an accepted 060 improvement. The sign-plane fallback did not improve KL significantly versus the INT8 pilot. Multi-block fits, matched tuning, and independent confirmation remain open. No production format changes are justified by this result.

The full GGUF audit also corrected the scale-storage assumption: 060's factor scales are BF16, whereas its norms and salient-column scales are F32. The initial pilot deliberately charged FP32 factor scales for every arm; its equal-cap comparisons remain valid as a conservative diagnostic format.

### Fresh multi-block confirmation result

The [multi-block gate](111-residual-stage-multiblock-gate.md) completed fresh online Fisher calibration and 800-step first/residual fits for blocks 0, 12, and 25 at seeds 0, 1, and 2. Representatives were fixed by reconstruction error before selection. All representations then received matched full-model KL scale tuning: 32 updates, identical fit tokens and optimizer, and 27,648 common first-stage scale parameters each.

The residual-factor composition passed the 48-sequence selection gate and then the independent 96-sequence confirmation gate without retraining. Confirmation KL improved 2.47% versus unchanged 060, difference -0.016969 with 95% interval [-0.024707, -0.010551]. NLL improved by 0.012317 nats/token, interval [-0.023061, -0.002957]. It also beat the matched tuned INT8 and sign controls on confirmation KL and NLL. The untuned residual composition improved incumbent KL by 1.81%, so the result is not entirely a scale-tuning gain.

This completes the initial equal-cap dense functional gate, with three changed projections and 279,240 additional diagnostic payload bytes versus their 060 sidecars. It does not complete the whole-model, downstream-task or runtime/export gates. Continue with the controlled distillation comparison; the missing historical pre-distillation state must be recovered or replaced by an explicitly defined common continuation checkpoint. Production format changes remain conditional on the broader winning-recipe gates.

### Controlled objective continuation result

The [objective continuation study](112-distillation-objective-continuation.md) preserved the 060 GGUF through a hard link and compared production top-64, full KL and full KL plus a 0.1 normalized hidden anchor. Each arm trained the same 901,056 fixed-sign factor scales for 256 updates on 64 fresh C4 sequences, with matched optimizer, schedule and BF16 scale persistence. Norms, outliers and embeddings remained frozen; this is a new post-060 scale-only continuation, not a historical rerun.

The preselected anchored checkpoint improved independent C4 confirmation KL by 18.66% and PTB perplexity by 8.87% versus the untreated reconstruction. Plain KL performed similarly; the anchor's incremental C4 benefit did not confirm. However, both variants scored 50.75% task mean versus untreated 50.92%, and their paired intervals failed the one-point noninferiority guard. WikiText perplexity worsened from 115.47 to about 121.2. Keep 060; no longer C4-only extension or production promotion is justified.

Before any longer-training extension, repeat the objective comparison using the original-style mixed calibration distribution and stronger held-out task coverage. The present screen demonstrates a likelihood benefit from full KL, not a capability improvement. Keep the residual candidate frozen and compare it against equally re-distilled 060 when the training objective and data mixture pass the broader gates. The plain-language explanation of partial sign planes and residual ranks is [Doc 113](113-partial-sign-planes-and-residual-ranks-eli5.md).

The retained 060 configuration confirms that its mixture is 50% UltraChat `train_sft` and 50% WikiText-2 `train`, with Gemma chat plus raw-text formatting; it is not the C4-only distribution used in this continuation screen. Generate new run-owned tokens from those pinned sources for the follow-up.
