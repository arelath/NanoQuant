# Controlled distillation objective continuation

Date: 2026-11-10

## Result and decision

Full-distribution distillation is a promising objective under this scale-only continuation: the preselected anchored checkpoint reduces fresh confirmation C4 KL by **18.66%** and separate-corpus PTB perplexity by **8.87%** versus untreated 060. Plain full KL performs similarly. The anchor's small incremental C4 benefit does not independently confirm.

The broader quality gate does **not** pass. Both full-KL variants average **50.75%** across six tasks versus **50.92%** for untreated 060; neither interval rules out a loss greater than the predeclared one-point margin. WikiText perplexity also worsens from **115.47 to about 121.2**. Keep 060 as the accepted baseline and **do not extend the present C4-only runs**. Before longer training, retest with the original mixed calibration distribution and the same controlled objective comparison. No production recipe, resident algorithm, or packed/runtime format was changed.

## Predeclared experiment

Continue step 2 of [Doc 109](109-residual-stage-quality-experiment-plan.md). The historical 060 pre-distillation checkpoint and its run store are unavailable. Preserve the retained 060 GGUF through a zero-copy hard link and bind the experiment to its SHA-256 `5e445cab6c93f03492ca0c9cea4ce7977fc780b9dbd75ad60578aef609dad02c`. This is a new matched continuation from post-distillation 060, not a rerun of its historical eight-epoch training. Keep the successful Doc 111 residual candidate frozen separately.

Compare three objectives from exactly this common state: production top-64 conditional cross entropy, full-vocabulary teacher KL, and full KL plus 0.1 times normalized hidden-state MSE. All use temperature 1. The top-64 arm calls the existing production loss; its teacher and student distributions are normalized only over the teacher-selected 64 tokens. It has no tail-mass term. Full KL includes every vocabulary entry. The hidden term averages `MSE(student,teacher)/mean(teacher^2)` over the 26 returned decoder states (embedding excluded; final state includes final norm), with denominator floor 1e-8. The coefficient is an unvalidated predeclared choice.

Train all pre/mid/post factor scales across the 130 shared/single 060 units, counting shared QKV scales once. Freeze signs, outlier values/scales/indices, norms, embedding and all other parameters. This is deliberately narrower than historical 060's scales/outliers/norms trainable surface, but identical across all three arms. Use FP32 master parameters and straight-through BF16 scale rounding in every forward; persist BF16 scales and evaluate only the persisted values. Student projections execute as BF16 dense weights reconstructed from fixed factors. No deployed format or resident algorithm changes are part of this diagnostic.

Use 64 new C4 validation sequences of 512 tokens, offsets 680–743, in sequential order for four cycles: **256 updates per arm**, batch one and all 511 next-token targets per sequence. Match production ParityAdamW defaults, learning rate 1e-5, zero weight decay, a 256-step cosine schedule and global gradient norm cap 1. No early stopping, sign updates, averaging, anchor-weight search, or selection-data fitting. Save FP32 parameters, optimizer, scheduler and progress at least every 16 updates. A two-step full-KL feasibility run uses the same declared 256-step schedule and resumes from its exact checkpoint if healthy.

Use a separate 48-by-512 selection window, offsets 744–791. Keep the untouched checkpoint as control and compare full KL with top-64, and anchored full KL with both. Report paired sequence-bootstrap 95% intervals (10,000 resamples, seed 0). This is a screening family with unadjusted exploratory comparisons. Require supported KL improvement and an NLL upper-difference bound at most +0.01 nats/token before extending an arm. A frozen selected candidate must pass on a separate 96-sequence confirmation window, offsets 792–887, before longer training. All consumed intervals are permanently retired before use.

An alternative objective must pass those KL/NLL bounds against both untreated 060 and matched top-64. Select the qualifying alternative with lowest mean selection KL. If both alternatives qualify but anchoring has no supported incremental KL benefit over plain full KL, prefer plain full KL. Freeze that choice before confirmation and transfer scoring; require confirmation against both controls. If neither alternative qualifies, retain the current objective and do not extend an alternative based on training losses.

The eventual objective decision also requires the standard downstream task suite and a genuinely separate out-of-domain perplexity corpus. The current standard evaluator provides WikiText-2 and six tasks; it has no standalone OOD perplexity protocol. A short C4-only screen cannot close those gates or establish whole-recipe parity. If every arm regresses the incumbent, stop extension and diagnose before spending longer training time.

## Checklist

- [x] Preserve and verify common checkpoint; record trainable inventory.
- [x] Measure two-step full-KL feasibility and memory; validate checkpoint resume.
- [x] Complete matched 256-update arms and storage-rounded checkpoints.
- [x] Complete fresh paired selection and independent confirmation if justified.
- [x] Run downstream and separate OOD gates before choosing a longer-training recipe.
- [x] Record results, validate tools, and commit stable implementation/evidence.

## Transfer protocol, fixed before scoring

Use 96 non-overlapping 512-token windows from the processed [PTB test file in the original LSTM reference repository](https://github.com/wojzaremba/lstm/blob/76870253cfca069477f06b7056af87f98490b6eb/data/ptb.test.txt), pinned to commit `76870253cfca069477f06b7056af87f98490b6eb`. Source SHA-256: `dd65dff31e70846b2a6030a87482edcd5d199130cdcfa1f3dccbb033728deee0`. Preserve its text/newlines and tokenize with the pinned Gemma tokenizer; each window's first token is context. The initial tokenizer BOS occurs only at the start of the stream. This is a separate corpus from the C4 continuation fit, not a claim that PTB was absent from model pretraining. Report Gemma BPE NLL/perplexity; it is not comparable with published word-vocabulary PTB perplexities. Retire the slice before scoring.

Score every frozen arm on PTB and the standard six-task replay (PIQA, ARC Easy/Challenge, HellaSwag, Winogrande, BoolQ; 200 examples each), plus the retained-style WikiText-2 64-by-128 protocol. Use exactly the same prepared inputs in every arm. Task pairs use the primary metric's per-example correctness, 10,000 bootstrap resamples within each task, equal task weighting, and seed 0. Require the PTB NLL upper difference bound at most +0.01 and the task-mean lower difference bound at least -1 percentage point before choosing an arm for longer training. These transfer comparisons are diagnostic screens on one corpus and the standard small task subset, not a broad capability result.

The common GGUF is preserved through a hard link without another 499 MB allocation. The audited surface is 901,056 trainable scales and 1,802,112 stored BF16 scale bytes; all arms replace existing scale values without adding deployment parameters. The full-KL two-step canary completed in 2.56 seconds of training and peaked at 4,544,455,168 CUDA bytes (about 4.23 GiB). Its optimizer, FP32 master parameters and 256-step scheduler state were saved for exact continuation. Four initial CPU objective/precision checks passed; the full suite passed with 1,533 tests and 49 deselected before adding the transfer-pair check.

The canary checkpoint resumed successfully in the matched run, retaining its first two loss records and continuing at step 3. The top-64 arm completed 256 updates in 415.7 seconds of training; 576,026 of its 901,056 stored BF16 scales changed (63.9%), and every persisted scale is finite. These are execution checks, not quality results. No selection sequence has been scored at this point.

The first PTB preparation stopped before any scoring or registry mutation because the registry validator rejected a 511-token payload interval for a non-WikiText dataset. The corrected preparation uses the existing generic contiguous-512-token convention in a separate directory; the failed attempt is preserved. The successful PTB token file has SHA-256 `cf4c5c083027b8cfba7598d225b3fa4e4e388f05c10ca36f607270401731c622`, and content hash `sha256:79c083761f33ec582b798520d740ba9b7faf61021e4b991dbbbbc4aecdaa8cc7`. This changes no registry schema or production behavior.

## Selection and frozen choice

All three arms completed exactly 256 updates and persisted 901,056 BF16 scales each. Their scale files each occupy 1,835,592 bytes including the safetensors header. Training took 415.7 seconds for top-64, 425.1 seconds for the resumed full-KL segment plus its 2.56-second canary, and 437.3 seconds for anchored KL. All use the same starting state, tokens, scale precision, optimizer, schedule and clipping. Top-64's cross entropy includes teacher conditional entropy, so its absolute training loss cannot be compared directly with the full-KL losses.

The 48-sequence selection screen completed on fresh data:

| Fixed endpoint | C4 NLL | Teacher KL |
|---|---:|---:|
| Untreated 060 reconstruction | 4.321528 | 0.693376 |
| Top-64 continuation | 4.352051 | 0.855661 |
| Full KL | **4.253006** | 0.570679 |
| Full KL plus hidden anchor | 4.254221 | **0.569725** |

Anchored KL minus untreated has KL difference -0.123651 (-17.83%), paired 95% interval [-0.136558, -0.111512], and NLL difference -0.067307, interval [-0.080367, -0.055441]. Against top-64, its KL difference is -0.285936 (-33.42%), interval [-0.310880, -0.262019], and NLL difference -0.097829, interval [-0.119372, -0.079397]. Plain full KL passes both controls too.

The anchor's incremental KL difference versus plain full KL is small but supported on selection: -0.000954 (-0.167%), interval [-0.001542, -0.000371]. Its NLL difference is +0.001216, interval [-0.000616, +0.002975], within the declared +0.01 margin. Under the predeclared rule, freeze **full KL plus the 0.1 hidden anchor** for primary confirmation. Its scale-file SHA-256 is `d7140d59da9a4fa2f89f5e94b0f23a5841456167cfea9b22722a6abd5b7bbc17`. No additional fitting or coefficient choice may use confirmation or transfer results.

Top-64 itself regresses incumbent KL by 23.41% and NLL by 0.030522, with both intervals excluding zero. This is a continuation result under this restricted scale-only protocol, not a claim that the historical 060 recipe has been rerun. Full KL's wider probability-mass coverage is a plausible explanation; this comparison does not separately measure probability migration into the vocabulary tail.

## Independent C4 confirmation

All 96 fresh sequences completed (49,056 next-token targets), with no additional training or checkpoint choice:

| Fixed endpoint | C4 NLL | Teacher KL |
|---|---:|---:|
| Untreated 060 reconstruction | 4.339042 | 0.703382 |
| Top-64 continuation | 4.349541 | 0.851210 |
| Full KL | **4.257121** | 0.572300 |
| Frozen anchored full KL | 4.258887 | **0.572133** |

The preselected anchored checkpoint passes against both controls. Versus untreated 060, its KL difference is -0.131249 (-18.66%), paired 95% interval [-0.139206, -0.123359], and NLL difference -0.080156, interval [-0.088459, -0.071751]. Versus top-64, KL changes by -0.279077 (-32.79%), interval [-0.294094, -0.264617], and NLL by -0.090655, interval [-0.102135, -0.079612].

The anchor's incremental KL benefit over plain full KL does **not** confirm: difference -0.000167 (-0.029%), interval [-0.000791, +0.000450]. Its NLL difference is +0.001766, interval [-0.000219, +0.003470]. The independently confirmed finding is the full-distribution objective's benefit under this continuation protocol; the extra anchor remains unproven as a prediction-quality improvement at this horizon. Its hidden-state effect is examined separately on PTB.

## Separate corpus transfer

All 96 PTB windows completed, covering 49,056 scored targets. Both full-KL variants pass the predeclared NLL guard against untreated 060 and top-64:

| Fixed endpoint | PTB BPE NLL | PTB BPE perplexity | Mean normalized hidden error |
|---|---:|---:|---:|
| Untreated 060 reconstruction | 5.775581 | 322.33 | 0.085969 |
| Top-64 continuation | 5.858155 | 350.08 | 0.090840 |
| Full KL | **5.677860** | **292.32** | 0.089951 |
| Frozen anchored full KL | 5.682662 | 293.73 | 0.088530 |

Anchored KL improves PTB NLL versus untreated by -0.092919, paired 95% interval [-0.107463, -0.077217], corresponding to 8.87% lower perplexity. Plain full KL improves NLL by -0.097721, interval [-0.112248, -0.082099], corresponding to 9.31% lower perplexity. Top-64 regresses NLL by +0.082574, interval [+0.075462, +0.089652]. The probability-objective result therefore transfers beyond the C4 fitting corpus in this test.

Anchoring lowers mean PTB hidden error by about 1.58% relative to plain KL, but both remain above the untreated checkpoint's hidden error. A better hidden-error proxy does not establish better prediction quality: plain KL has the slightly lower PTB NLL, while their C4 KL difference did not confirm. This coefficient and short horizon do not demonstrate a general benefit from the anchor.

All arm comparisons use the same CUDA FP32 factor reconstruction followed by BF16 dense-weight materialization. A numerical audit on the first PTB sequence compared the untouched CPU GGUF restoration with that canonical path: KL changed by +0.001290 and NLL by -0.000551. This audit records an arithmetic-path difference, rather than claiming bit-identical native GGUF execution. The task and NLL comparisons are against the untreated canonical reconstruction, with the same path in every arm.

## Standard quality replay and stopping decision

All four variants completed the same six tasks at 200 examples per task and the same WikiText-2 64-by-128 protocol. These are the existing standard benchmark partitions, not newly reserved capability-test examples.

| Task primary accuracy | Untreated 060 | Top-64 | Full KL | Anchored full KL |
|---|---:|---:|---:|---:|
| PIQA | 66.0% | 67.0% | 66.0% | 65.0% |
| ARC Easy | 43.5% | 44.0% | 45.5% | 46.0% |
| ARC Challenge | 26.0% | 23.5% | 22.5% | 22.5% |
| HellaSwag | 44.0% | 45.0% | 44.0% | 44.0% |
| Winogrande | 59.5% | 58.0% | 58.0% | 58.5% |
| BoolQ | 66.5% | 68.5% | 68.5% | 68.5% |
| **Equal task mean** | **50.9167%** | **51.0000%** | **50.7500%** | **50.7500%** |
| WikiText-2 perplexity | **115.4708** | 122.8423 | 121.1250 | 121.1673 |

Anchored KL minus untreated has task-mean difference -0.1667 percentage points, paired 95% interval [-1.5833, +1.2500]. Plain KL has the same point difference with interval [-1.5833, +1.1667]. Against matched top-64, both full-KL variants change task mean by -0.2500 points, with intervals [-1.5000, +1.0000]. These results do not establish task regression, but they **fail** the declared noninferiority criterion because the lower endpoints are below -1 point. The task sample is too small to rule out that loss.

Anchored and plain KL worsen WikiText perplexity by about 4.93% and 4.90%, respectively. C4-only fitting versus 060's historical mixed calibration is a plausible contributor, not a demonstrated cause. This tradeoff and the failed task guard block promotion even though C4 and PTB likelihood improve. Better logit KL is insufficient evidence of better task capability.

Preserve all three short-run checkpoints. Do not start a 2,048-step extension from this screen. Next, compare top-64/full-KL/anchored-KL using the same original-style mixed calibration distribution, first at this short horizon, and increase held-out task coverage if the likelihood gains survive. Keep the training surface fixed for that diagnostic; norm/outlier tuning, sign movement and averaging remain separate ablations. Retest residual-stage candidates against equally re-distilled 060 rather than adding the two experiments' percentage gains together.

The retained 060 resolved configuration identifies that original distribution precisely: **50% UltraChat `train_sft` and 50% WikiText-2 `train`**, using `gemma-chat-plus-raw-text-v1` formatting, shuffling, and selection seed 0. UltraChat revision is `8049631c405ae6576f93f445c6b8166f76f5505a`; WikiText revision is `b08601e04326c79dfdd32d625aee71d232d685c3`. A follow-up can reproduce this recipe with newly generated run-owned calibration tokens; the missing historical token file cannot be claimed as recovered.

## Evidence and validation

Run root: `outputs/objective-continuation-20261110/run/`. `common060.gguf` is the preserved hard link; `protocol.json` fixes identities and training settings; `parameter-inventory.json` records the shared scale surface; each arm has resumable optimizer/master-parameter checkpoints, progress, a BF16 `scales.safetensors`, and a hash receipt. `frozen-endpoints.json` binds all three completed endpoints. The exact executed driver and helper are archived and match the training protocol hashes.

`selection/` and `confirmation/` preserve evaluation identities, all sequence results and paired statistics. `transfer/` preserves the pinned PTB protocol, sequence metrics and hidden error, the canonicalization audit, all standard task examples/scores, and paired task intervals. Compact copies of the protocols, sequence results, training summary and task-correctness audit are committed under `evidence/objective-continuation-20261110/`. Large checkpoints and the corpus file remain in outputs.

All four new slices were permanently retired before use: C4 fit 680–743, selection 744–791, confirmation 792–887, and the independent PTB 96-window transfer slice. Existing unrelated working registry changes are preserved. The registry validates with 28 retired slices and no reserved slices.

The full CPU suite passed with **1,534 tests and 49 deselected**. Ruff passed repository-wide; mypy passed all 217 source files. New tests cover BF16 value/FP32-gradient semantics, fixed-sign shared-unit gradients and protected columns, normalized hidden anchoring, chunked top-64 gradients versus the production loss, and paired task identities/equal task weighting. The real two-step checkpoint resumed successfully without repeating its first two updates. Peak training CUDA allocation was 4,544,455,168 bytes (4.23 GiB); peak standard-evaluation allocation was 6,895,435,776 bytes (6.42 GiB), excluding other applications. All experiment processes exited successfully and the device lease was released.

This remains a short, one-seed, scale-only continuation on Gemma 3 1B, using a GGUF-derived dense execution path. It does not validate the historical full trainable surface, longer horizons, native deployment, other bit budgets, or larger models. The task intervals describe sampling of the fixed small benchmark partitions and do not cover training-seed or prompt variation.
