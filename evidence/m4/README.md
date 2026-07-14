# Milestone 4 Gemma 3 1B resident diagnostic

The rewrite completed a 26-block, 182-layer resident quantization of the locally pinned
`google/gemma-3-1b-it` revision `dcc83ea841ab6100d6b47a070329e1ba4cf78752`.
The source model stayed resident on the RTX 4000 Ada Laptop GPU while every accepted layer and
block was committed through the new artifact store and progress journal.

This is a pipeline and allocation diagnostic, not an approved quality-parity run. It uses one
ADMM outer/inner iteration and deliberately disables scale fitting and both tuning loops. Those
differences make its reconstruction and quality values unsuitable for promotion.

## Rewrite result

| Measure | Result |
|---|---:|
| Blocks / quantized layers | 26 / 182 |
| Requested / effective logical BPW | 1.0 / 0.999832 |
| Rank minimum / mean / maximum | 128 / 738.99 / 1152 |
| ADMM attempt wall time | 46.25 seconds |
| End-to-end wall time | 143.57 seconds |
| Peak allocated GPU memory | 2,655,558,144 bytes |
| Peak process working set | 2,754,904,064 bytes |
| Auditable run artifacts | 12,393,832,745 bytes |
| Fixture base / compressed NLL | 21.5141 / 24.5126 |
| Fixture logit MSE | 58.4299 |
| Fixture argmax agreement | 0.0 |
| Mean objective-weighted normalized layer error | 0.999554 |
| Median final-block loss change vs block entry | +1.19% |

The immutable report is
`sha256-1179f8accd5df8c1e3741517f35f48a6cf685cc941b86c3c509881db9e16f4fa` under
`gemma-3-1b-it-dcc83ea8-admm1-sensitivity/artifacts`.

## Real interruption, resume, and replay

A separate run in `gemma-3-1b-it-interrupt-resume` was forcibly interrupted after ten durable
layer commits. Resume discovered and reused 11 commits (the ten layers plus the completed first
block), restored the partial second block, and completed all 182 layers. It exactly matched the
uninterrupted control on effective BPW (`0.9998319656`), rank distribution, compressed fixture NLL
(`24.5125637`), and logit MSE (`58.4299240`). The resumed segment took 134.44 seconds.

The committed block-1 `self_attn.k_proj` layer was then captured as canonical fixture
`sha256-eb4d7d1b40e5a87688a7bf3c5e9250882a5cea620dd028440b9294275ba84e5c` and replayed on CUDA
from pinned source/objective inputs. Replay completed in 1.84 seconds, matched the accepted
reconstruction within `4.66e-09` maximum absolute difference, and reproduced weighted normalized
error `0.9802689`.

## Available legacy comparison

The closest retained baseline is Experiment 018 (`phase1-no-hessian`). It used 182 accepted
layers plus 15 retry attempts, residual BF16 outliers, 800 ADMM iterations, scale fitting,
non-factorized/factorized tuning, post-block refits, and model-level KD.

| Measure | Rewrite diagnostic | Legacy Experiment 018 |
|---|---:|---:|
| Accepted layers | 182 | 182 |
| Rank minimum / mean / maximum | 128 / 738.99 / 1152 | 128 / 580.92 / 1056 |
| Mean weighted normalized layer error | 0.999554 | 0.284159 |
| ADMM attempt time | 46.25 s | 952.16 s |
| Compression time | 143.57 s | 9,564 s |
| GPU memory evidence | 2.47 GiB peak allocated | 4.71 GiB reported at compression boundary; no retained peak |
| Host memory evidence | 2.57 GiB peak working set | not retained |
| Stored output | 12.39 GB auditable run directory | 390,375,090-byte executable `.pt` checkpoint |
| Quality | eight-token diagnostic only | WikiText-2 limited PPL 384.954 and retained task suite |

The rewrite is faster only because the algorithm settings are intentionally reduced. Its layer
error and fixture quality confirm that one-step ADMM without scale/tuning is not a parity
candidate. A protocol-matched quality comparison and retained legacy peak-host/peak-GPU data are
still required before **M4.26** can be checked.

## Factor/outlier/scale isolation run

The follow-up `gemma-3-1b-it-parity-factor-scale` run used the legacy layer order, 800×5 ADMM,
0.1% residual BF16 outliers, and two-pass least-squares scale fitting. It deliberately retained
the tiny input-only calibration fixture and disabled tuning, isolating the effect of those three
stages. It completed 26 blocks and 182 layers in 1,439.23 seconds, reused the intentionally
interrupted first layer, reached 1.024438 effective BPW (outliers are free from the 1.0-BPW
factor budget), and peaked at 2,623,874,048 allocated GPU bytes. The run contains 12,406,595,291
bytes of validated artifacts and report
`sha256-793750661eed73613716bb30f15742c239eab3dfcaabb3fcfbac5b6241ec3c4c`.

The exact retained WikiText-2 protocol uses 64 windows of 128 tokens, Gemma BOS handling, causal
shift, and 8,128 scored targets. Its token hash is
`sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`.

| Model | NLL | Perplexity |
|---|---:|---:|
| Pinned BF16 base | 4.572604 | 96.795812 |
| One-step/input-only diagnostic | 24.864677 | 62,891,503,910.87 |
| ADMM-800 + outlier + scale isolation | 8.650405 | 5,712.460528 |
| Legacy Experiment 018 | — | 384.954437 |

This establishes that complete factorization, outlier selection, and scale fitting recover many
orders of magnitude of quality, but do not close parity without full output-Fisher calibration
and tuning. The raw evaluator result is
`gemma-3-1b-it-parity-factor-scale/wikitext2-limited.json`.

## Full-Fisher streamed factor/outlier/scale run

The canonical `gemma-full-fisher-quantization` run uses all 256 pinned Experiment 018 calibration
sequences at length 2,048, online causal-loss Fisher statistics with 0.6 shrinkage, legacy
sensitivity allocation, 800×5 ADMM, residual BF16 outliers, and two-pass scale fitting. Tuning and
model KD remain disabled so this run isolates calibration parity.

| Measure | Rewrite full-Fisher result | Legacy Experiment 018 |
|---|---:|---:|
| Blocks / layers | 26 / 182 | 26 / 182 |
| Effective BPW | 0.998674 | approximately 1.0 target |
| Rank min / mean / max | 128 / 580.57 / 1088 | 128 / 580.92 / 1056 |
| Factorization attempts / time | 183 / 1,063.42 s | 197 / 952.16 s |
| Outlier-stage attempts / time | 198 / 154.78 s | not separately retained |
| Scale-fit attempts / time | 183 / 13.54 s | not separately retained |
| Peak factor-only GPU allocation | 505,351,168 bytes | not retained by stage |
| Peak block-assembly GPU allocation | 2,341,744,128 bytes | 4.71 GiB reported boundary |
| Frozen-evaluator GPU allocation | 2,297,270,272 bytes | not retained |
| Auditable run artifacts | 75,007,326,877 bytes | 390,375,090-byte executable checkpoint |
| WikiText-2 limited PPL | 3,334.254567 | 384.954437 |

The exact retained evaluator again scores 8,128 targets with token hash
`sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`; the pinned BF16
base remains 96.795812 PPL. Full Fisher improves the untuned rewrite from 5,712.46 to 3,334.25 PPL
(41.6%), while the remaining 8.66× gap to legacy identifies tuning/KD as the next parity stage.

The immutable run report is
`sha256-5f070d85dd69bfc7d1bfe5c493373e139f7018ca93e83879a38a6ca055f43fba` and the raw quality
result is `gemma-full-fisher-quantization/wikitext2-limited.json`. Block loss snapshots were
deferred during bounded assembly, so **M4.26** and **M4.GATE** remain open pending the tuned run
and non-deferred aligned block comparison.

## Resident-memory correction and real canary

The first full-Fisher attempts exposed rewrite-specific allocation bugs, not an inherent limit of
the 1B model: full-vocabulary float logits, full calibration-set float MSE temporaries, duplicated
concatenation outputs, CUDA-resident block-boundary activations, retained replaced source blocks,
and three accidentally orphaned concurrent launcher children. Stopping the duplicate processes
dropped device use from 11,349 MiB to 227 MiB immediately.

The corrected resident path computes exact causal gradients in vocabulary/token chunks, streams
MSE, preallocates outputs, stores block-boundary activations on CPU, moves only microbatches to
CUDA, and releases source blocks progressively. An exclusive cross-process CUDA lease now rejects
concurrent resident runs.

The real `gemma-single-process-memory-canary` used the pinned model, full 2,048-token sequences,
online output-Fisher calibration with 0.6 shrinkage, sensitivity planning, and a real first-layer
commit. With eight calibration sequences it peaked at **2,590,282,240 bytes (2.41 GiB)** and
returned to 234 MiB after the intentional interruption. Its durable layer commit is sequence 1 in
the canary journal, proving the measurement includes loading, calibration, planning, factorization,
and artifact commit rather than model loading alone.

The final 256-sequence calibration was then executed as exact durable online-accumulator slices
under the managed shell's short process limit. The checkpoint preserves each layer's cumulative
input/output totals, robust global maxima, hook-update counts, and the separate logical processed
sample count needed because gradient-checkpoint recomputation fires input hooks more than once.
Uninterrupted-versus-resumed tests are bit-exact. All 256 sequences completed with per-slice peaks
of 2,696,183,296 GPU bytes and approximately 2.83 GB host memory. The durable state is in
`gemma-full-fisher-state`.

Materialization with 0.6 shrinkage produced calibration
`sha256-88fc3ce6f52d721fa8aa68d4c49bca2b6bf0fb1db7ada129161ec10854042e12`, objectives
`sha256-d76e00eff0f606ed36ff92fece57e85dfde63c8779a2af2f77ddb582cb8d661b`, and plan
`sha256-9887c339d12369b9a4e84476f42915b62695bfe733c3071519d20e0aa79e8793` in
`gemma-full-fisher-quantization`. The first block's ranks are
`gate=1088, up=1088, down=864, v=160, o=512, q=448, k=128`; five of seven exactly match the
retained legacy plan, while gate/up are one 32-rank step higher.

The first two resident ADMM-800 layer commits completed at approximately 2.59 GB peak. Gate
has weighted/raw normalized reconstruction error `0.190436/0.275998` (legacy:
`0.198450/0.288690` at rank 1056), and up has `0.248985/0.288218`. The remaining layers and final
WikiText evaluation are intentionally still in progress; **M4.26/M4.GATE remain open**.

Weight-only factor slices now bypass Transformers model/prefix/teacher activation setup and use
only the cached plan, one source weight, the outlier/ADMM/scale stages, and an atomic layer commit.
This reduced observed peak allocation to 23–461 MB and typical elapsed time to 3–14 seconds per
layer. The durable full-Fisher journal ultimately reached all **182 layer commits and 26 block
commits**; shell deadline termination between layers did not invalidate completed artifacts.

## Full legacy-tuning canary

The resident pipeline now executes the Experiment 018 layer sequence rather than stopping after
factor/scale fitting: before each layer it tunes the remaining full-precision block with the
`8,4,3,2,2,2,2` epoch schedule, tunes the newly factorized layer for eight epochs, and jointly
refits finalized scales/outliers/biases for two epochs after the block. Deterministic replay of the
non-factorized steps makes interruption after any immutable layer commit resume-equivalent.

A bounded block-0 canary used all 256 pinned 2,048-token Fisher sequences, the retained sensitivity
plan, 800×5 ADMM, residual BF16 outliers, two-pass least-squares scale fitting, and the complete
legacy tuning schedule. It committed all seven layers and the refitted block in 862.73 seconds,
with a recorded peak allocation of 5,095,590,400 bytes.

| Measure | Untuned full-Fisher block 0 | Fully tuned block 0 |
|---|---:|---:|
| Final block activation MSE | 6.865820 | 3.725633 |
| Improvement from tuning | — | 45.74% |
| Loss immediately before joint refit | — | 3.740169 |
| Loss after joint refit | — | 3.725633 |

Every factorized layer restored its best epoch. Relative tuning recovery ranged from 1.05% for
`q_proj` to 48.11% for `gate_proj`; the three MLP projections recovered 48.11%, 16.55%, and
21.53%, respectively. The immutable block result is
`sha256-1b2c2e76c3c2d372ad227ca9de260ad8a7cdba495f444081512b05153d5c9185`.

The detached, block-bounded resume run in `gemma-full-fisher-tuned-canary` is advancing the same
recipe through the remaining 25 blocks. The block boundary is intentional: each child owns one
CUDA lease and leaves a validated journal cursor before the next child resumes.

## Exact-runtime v17 tuned canary

The `gemma-eager-fullbatch-canary` v17 identity
`sha256:f9cff7222473a45a5eadbd84f0c1b970eee0aacbfb52d4894faa0745f4b53c0f`
adds two parity corrections: block commits retain named non-factorized norm parameters, and every
rank retry reruns residual-outlier selection, ADMM, scale fitting, and the full reconstruction
metric. Blocks 0 and 1 completed with batch size 8 at 6,120,173,056 and 6,106,409,472 peak CUDA
bytes. Their final pre-KD block losses are 1.3794084 and 3.5831392. Block 1 `k_proj` reran the
complete attempt at rank 160 after its rank-128 retry gate failed. Both blocks persist six named
Gemma norm parameters, so frozen loading and completed-block restoration no longer revert tuned
auxiliary state.

The retained Experiment 018 log reports lower block losses (1.1624 and 3.1222), but its exact
initial factor/calibration tensors were not retained. A direct frozen-source diagnostic used the
current official implementation, the recaptured gate factors, all 256 pinned sequences, the exact
output importance, PyTorch 2.12.1+cu130, and Transformers 4.51.3. Official and rewrite tuning both
started at 0.8200597763 and finished at 0.3709969819 after eight epochs. This proves the rewrite
tuner is numerically equivalent for the reproducible state; the historical 0.31145 gate result
depends on an intermediate legacy state that is present only as aggregate log evidence. Future
quality comparisons must distinguish the reproducible frozen-source oracle from that historical
checkpoint baseline.

## Exact-runtime v17 parity canary

The current `gemma-eager-fullbatch-canary` identity
`sha256:f9cff7222473a45a5eadbd84f0c1b970eee0aacbfb52d4894faa0745f4b53c0f` adds two correctness
properties missing from the earlier canary: block commits persist all tuned norm/auxiliary
parameters by name, and every bumped-rank attempt reruns residual outlier selection, ADMM, scale
fitting, and the full reconstruction metric before the retry decision. Blocks 0 and 1 completed
at 6,120,173,056 and 6,106,409,472 peak allocated bytes respectively, so the exact 256-by-2,048,
batch-8 workload no longer exhausts the 12 GB device.

Block 0 committed as
`sha256-b84724a7db36b11f967559c12a2204b9372a2e459ba7c4b373bde884e69ae33b` in 546.75 seconds with
final loss `1.37940836`. Block 1 committed as
`sha256-a0d5b232f4c8252e55faaa8295cef32af200d2f22c161d111930fcc3171c0e37` in 576.34 seconds with
final loss `3.58313918`; its `k_proj` correctly retried residual selection at ranks 128 and 160
and accepted rank 160.

A direct diagnostic invoked the frozen official `tune_fact` implementation on the same block-0
gate factors, 256 inputs/targets, Fisher importance, eager Gemma metadata, batch size 8, and eight
epochs. Official and rewrite results were bit-identical: initial loss `0.8200597763061523` and
final loss `0.37099698185920715`. The lower historical Experiment 018 gate result (`0.31145`)
therefore came from factor/calibration state that was neither committed nor retained as tensors;
it cannot serve as a bitwise tuning oracle. The reproducible official source plus retained inputs
is the executable tuning oracle, while the historical checkpoint/evaluation remains the required
end-to-end quality comparison.

## Complete v17 quality and legacy-checkpoint comparison

The v17 identity completed all 26 blocks and 182 layers. Its report is
`sha256-94293b21cf00f14e9fe208fe7c7dace3382a096f8d27922b4b42c6ce64cb1f9e`:
effective BPW `0.9963181`, 15,284.04 seconds total wall time, 6,107,513,856 peak CUDA bytes,
and 6,908,862,464 peak host bytes. The exact retained WikiText-2 protocol uses 64 windows of
128 tokens, 8,128 scored targets, dataset fingerprint `a29ea8a573703a32`, and token hash
`sha256:ef19dc950344a837a1fd6e087c451ed9b26234408e85d0b0e3da4f6c7045ff27`.

| State | WikiText-2 limited PPL | Result |
|---|---:|---|
| BF16 source | 96.795812 | retained baseline |
| Legacy Experiment 018 after KD | 384.954437 | required quality baseline |
| Legacy Experiment 018 checkpoint through rewrite backend | 383.938808 | 0.26% below retained legacy evaluation |
| v17 immutable block commits before KD | 432.078117 | 12.24% above legacy |
| v17 after PyTorch-default AdamW KD | 444.395669 | 2.85% worse than pre-KD |
| v17 after legacy-compatible Optimi/Kahan KD | 459.408045 | 6.33% worse than pre-KD |

The legacy-compatible replacement KD artifact is
`sha256-dd81a312c30fbbb37c5170d8756a9a7693109c74f624433e7934674e7f503224`.
It completed 2,048 steps, reduced cached top-k loss from `2.3925863` to `2.1383089`, reused the
411,041,792-byte teacher cache, took 4,923.63 seconds, and peaked at 4,079,121,920 CUDA bytes and
4,883,726,336 host bytes. Source inspection found that the earlier rewrite used PyTorch AdamW
defaults instead of legacy Optimi AdamW's debiased-beta recurrence, `(0.9, 0.99)` betas,
`1e-6` epsilon, and BF16 Kahan compensation. Correcting that behavior improved parity of the
training implementation but did not improve held-out quality from this pre-KD state.

The retained 390 MB Experiment 018 checkpoint provides a stronger structural oracle than the
aggregate logs. Direct comparison against immutable v17 pre-KD state found different final ranks
in 30 of 182 layers and different residual-outlier column sets in 112 of 182 layers. Total rank
sum differs by only 128 (`105,728` legacy versus `105,856` rewrite), so this is rank placement,
not a broad BPW error. Same-rank finalized binary tensors have approximately 50% element agreement,
showing that their factor trajectories are unrelated. Blocks 2 through 10 generally recover
2 to 7 percentage points less of their block-entry loss than legacy, while most later blocks are
close. The next parity experiment must replay the retained checkpoint's exact structural choices
or reconstruct the missing historical calibration/factor state; further KD on the current v17
state is not supported by the measured quality result.

The read-only legacy-checkpoint adapter independently clears the rewrite loader/execution/evaluator
boundary. It unpacks the retained LSB-first binary tensors, applies the legacy row-wise int8
embedding, outliers, scales, and auxiliary parameters to the pinned base architecture, and executes
the normal rewrite `FactorizedReferenceLinear` path. The resulting exact-protocol artifact
`wikitext2-legacy018-rewrite-backend.json` reports PPL `383.938808` at 4,636,544,000 peak CUDA
bytes. This is within 0.26% of the retained official `384.954437` result. Consequently, importing
the correct frozen state closes quality parity; generating that state from the reconstructed
calibration and plan remains the open gate.

### Residual-probe parity correction

Source comparison found that the rewrite residual-outlier probe hard-coded three inner SVID
iterations even when the resident ADMM configuration requested five. Experiment 018 passes its
configured `admm_inner_iters=5` into both the 80-iteration probe and full factorization. On the
pinned block-0 `up_proj` source weight and reconstructed objective, current official and rewrite
ADMM reconstructions are bit-identical when both use five; both select legacy checkpoint columns
`[768, 941]`. The old three-iteration rewrite probe selected `[367, 768]`. The outlier stage now
uses the configured inner iteration count and resident algorithm version 19 prevents adoption of
older commits.

A complete v19 block-0 canary committed as
`sha256-588d6838bf393e393739a0485c8736195dee819e5bc958d57dde353e4625f0ce`.
It finished in 1,548.39 seconds at 6,196,452,864 peak allocated CUDA bytes. Final block loss was
`1.37848997`, only 0.07% below v17's `1.37940836` and still above retained legacy `1.1624`.
The gate trajectory remained `0.82005978 -> 0.37099698`; its selected columns were already the
same in v17. The retained legacy gate objective reports target weighted norm squared `198.86096`,
while the reconstructed rewrite scale-fit objective reports `199.08095`. Thus the probe bug is a
real corrected boundary mismatch, but the first-layer quality gap still depends on the slightly
different historical objective/factor state that was not retained with Experiment 018.

### Corrected CCE state and tuning-protocol diagnostics

The obsolete `gemma-full-fisher-state` predates the CCE calibration correction and is not a valid
Experiment 018 objective source: its output-importance vectors differ from the retained legacy
statistics by 29.17% mean relative error. The corrected `gemma-cce-fisher-state` reduces that to
0.52% mean relative error, with essentially exact input-importance vectors. Regenerating the
preprocessing artifacts from that state produced calibration
`sha256-b59a599ffd6e710943fd7b6d63052bce48954fc449de8b25474f1d5abdcac3fe`, objectives
`sha256-ed0068d5a7ecb00beab2e51171a127d4f3e20c6d239c564dc8e3624c33caa888`, and plan
`sha256-8a1e01952dc64894efbc4c7c6f8536365df3c2142c90546e0731e0bff1de18d0`.

The complete corrected-CCE block-0 canary committed as
`sha256-ba5232284e192898c32c06a9391f25aaeb03f20dbaf29dcf0312708f709d4c9e` in 2,130.98 seconds.
Its final loss was `1.36599243`, 0.91% below v19 but still 17.52% above the retained legacy
`1.1624`. Peak allocated CUDA memory was 3,841,763,840 bytes. This supersedes the obsolete
full-Fisher objective as the current reconstructed-calibration comparison.

A second preprocessing set materialized the retained `legacy018-fisher-stats` vectors as closely
as their stored precision permits: calibration
`sha256-fd331450e60ca549a6be5d324c64bfab89efa2f7696fb8a77cf47fbefd956cdb`, objectives
`sha256-5e22c4768b37a6b1e12483d08c7251fd892c45becf5236ab47acc315e6b22c25`, and plan
`sha256-62c5047dc57f9fcf1ce1896b4bedbc7a07914a964edf84bbe6e74a1be274c18c`.
Gate-only canaries reached `0.374311` with tuning microbatches of one and `0.373442` with the
historical unsplit batch of eight, versus historical Experiment 018 `0.31145`. The full-batch
artifact is `sha256-577d02ef55724a1ca069588a9212ca551b595081281e8b8e7fafe01614169979` and peaked at
4,111,409,664 allocated CUDA bytes. Microbatch accumulation is therefore not the remaining
quality gap.

The Experiment 018 calibration source reduces each 2,048-token hook tensor as four ordered
512-token partial sums. Schema-2/algorithm-2 checkpoint
`gemma-legacy-chunked-fisher-state` now reproduces that path and completed all 256 pinned samples.
All **182 input-importance vectors are bitwise equal** to the retained statistics. The output
vectors have **0.543882%** layer-mean relative error, versus **0.490226%** for the prior unchunked
CCE state. Two fresh one-sample runs with identical inputs produced bitwise-equal inputs but
**0.271838%** layer-mean output error between each other. The pinned CCE Triton forward/backward
kernels use lock-protected multi-CTA accumulation, so their floating-point accumulation order is
not deterministic; the retained Experiment 018 output vectors are one valid realization rather
than a reproducible bitwise oracle.

The code-faithful state produced calibration
`sha256-590a425253af2503a4237b079f9afc3c950608702b500c210cb35f3a85768ac0`, objectives
`sha256-49525a88f75e991e77f43dbab77cf24d574ed658a4a5acba8be70fccd19d2f7d`, and plan
`sha256-7643fe17e527e68bb127fec1cce4dccb75fa0b51d64b799bac404e778ce9c825`.
The plan is rank-for-rank identical to the prior corrected-CCE plan: 38 legacy rank mismatches,
rank sum 105,376 versus legacy 105,728, and matching per-layer outlier counts. A bounded untuned
block-0 factor canary completed all seven layers in 41 seconds with at most 329,910,784 allocated
CUDA bytes. Only `gate_proj` matched the retained outlier set; six layers differed, `o_proj`
remained rank 544 versus legacy 512, and same-rank left/right sign agreement averaged
0.500387/0.499952. In particular, `up_proj` selected `[367, 768]` instead of legacy `[768, 941]`.
This fails the structural gate, so no expensive tuned/full run is justified from this state.
The detailed comparisons are `gemma-legacy-chunked-fisher-comparison.json`,
`gemma-legacy-chunked-plan-legacy-comparison.json`, and
`gemma-legacy-chunked-parity-canary/block0-factor-only-legacy-comparison.json`.

An exact-retained diagnostic then bypassed CCE variance entirely. The validated replay produced
calibration `sha256-ce8e232460d76986e4fab74a669e3970dc9ca1458701dc67e1f8f9e0ea2a5c60`, objectives
`sha256-c48935f94d9e1db0c8fe8f0105affd4b1cc682bf3708047503c665efb6b74a2f`, and plan
`sha256-0e45d3b6a3af0d64a8199885f2be215fc1afc77488051b260b143627e4abb5fb`. Current official
allocation and the rewrite plan match on all 182 layers with these vectors. Experiment 018's
logged initial ranks differ in 32 layers because the historical numerical environment crossed
discrete allocation thresholds. The exact-objective block-0 tune committed as
`sha256-9ce665d15e271fbdac3b6a6e3be2fb0b0a5184497592cf612dbbfdeb08c13a86`: 476.06 block seconds,
6,194,081,280 peak allocated CUDA bytes, and final loss `1.3784899712`. This is effectively
identical to v19 and remains 18.59% above legacy `1.1624`; it is not extended to later blocks.
An eight-seed gate factorization screen (`gemma-retained-fisher-gate-seed-sweep.json`) ranged only
from `0.1971011` to `0.1972472` weighted normalized error. The best start improved seed 0 by
0.026% relative. Its bounded eight-epoch tune then regressed to `0.377420`, versus seed 0
`0.370981`, so multi-start ADMM is not promoted to full-model testing.

Direct CUDA diagnostics then ruled out the other isolated tuning boundaries:

- official and rewrite ADMM produced exactly equal latent factors, binary factors, and scales
  from the same post-probe RNG state;
- CUDA default-generator and private-generator states were exactly equal after the 80-iteration
  residual probe;
- the legacy scale fitter and rewrite scale fitter reached the same weighted error
  (`51.296413`), with only sub-BF16 floating-point differences in the fitted scale tensors;
- restoring the legacy column-major stride of the right latent factor left the gate tuning
  metrics bit-identical (`0.858794 -> 0.373442`) and is rejected because it added no quality;
- the forced historical single-tensor Optimi AdamW update and `ParityAdamW` produced bit-identical
  parameters, moments, and Kahan compensation over 100 BF16 CUDA steps.

These results narrow the historical gap to the unretained factor/objective trajectory already
visible in the checkpoint comparison. A 0.11% target-objective difference is enough to send ADMM
to unrelated binary signs, so the aggregate Experiment 018 log is not a bitwise factor oracle.
The retained checkpoint remains authoritative for end-to-end quality and runtime loading, while
new quality work must be judged by the exact WikiText-2 protocol rather than by attempts to
reconstruct missing transient state.

### Corrected global KD result and cache-sampling mismatch

The zero-weight-decay, BF16/Kahan global KD run completed all eight epochs and activated artifact
`sha256-078aeb721c8257347297eb9d5d477da8899f5530addad4dcb7e9d7479b32774a`. It completed 2,048
steps, reduced cached top-k loss from `2.39330782` to `2.14116740`, cached 411,041,792 teacher-target
bytes, and peaked at 2,702,332,928 allocated CUDA bytes. The final bounded epoch took 165.90 seconds.
Exact retained WikiText-2 evaluation in `wikitext2-v19-corrected-kd-factorized-exact.json` produced
PPL `462.207656`, 6.97% worse than immutable pre-KD `432.078117` and 20.39% worse than the rewrite
loader's legacy-checkpoint result `383.938808`. This artifact is evidence, not an accepted parity
improvement.

Source comparison after that result found a protocol-level RNG mismatch in the teacher cache. The
retained rewrite cache starts epoch 1 with samples `[172, 55, 225, 105, ...]`; Experiment 018's seeded
Python `random.shuffle` order is `[1, 88, 132, 233, ...]`. The legacy loop also selects its 512 token
positions with the CUDA RNG and carries both RNG streams across epochs, while the rewrite used
independent CPU Torch generators. The sampler now reproduces the persistent Python/device RNG streams
and versions that behavior in the protocol identity so the incompatible cache/checkpoints cannot be
resumed.

The replacement run activated artifact
`sha256-ca02c12e1c1fc339e0a6844af2beaadf8f6d80526352efc4a8f9d3b8437d3305`. It completed 2,048
steps with losses `[2.40027411, 2.24463855, 2.20950463, 2.18232836, 2.16110932, 2.14792685,
2.14395084, 2.14039628]`, retained a 411,041,792-byte cache, selected 885 parameters, and peaked at
2,702,332,928 allocated CUDA bytes. Exact serial evaluation is retained as
`wikitext2-v19-legacy-sampling-kd-factorized-exact.json` and reports PPL `454.431449`. This is 1.54%
better than the incompatible-cache serial result `461.544627`, but 4.97% worse than immutable pre-KD
`432.930572` and 18.36% worse than the legacy-checkpoint rewrite result `383.938808`. M4.29 is complete
as an experiment, but the quality gate remains failed; the retained checkpoint comparison identifies
the upstream factor/outlier trajectory as the next boundary rather than further KD optimization.

The retained Experiment 018 console log independently supports that boundary decision. Legacy selected
the same 885 parameters and reported epoch losses `[2.3058, 2.1647, 2.1249, 2.0972, 2.0776, 2.0584,
2.0481, 2.0443]`, a reduction of about `0.2615`. The exact-sampler rewrite reduces its objective by
`0.259878`, but begins and ends roughly `0.09`–`0.10` higher. Matching optimization gain from a worse
initial objective is evidence that KD now follows the legacy update behavior and inherits the already
measured pre-KD frozen-state mismatch.

### Contemporary protocol-matched legacy parity baseline

`gemma-legacy018-contemporary-pinned-v3` reran the current legacy Experiment 018 implementation against
the exact pinned 256 x 2,048 calibration tensor and pinned local Gemma snapshot. The worker held the
rewrite cross-process device lease throughout. Compression plus eight-epoch model KD completed in
14,267.90 seconds and produced the 390,323,127-byte checkpoint with SHA-256
`798d9fe78e695a1f8f89e6dca804e9cc1fb4913de3cccae174b47e0d2b548764`.

The contemporary result resolves the historical-quality ambiguity:

- block 0 finishes at `1.3728` after refit, versus rewrite `1.37848997` and historical `1.1624`;
- all 26 contemporary/rewrite post-refit boundaries differ by only -2.20% to +2.01%;
- all 182 ranks match, with rank sum 105,856 for both implementations, so binary BPW and outlier-count
  cost match exactly;
- KD losses end at `2.1430` legacy versus `2.140396` rewrite from nearly identical starting objectives;
- `wikitext2-factorized-exact.json` reports exact serial PPL `444.332773`, versus rewrite
  `454.431449` after the matching legacy-sampled KD protocol and `432.930572` before KD.

The 2.27% tuned-PPL spread, exact rank/BPW match, <=2.20% block-boundary spread, and near-identical KD
trajectory are the approved current-environment M4 parity result. The retained historical checkpoint's
PPL `383.938808` remains valuable quality provenance, but its CCE/factor trajectory is not reproducible
bitwise in the current CUDA environment. `rewrite-structural-comparison.json`, the contemporary log,
checkpoint result manifest, and exact WikiText result retain the comparison details.

### Exact-retained-Fisher memory replay (rejected after block 1)

`gemma-retained-fisher-legacy-schedule-barrier-v3` is a fresh v20 resident replay after releasing
uncompleted decoder shells, bounding tuning staging, synchronizing the completed CUDA backward-to-optimizer
handoff, and adding bounded factorized-epoch checkpoints. It uses the exact retained-Fisher preprocessing,
legacy batch size 8, eight factorized epochs, the tapered non-factorized schedule, and two post-block refit
epochs. Earlier `batch8-v2` commits are invalid because their asynchronous optimizer handoff did not improve
the first four layer objectives; they must not be mixed with this fresh artifact store.

Block 0 is durably committed as
`sha256-737eea482f1226759dbcc13b3cd9f3184d99e5820899fd0e31532dd71205b4f6`. Store-aware validation followed
41 referenced artifacts (2,792,948,178 bytes) and found a complete seven-layer graph. The rank sum is 4,256,
all seven ranks exactly match the contemporary legacy prefix, and final pre-KD loss is `1.3784899712` versus
contemporary legacy `1.3728` (+0.414%). Quantized-layer cost is 27,790,027 bits over 26,836,992 source
parameters (1.035512 effective BPW including scales and outliers). Peak allocated CUDA is 4,743,755,264
bytes; host high-water was not captured by this frozen v20 worker and remains zero in the block payload.

The block reports 578.73 seconds versus contemporary legacy 424.87 seconds, a 36.2% gap. It includes the new
per-epoch durable checkpoints and the correctness synchronization, while the older rewrite comparison predated
both.

The v20 run was stopped immediately after its validated block-1 prefix failed the accumulating-state gate. Block 1
finished at `6.0843391418` versus contemporary legacy `3.6029` (+68.87%), despite all 14 prefix ranks matching
exactly. This is not an activation-store or staging corruption: block 0 produced the byte-identical retained
activation generation `sha256-f10ba00d42e600df92579295a406fa139e0f77ca034f855f5468a79e623ae749`
from the earlier exact-Fisher replay. It is not a full-run candidate, but this result did not isolate the objective
from the v20 numerical execution path.

`gemma-cce-memory-corrected-v21-canary` is the replacement bounded gate. Despite its historical directory name,
it uses calibration `sha256-49c65096...`, objectives `sha256-49cd2430...`, and plan `sha256-8656b828...`: the same
exact-Fisher preprocessing closure as the rejected v20 run, not the original v17 plan `sha256-7513d62f...`. It
adds the current memory, CUDA gradient-handoff synchronization, checkpoint, resource-reporting, and execution-only
cooldown code. The comparison therefore measures the combined corrected v21 execution path; it does not prove
that cooldown alone explains the v20 divergence.

The gate completed and passed. Block 0 committed as
`sha256-5e10f391f53a7835c236ee32b7618fa7c09af026a5680641da7f21c04321c56a` at
`1.3784899712`, versus contemporary legacy `1.3728` (**+0.41%**). Block 1 committed as
`sha256-ec9aae8d522eab549d3c5bcc4dbdb55c2c4191913b459fcb92fd77ca9c1df117` at
`3.5971968174`, versus contemporary legacy `3.6029` (**-0.16%**). The mean absolute delta across both
boundaries is **0.29%**, comfortably inside the approved 2.20% current-environment envelope and, unlike the
rejected v20 execution, preserves the accumulating error direction.

Fresh store-aware validation followed 79 reachable artifacts (3,153,557,256 bytes), found a contiguous
two-block/14-layer prefix under one identity, and reported 1.017989 effective BPW over the prefix. The
cooldown-aware block-1 resume peaked at 4,788,224,512 allocated CUDA bytes; live board sampling remained near
7.6 GiB and 73--80 C. Execution-only cooldown time is included in its wall clock and is not performance
evidence. This result clears the gate for bounded extension through the remaining blocks; final profiling
remains deferred until the complete parity trajectory and end-quality checks pass.

The first bounded extension committed block 2 as
`sha256-a98b14c83477ca15a3aff280d630b5a83d0b9726ce108ea4b3d7285069fb6f35` at `5.7917079926`,
versus contemporary legacy `5.7415` (**+0.87%**). Block 3 then committed as
`sha256-b7a553fb4eab7f41f91e623fe27f611731ac3da1c271a87ede5cef118c3410f6` at `43.9182624817`,
versus `43.693` (**+0.52%**). Across the first four boundaries the mean absolute delta is **0.49%**, the maximum
is **0.87%**, and all 28 ranks match contemporary legacy exactly. Store-aware validation followed 153 reachable
artifacts (3,885,792,173 bytes) and reported 1.018013 effective BPW, 6,410,993,664 peak allocated CUDA bytes,
and 11,120,852,992 peak host bytes.

The next bounded workers committed block 4 as
`sha256-81284ee29b6aa9bc34a12ffee27d70244a9b1fe36a309f7b29db4aa6abd9f743` at
`656.8922119141` versus contemporary legacy `667.26` (**-1.55%**), block 5 as
`sha256-1976eb9def049e5d74835db33984f04e561dbfc7f3397ed6a5c341d5ee9bf2b5` at
`34.1128807068` versus `34.3` (**-0.55%**), and block 6 as
`sha256-3d36f48a6e4f5bfc35aa2b51001b28189ea6c913e2a614c7e4f2adc98d8d97c5` at
`41.2283020020` versus `42.129` (**-2.14%**). Seven-block validation followed 265 reachable artifacts
(4,917,559,347 bytes), found all 49 ranks equal to contemporary legacy, and reported 0.992245 effective BPW.

Block 7 committed as
`sha256-67b1cb1aa125e65f5ba53a4ed8f194f9431cab7366dafd2566705448df710b98` at
`247.3277435303` versus contemporary legacy `256.14` (**-3.44%**). This crosses the previously observed
2.20% absolute legacy envelope, so the extension paused before block 8. Store-aware validation still found a
complete eight-block/56-layer graph under one identity: 302 reachable artifacts (5,264,580,346 bytes), all 56
ranks exactly equal to legacy (rank sum 32,448), 0.989109 effective BPW, 6,647,971,840 peak allocated CUDA bytes,
and 11,121,049,600 peak host bytes. Direct comparison with the already accepted rewrite block 7
(`250.5180664063`) is only **-1.27%**, and every corrected-v21 post-layer boundary is lower than that accepted
rewrite boundary. The alert therefore does not indicate corrupt artifacts, rank drift, or worse reconstruction;
the remaining full-trajectory and exact-quality checks must determine whether the newer numerical execution path
is an acceptable realization before performance measurements begin.

### Legacy wide-matrix factor orientation (v26)

The earlier source audit had proved exact official/rewrite ADMM parity only on a tall MLP gate projection. A formal
three-layer replay in `evidence/m2/gemma-admm-factorization-parity-native-v25.json` exposed the missing boundary:
legacy transposes matrices with fewer output than input features before solving, while the rewrite v25 solved every
matrix in native orientation. Both paths consumed the same amount of RNG and the wide attention/down-projection
objective deltas were only `1.68e-5` and `4.78e-4` relative, but their binary factors were unrelated.

`RESIDENT_ALGORITHM_VERSION = 26` restores that legacy orientation in the pure factorizer. The replacement
`evidence/m2/gemma-admm-factorization-parity.json` replays block-0 `self_attn.q_proj`, block-0 `mlp.down_proj`, and
block-1 `mlp.gate_proj` from the immutable outlier residual and post-probe generator-state artifacts. All latent
factors, binary factors, three scales, dense reconstructions, and final RNG states are now exactly equal to the
source-hashed legacy implementation, and all normalized-objective deltas are zero. The two wide results correctly
do not match their retained v23 native-orientation factor artifacts; version 26 prevents those commits from being
adopted into a new run. A fresh resident trajectory is therefore required before superseding the accepted v19/v21
end-to-end evidence.

The first v26 resident prefix is retained under `gemma-legacy-wide-v26-block0-canary`. It reuses the validated v23
content-addressed store through a directory junction, avoiding duplicate preprocessing and tensor objects. Block 0
committed at `1.3673610687` versus contemporary legacy `1.3728` (**-0.40%**) and prior accepted rewrite
`1.3784899712` (**-0.81%**). Block 1 committed at `3.5729382038` versus contemporary legacy `3.6029`
(**-0.83%**) and prior corrected-v21 rewrite `3.5971968174` (**-0.67%**). All 14 ranks match contemporary legacy
exactly (rank sum 8,352).

Store-aware validation followed 79 reachable artifacts (3,153,557,242 bytes), found one identity and a contiguous
two-block prefix, and reported 1.017989 effective BPW, 6,295,650,304 peak allocated CUDA bytes, and
11,129,999,360 peak host bytes. Block 0 took 399.22 seconds, 6.0% faster than contemporary legacy's 424.87 seconds
and 31.0% faster than the older 578.73-second rewrite gate; performance remains provisional until a complete run.
`validation.json` and `legacy-comparison.{json,md}` retain the compact validation and comparison evidence. At two
blocks this prefix appeared to clear the accumulating-state gate, so it was extended. The longer prefix rejected
that conclusion. Block 2 remained close at `5.7664198875` versus contemporary legacy `5.7415` (**+0.43%**), but
block 3 finished at `125.4999160767` versus `43.693` (**+187.23%**). All 28 ranks still matched exactly (rank sum
16,608), the artifact graph remained valid, peak allocated CUDA memory stayed bounded at 6,534,725,632 bytes, and
effective BPW was 1.018013. This isolates a numerical-trajectory regression rather than rank, storage, resume, or
memory corruption.

The divergence was already visible at the block-3 gate boundary: the transposed run recovered only from `143.69`
to `82.91`, while contemporary legacy recovered from `62.986` to `28.467`. Its down-projection also reported a
much lower local weighted reconstruction error than either legacy or the accepted native-orientation rewrite while
producing a substantially worse downstream boundary. Exact component replay therefore does not imply a better
composed trajectory; the orientation changes the factor error geometry that later tuning must absorb.

The v26 worker was intentionally stopped after four complete blocks and two partial block-4 layer commits. The
updated `validation.json` and `legacy-comparison.{json,md}` preserve the rejected four-block evidence. Version 27
makes `ADMMConfig.transpose_wide` explicit: `false` is the system-validated production default, while `true`
remains available to `tools/compare_admm_factorization.py` and `--transpose-wide` for exact legacy-source replay.
A fresh native-orientation v27 prefix is required before extending the run or superseding accepted v19/v21
end-to-end evidence.
