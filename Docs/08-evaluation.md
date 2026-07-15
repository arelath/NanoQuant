# Evaluation Strategy

## 1. Evaluation answers decisions

Evaluation is not one final benchmark command. It is a staged decision system:

- Is the artifact structurally and numerically valid?
- Is the idea clearly harmful or promising?
- Is the observed difference large enough to justify a full run?
- Did quality improve at the same effective size and runtime point?
- Is the result robust enough to become the new baseline?

The rewrite separates cheap rejection tests from expensive confirmation while keeping their limits visible.

## 2. Evaluation dimensions

No single score represents NanoQuant quality. Reports preserve separate dimensions.

### Representation correctness

- packed artifact validation;
- effective bits per weight and total bytes;
- per-layer raw and objective-weighted reconstruction error;
- latent-to-export and scale-fit error changes;
- backend reference parity;
- non-finite values and invalid metadata.

### Local behavioral quality

- block-output reconstruction loss;
- quantization jump and tuning recovery;
- final frozen block error versus both the source/base-model reference and the block-entry pre-quantization baseline;
- pre-KD versus post-KD block recovery;
- held-out block replay loss;
- rank/outlier utility per added bit;
- sensitivity of results across calibration partitions.

### End-to-end model quality

- held-out negative log-likelihood and perplexity;
- established zero/few-shot task accuracy or normalized scores;
- long-context behavior where supported;
- generation sanity and deterministic prompt regression cases;
- task-family aggregates with the underlying task results retained.

### Deployment behavior

- packed size and load time;
- peak VRAM and host memory;
- time to first token;
- prefill throughput;
- decode throughput and latency distribution;
- energy per token where available;
- backend coverage and fallback count.

### Quantization cost

- calibration, factorization, tuning, packing, and evaluation time;
- peak GPU/CPU/disk use;
- bytes transferred and temporary storage;
- failed/retried work;
- estimated hardware cost where meaningful.

## 3. Evaluation tiers

Exact task/sample definitions live in a versioned evaluation registry. The following illustrates intent.

| Tier | Approximate scope | Purpose | Promotion rule |
| --- | --- | --- | --- |
| `smoke` | artifact validation, reference parity, tiny held-out token slice, deterministic prompts | Catch broken outputs and catastrophic regressions | All structural/parity gates pass; no catastrophic loss |
| `quick` | fixed held-out PPL slices, selected task examples, representative block fixtures, short runtime case | Reject clearly inferior ideas cheaply | Candidate meets configured quality, size, and runtime guards |
| `standard` | complete PPL dataset(s), curated task subset, representative performance matrix | Decide whether to spend on full evaluation | Improvement or ambiguity exceeds minimum meaningful threshold |
| `full` | complete benchmark suite, long-context/performance matrices, repetitions/seeds as required | Accept/reject a candidate baseline or publish a result | All release gates and comparability checks pass |

An evaluation result always displays its tier. Quick results cannot be presented as if they were full-suite evidence.

## 4. Cheap early signals

Useful early signals include:

- non-finite or malformed packed state;
- reference parity failure;
- large held-out NLL regression on a fixed token slice;
- severe block loss in historically sensitive blocks;
- a material Experiment 019-style `Block final` regression after all local tuning/refitting;
- effective BPW above the intended budget;
- unexpected inference fallback or drastic throughput regression;
- deterministic generation degeneration such as empty/repeated output.

These signals are chosen for high precision when rejecting obviously bad candidates. They are not assumed to rank close candidates accurately.

## 5. Held-out data

Evaluation data is separated from calibration and tuning data by construction:

- sample identities are content hashed;
- overlap checks run before evaluation;
- calibration, early-decision, and final-evaluation partitions have distinct IDs;
- task revisions and prompt templates are pinned;
- chat-template and BOS/EOS handling are recorded;
- samples excluded due to length or formatting are counted and reported.

Using calibration loss as a primary quality metric is prohibited. It remains a diagnostic only.

## 6. Evaluation registry

Each evaluator entry defines:

```yaml
id: ppl-wikitext2-v1
kind: perplexity
dataset:
  name: wikitext2
  revision: abcdef12
split: test
tokenizer_policy: source-model
sequence_length: 2048
stride: 2048
max_samples: null
metric:
  name: perplexity
  direction: minimize
implementation_version: 1
resource_class: single_gpu
```

Task registries also pin few-shot count, selection seed, prompt formatting, answer normalization, batch behavior, and evaluator package version.

### 6.1 Pinned legacy multiple-choice suite

The rewrite implements the six multiple-choice tasks used by legacy `007-evaluate-all-gemma-quality.py` under
lm-eval 0.4.12 at harness commit `3ba40d3`. The retained protocol is zero-shot, ordered, limited to the first 200
evaluation rows, maximum causal input length 2048, and batch size one. The generic renderer also accepts an exact
ordered demonstration set for few-shot variants; demonstration content hashes, count, and selection seed are part
of the task-input cache identity. None of the six retained legacy task invocations uses demonstrations.

| Task | Dataset revision | Split | Primary metric | 200-row task-input key |
| --- | --- | --- | --- | --- |
| PIQA | `142f6d7367fd9877f0fb3b5734ea6a545f54cdd1` | validation | `acc_norm` | `sha256:05a86219cc331fd279cd0b8e6a4620f8228ffa750799f4ae1337db5c45412067` |
| ARC Easy | `210d026faf9955653af8916fad021475a3f00453` | test | `acc_norm` | `sha256:4b8aef8a11c2d13314735ed380f97bf46a15deddfb0f660cb29c86b5b9bd39d0` |
| ARC Challenge | `210d026faf9955653af8916fad021475a3f00453` | test | `acc_norm` | `sha256:4a2df3cc5b5b7bd1c9976142acfecae6aee5c4163fa4667bd5bd5d812843ca5e` |
| HellaSwag | `218ec52e09a7e7462a5400043bb9a69a41d06b76` | validation | `acc_norm` | `sha256:54aa364a4356023d5c0e15a0de7e13942074e9ab9c3798ae3e85713106d29176` |
| WinoGrande | `01e74176c63542e6b0bcb004dcdea22d94fb67b5` | validation | `acc` | `sha256:3cf915b5936d0beec011549d3b9238554d4cf0e51c0a8d3b44afef1b94cff96f` |
| BoolQ | `3de24cf8022e94f4ee4b9d55a6f539891524d646` | validation | `acc` | `sha256:2162a165f5409ec974b543350024ff6998f3d13004114d9a16b94151fc934cde` |

ARC deliberately uses the test split: the harness declares both validation and test data and `simple_evaluate`
selects test when it exists. The retained legacy row-zero IDs and prompt arguments prove that behavior. PIQA,
HellaSwag, WinoGrande, and BoolQ use validation because those task definitions do not declare a test split.

The Hugging Face adapter reproduces lm-eval causal pair encoding: trailing context spaces move to the continuation,
the concatenated string and context are encoded separately, and the continuation is the suffix after the context
token count. Gemma BOS insertion is explicit (`add_special_tokens=true`) instead of inherited from a mutable
tokenizer default. The pinned tokenizer behavior-file hash is
`sha256:19317db471b30f6cfa877d781ecac1db28de6628e44e3751df0c44344444a811`.
The causal window also matches the harness's `max_length + 1` context-plus-target rule: the final target token is not
fed as model input, so left truncation does not discard one extra context token.

Evaluation retains both summed and continuation-length-normalized log likelihoods, computes log-softmax and score
accumulation in FP32, records both accuracy variants, ties, per-example scores, and truncation counts, and chooses
the task's pinned primary metric. Tests cover known logits, serial/batched equality, limiting, exact window edges,
cache invalidation, and all six real cached dataset/tokenizer row-zero hashes against the retained legacy samples.

### 6.2 Gemma long-context protocol

The full-tier `gemma3-hybrid-cache` evaluator binds the model's declared 32,768-token context limit, 512-token
sliding window, six-layer global-attention interval, prefill chunk size, and zero-fallback policy into one semantic
identity. Each case pins prompt and expected token IDs plus its stop reason, must cross both a chunk and the sliding
window, and reports exact tokens, first mismatch, prefill/decode call counts, cache bound, fallback count, and peak
device allocation. Requests beyond the model limit fail with `NQ-GEN-CONTEXT` before a forward.

Runtime prefill now streams a prompt through one HybridCache. Prepared prefill dispatch accepts any positive token
geometry up to its planned prompt bound while decode remains fixed at one token per batch item. Gemma local layers
retain the previous window in chronological order across multi-token cache updates and construct the causal sliding
mask from absolute query/key positions. This avoids materializing a full 32K attention operation while preserving
global-layer history.

The retained packed Gemma runtime-bundle gate uses F32 shell execution and four forced greedy tokens. A monolithic
oracle and 512-token candidate agree exactly at 1,025 and 4,097 prompt tokens; the 4K candidate uses nine prefill
forwards and peaks at 899,101,696 allocated bytes versus 1,737,259,008 for the monolithic oracle. The near-ceiling
case uses an independently bounded 256-token oracle and 512-token candidate: 32,761 prompt plus four generated
tokens, 128 versus 64 prefill forwards, exact token/stop/cache parity, zero dispatch fallbacks, and a candidate peak
of 1,592,178,176 bytes on the 12 GB designated GPU. The model's exact configured ceiling is also exercised against
Transformers generation by a tiny deterministic Gemma fixture.

Evidence is retained under `evidence/m8`: `gemma-pageable-v28-long-context-1024.json` (SHA-256
`75d5141123d7b8416a5e427874cc012e24f32ef1dab2245e5cddab616000b688`),
`gemma-pageable-v28-long-context-4096.json` (SHA-256
`f63fc2025b4b10356a8a6a77b19e4c3593285b984475e234d5447dc5912ec581`), and
`gemma-pageable-v28-long-context-ceiling.json` (SHA-256
`c17f2cf200fcd151d880c4389ae191a4c6dfb296628047abc950803b06db3cf4`).

### 6.3 Ordered evaluation campaigns

`run_evaluation_campaign` is the shared promotion boundary from a captured layer replay through exact tier-local
quick, standard, and full evaluator sets. A campaign requires those three plans in order, binds every evaluator
specification/result and gate policy by semantic hash, rejects missing or unexpected tier requests, and stops before
later work on rejection or inconclusive evidence. Its result preserves the last completed tier, every metric and
decision, the terminal outcome, and a deterministic next action.

The retained Gemma v28 campaign at `evidence/m8/gemma-pageable-v28-evaluation-campaign-v2` copies all seven compact
inputs into its own directory and derives its manifest, event stream, canonical campaign JSON, summary, comparison,
environment, resource observations, conclusion, and recommendation exclusively from those files. It promotes:

- layer replay at 0.3540% mean absolute loss delta over the first four blocks;
- quick validation with all 979 artifacts valid, 182 layers, identical ranks, 0.996318 BPW, and a 4.2188% maximum
  block delta;
- standard exact WikiText-2 with 8,128 scored targets and 453.571 PPL, +2.079% from contemporary legacy within the
  frozen +2.27% environment-matched band;
- full deployment evaluation with exact 32,765-token context behavior, zero fallbacks, 1,592,178,176 peak device
  bytes, and 160.743 versus 184.5 tokens/s (87.12%, above the predefined 70% gate).

The canonical campaign result is SHA-256
`ccacea00ee1b4e7678ae9f662827fd45cf5ec28071918c88ef67f799a0f35cc8`; the generated summary has no consistency,
warning, or error findings. This workflow demonstration closes M8.GATE, not the broader release-candidate task suite
in M10.14.

## 7. Baselines

A candidate can be compared with several named baselines:

- original BF16/FP16 source model;
- currently accepted NanoQuant artifact at a similar BPW;
- direct parent run;
- compatible external quantization/runtime implementation;
- algorithm ablation.

The primary baseline must be declared before results are known. Reports warn when model revision, tokenizer, dataset, evaluator, effective BPW, or runtime workload differs materially.

## 8. Statistical comparison

Where per-example scores exist, comparisons use paired data. Reports include:

- sample count;
- candidate and baseline value;
- absolute and relative delta;
- paired bootstrap confidence interval or another evaluator-appropriate interval;
- repeated-run variability for stochastic execution;
- a predefined minimum meaningful difference;
- pass, fail, or inconclusive status.

For accuracy on small quick subsets, a one-example change can be misleading; the report shows counts and uncertainty rather than only a percentage. Multiple-task sweeps retain individual task results and avoid declaring success solely from the best-moving task.

Perplexity is computed from accumulated token-level negative log-likelihood and valid-token count. The implementation tests BOS shifting, padding masks, stride, and partial final windows.

## 9. Sequential promotion

Evaluation proceeds until a decision is justified:

```text
artifact/parity fail ──► reject immediately
        │ pass
        ▼
quick clearly worse ──► reject
        │ promising or inconclusive
        ▼
standard clearly worse ──► reject
        │ promising or inconclusive
        ▼
full evaluation ──► accept / reject / retain as tradeoff point
```

Promotion policy is part of the run intent and cannot be edited after seeing results without creating a documented policy revision.

Examples of guards:

- effective BPW must remain within 0.01 of baseline;
- quick held-out NLL may not regress more than a predefined threshold;
- no task-family aggregate may cross a critical regression limit;
- decode throughput may not regress more than 10%;
- peak VRAM may not exceed the deployment target;
- quantization time increase must be justified by quality gain.

## 10. Pareto decisions

The preferred output is a Pareto comparison across:

- model quality;
- effective BPW/packed bytes;
- decode and prefill speed;
- runtime memory;
- quantization cost.

The system should not hide these dimensions behind one permanently weighted score. A score may be used for a named deployment profile, but its weights and thresholds are versioned and all raw metrics remain visible.

## 11. Result schema

Every evaluation result contains:

```json
{
  "schema_version": 1,
  "evaluation_id": "ppl-wikitext2-v1",
  "tier": "quick",
  "model_artifact": "sha256:...",
  "baseline_artifact": "sha256:...",
  "dataset_identity": "sha256:...",
  "implementation_version": 1,
  "seed": 0,
  "sample_count": 256,
  "valid_token_count": 524288,
  "metrics": {},
  "comparison": {},
  "resource_usage": {},
  "warnings": [],
  "status": "completed"
}
```

Partial and failed evaluations retain completed task results and terminal error context but cannot satisfy promotion gates.

## 12. Evaluation caching

Evaluation cache identity includes the complete model artifact, evaluator/task revision, sample selection, tokenizer behavior, runtime mode where numerically relevant, and evaluator implementation version. Changing console output or report formatting does not invalidate evaluation.

Preprocessed task inputs may be cached independently from model results. Cached results are never reused across a changed packed artifact merely because the source recipe is similar.

The implementation uses two immutable artifact types and a run-local atomic index:

- `evaluation-task-inputs` binds the evaluator semantic key; task and pinned dataset revisions/content; split;
  partition name, version, content hash, and ordered sample hashes; tokenizer revision/content and all behavioral
  parameters; prompt revision/content; exact few-shot demonstration hashes; selection seed; and preprocessing
  implementation version. It intentionally excludes the model artifact so identical preprocessed inputs can be
  reused across candidate and baseline models.
- `evaluation-result` binds the exact model `ArtifactRef`, evaluator key, task-input key, runtime backend/version/mode
  and numerical parameters, optional environment hash, and evaluation seed. Consequently a changed packed model,
  runtime numerical mode, evaluator implementation, or selected example is a miss even when the recipe name is the
  same.

Every cache hit first validates the content-addressed artifact and then compares its embedded identity with the
requested identity and index key. Publication is serialized across processes and atomically replaces the sorted
index. A second payload under the same semantic identity is an error, not an overwrite; interrupted publication can
leave only an unreferenced immutable object that ordinary artifact garbage collection can reclaim. Lookup results
carry an explicit hit/miss status and reuse/invalidation reason. Report formatting and console verbosity are absent
from both identities.

## 13. Evaluator validation

Before an evaluator can gate research decisions, tests establish:

- known logits produce expected loss/perplexity;
- token shifts and masks are correct;
- batching and unbatched evaluation agree;
- cached and uncached inputs agree;
- reference source-model results are within expected published/internal ranges;
- sample limits select deterministic examples;
- distributed reduction agrees with single-process execution;
- evaluator version changes produce an explicit new identity.

The causal evaluator now exposes `maximum_samples` as part of its request and applies the deterministic leading
selection before window construction or batching. Results retain selected-sample, window, and valid-token counts.
Distributed workers return the same sufficient statistics as a local run; reduction uses an accurate sum of total
negative log likelihood and divides once by the global valid-token count, then sums windows and selected samples.
It never averages shard means or perplexities. Validation uses unequal shards with different loss distributions so
an unweighted mean-of-means implementation would fail, alongside serial/batched, exact next-token-logit,
cached/no-reexecution, deterministic sample-limit, padding, stride, and partial-window cases.

## 14. Reporting an inconclusive result

Close results are expected. The correct output may be:

```text
INCONCLUSIVE at quick tier
Observed held-out NLL delta: +0.0007
95% paired interval: [-0.0012, +0.0025]
The interval crosses both zero and the meaningful-regression boundary.
Recommended next action: run standard tier on the same artifact.
```

This is preferable to turning noise into a new full quantization direction.
