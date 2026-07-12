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
