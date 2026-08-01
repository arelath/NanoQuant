# Qwen3 Thinking-Mode Quality Recovery

Status: design proposal; not yet implemented

Date: 2026-07-25

Audience: dataset, calibration, block tuning, evaluation, export, and experiment recipe maintainers

## 1. Summary

Qwen3 checkpoints are hybrid reasoning models: the chat template defaults to thinking mode
(`enable_thinking=True`), in which the model emits a `<think>…</think>` reasoning span before its
visible answer. Our compressed Qwen3 artifacts (Experiments 028 and 029) produce acceptable output
in non-thinking mode but very poor output in thinking mode — the mode users get by default.

The root cause is that every data-driven stage of the compression pipeline — calibration
statistics, sensitivity-based rank allocation, block tuning, post-block refit, outlier selection,
and quality evaluation — runs exclusively on a mixture that contains no reasoning traces:
UltraChat 200k chat turns plus WikiText-2 raw text
([base_compression.py:44](../experiments/recipes/base_compression.py:44)). Thinking-mode token
sequences are therefore out-of-distribution for the compressed weights, and no quality gate ever
exercised that mode, so the regression shipped invisibly.

This document proposes three workstreams:

1. **W1 — thinking-trace calibration data** (the actual fix): add a pinned reasoning-trace dataset
   to the Qwen3 calibration mixture, formatted so `<think>` spans survive the chat template.
2. **W2 — thinking-mode evaluation** (make the gap measurable and gate on it): dual-mode quality
   scoring in both the native evaluator and the llama.cpp GGUF quality path.
3. **W3 — interim mitigation for published artifacts**: document and default the published GGUFs
   to non-thinking usage until re-runs with W1+W2 land.

## 2. Background and root cause

### 2.1 How Qwen3 thinking mode works

- With `add_generation_prompt=True` and default settings, the Qwen3 template ends the prompt with
  an assistant header and the model opens a `<think>` span (dedicated special tokens) containing
  free-form reasoning, closes it with `</think>`, then writes the visible answer.
- `enable_thinking=False` (or a `/no_think` soft switch in the user turn) makes the template emit
  an empty `<think>\n\n</think>\n\n` pair so the model skips reasoning.
- When formatting *completed* conversations (`add_generation_prompt=False`, which is what our
  dataset preparation uses), the Qwen3 template strips `<think>` content from all but the final
  assistant turn. Reasoning content only survives formatting when it is present in the last
  assistant message of the rendered record.

### 2.2 What the pipeline feeds the model today

- The calibration mixture is 50% UltraChat 200k (`train_sft`) + 50% WikiText-2 raw text, 256
  samples at `sequence_length=2048`
  ([base_compression.py:44](../experiments/recipes/base_compression.py:44),
  [schema.py:162](../src/nanoquant/config/schema.py:162)).
- Chat records are rendered with `tokenizer.apply_chat_template(..., add_generation_prompt=False)`
  ([dataset.py:121](../src/nanoquant/application/dataset.py:121),
  [hf_calibration_dataset.py:144](../src/nanoquant/infrastructure/hf_calibration_dataset.py:144)).
  UltraChat contains no reasoning traces, so even the final-turn preservation path never emits a
  non-empty `<think>` span. The thinking special tokens effectively never appear in calibration
  activations.
- The Qwen3 experiment templates inherit this mixture unchanged from the Llama template
  ([base_compression.py:412](../experiments/recipes/base_compression.py:412)); nothing in the
  repository references `enable_thinking` at all.

### 2.3 Why this breaks thinking mode specifically

All of the following consume calibration activations and therefore optimized the compressed model
for the non-thinking distribution only:

- covariance/Hessian statistics used for factorization and quantization (`calibration.*`);
- sensitivity-based rank allocation (`AllocationStrategy.SENSITIVITY`) — ranks were assigned where
  *non-thinking* traffic needs fidelity;
- block tuning and post-block refit — the factorized layers were explicitly trained to match
  teacher outputs on non-thinking activations;
- residual outlier selection.

Thinking-mode generation drives the model through activation regions the compression never saw:
the `<think>`/`</think>` special tokens themselves, long self-referential reasoning spans, and a
different length/position profile. Low-rank approximations are precisely the kind of compression
that discards directions unused by the calibration set, so degradation concentrated in the unseen
mode is the expected failure signature — analogous to the distribution findings in
[33-error-budget-driven-quality-improvements.md](33-error-budget-driven-quality-improvements.md).

### 2.4 Why no gate caught it

Quality scoring (native causal NLL partitions and llama.cpp GGUF perplexity) evaluates
teacher-forced likelihood on the same kind of non-thinking text as calibration. No stage generates
in thinking mode or scores likelihood over reasoning traces, so the published artifacts passed all
gates while being broken in their default mode.

## 3. Goals and non-goals

Goals:

- G1: Compressed Qwen3 models degrade comparably in thinking and non-thinking modes; thinking-mode
  quality loss versus the pinned base model is bounded and measured.
- G2: Thinking-mode quality is a first-class, gated metric for any hybrid-reasoning model, so this
  class of regression cannot ship silently again.
- G3: Non-Qwen recipes and their fingerprints/caches are untouched; determinism and revision
  pinning are preserved for the new data.

Non-goals:

- Fine-tuning or distilling the base model beyond the existing block-tuning machinery.
- Changing the packed format, GGUF export contracts, or the compression algorithm itself.
- Multi-turn thinking-trace retention (upstream template strips prior-turn traces; we follow it).

## 4. W1 — thinking-trace calibration data

### 4.1 Data source

Two options:

- **Option A (recommended first step): pinned public reasoning-trace dataset.** Use an open
  dataset whose records carry explicit reasoning traces (e.g. `open-r1/OpenR1-Math-220k` or
  `open-thoughts/OpenThoughts-114k`; final selection is an implementation decision gated on
  license review and a schema check). Pros: deterministic, revision-pinnable exactly like
  UltraChat/WikiText-2, no new infrastructure. Cons: traces were generated by a different model
  (R1-style), so the distribution is close to but not identical to Qwen3's own thinking style.
- **Option B (follow-up if A is insufficient): self-generated traces.** Run the pinned base Qwen3
  checkpoint in thinking mode with greedy decoding over a fixed prompt set, cache the traces as a
  pinned artifact, and feed them back as calibration data. Pros: exactly the target distribution.
  Cons: a new generation stage with GPU cost, determinism caveats across hardware/kernels, and a
  new artifact-caching surface.

Option A directly removes the "thinking tokens never seen" failure and is cheap; run it first and
only invest in Option B if the Experiment 030 acceptance gate (Section 6) does not pass.

### 4.2 Formatting

Add a formatting variant (e.g. `qwen-thinking-chat-plus-raw-text-v1`) alongside the existing
`gemma-chat-plus-raw-text-v1`:

- Render each reasoning record as a **single-turn** conversation whose final (only) assistant
  message embeds the trace: `user: <problem>`, `assistant: <think>\n<trace>\n</think>\n\n<answer>`.
  Single-turn packing guarantees the template's final-turn preservation rule keeps the trace.
- Reuse the existing windowing over `sequence_length=2048`. Traces longer than the window are
  truncated mid-trace; that is acceptable for activation statistics, but windows must start at the
  sample start so the `<think>` open token is always in-window.
- Add a unit test that formats one reasoning record through the real pinned Qwen3 tokenizer and
  asserts the `<think>`/`</think>` token ids appear in the rendered ids. This test is the guard
  against silent template-stripping regressions when tokenizer revisions are bumped.

### 4.3 Mixture and config plumbing

- Override `dataset.sources` in `QWEN_3_0_6B_COMPRESSION_TEMPLATE` (inherited by the 8B template)
  rather than in any shared template: reasoning traces 40%, UltraChat 30%, WikiText-2 30% as the
  starting point. Raise `calibration.sample_count` from 256 to 384 so the absolute count of
  non-thinking samples stays roughly constant and only the thinking share is additive.
- `DatasetSourceConfig` already carries name/revision/split/subset/weight; the new source needs a
  record-to-messages mapping for the reasoning dataset's schema (problem/trace/answer fields),
  which lands next to the UltraChat mapping in
  [hf_calibration_dataset.py](../src/nanoquant/infrastructure/hf_calibration_dataset.py).
- Dataset fingerprints already hash sources, formatting id, and chat-template hash
  ([dataset.py:160](../src/nanoquant/application/dataset.py:160)), so caches invalidate correctly
  and non-Qwen fingerprints are untouched. No fingerprint-schema change is needed.

The mixture ratio is a tunable, not a commitment; Experiment 030 (Section 6) measures whether 40%
is enough and whether non-thinking quality regresses.

## 5. W2 — thinking-mode evaluation and gating

Without W2 we cannot verify W1, so W2 lands first or together with W1.

- **Native NLL partitions.** Extend partition construction so hybrid-reasoning models get
  thinking-trace items in all three partitions (`calibration` / `quick_decision` /
  `final_evaluation`), drawn from held-out records of the same reasoning dataset, with a partition
  version bump. Report thinking and non-thinking NLL separately in the comparison report rather
  than folding them into one number, so the per-mode delta versus the pinned base model is visible.
- **llama.cpp GGUF quality.** The GGUF is the deployed artifact and the surface where the failure
  was observed, so it must be scored per-mode too: score perplexity over a rendered thinking-trace
  text file in addition to the current corpus, and add a generation smoke test — a small fixed
  prompt set run once with the default (thinking) template and once with `/no_think`, with outputs
  captured into the run report for human inspection. The smoke test is diagnostic, not gated;
  the likelihood metrics are gated.
- **Gate.** For hybrid-reasoning models: thinking-mode NLL ratio (compressed / base) must not
  exceed the non-thinking ratio by more than a margin (proposed 10% relative; finalized after the
  Experiment 030 control run establishes the current gap). A model that fails only the thinking
  gate fails the experiment.
- **Scope control.** Both partition changes and gates activate per-model (a flag on the experiment
  template, e.g. `evaluation.thinking_mode=True`), so Gemma/Llama runs, their partitions, and
  their evaluation caches are byte-identical to today.

## 6. Rollout

1. **Experiment 030 — Qwen3 0.6B, dual-mode baseline + fix.** Re-run Experiment 028's settings
   with the W1 mixture and W2 evaluation; also score the existing 028 artifact with W2 to quantify
   the current thinking-mode gap (the control measurement). Acceptance: thinking gate passes and
   non-thinking quality within noise of 028.
2. **Experiment 031 — Qwen3 8B.** Mirror Experiment 029 with the accepted 030 settings, including
   the serial llama.cpp quality path.
3. **Republish.** Upload the new GGUFs over the existing Hugging Face releases with model cards
   documenting per-mode quality numbers.

## 7. W3 — interim mitigation (before re-runs land)

The published `Qwen3-0.6B`/`Qwen3-8B` NanoQuant GGUFs are broken in their default mode today:

- Update the Hugging Face model cards immediately: state that the current artifacts are calibrated
  for non-thinking use, and show how to disable thinking (`enable_thinking=False` via
  `apply_chat_template`, `/no_think` in llama.cpp chat, or `--reasoning-budget 0` where supported).
- Do **not** rewrite the embedded GGUF chat template to force non-thinking: diverging from the
  upstream template creates a second, sticky behavioral difference that survives after the real
  fix ships. Documentation plus recommended flags is the reversible mitigation.

## 8. Risks and open questions

- **Trace provenance mismatch (Option A):** R1-style traces approximate but don't match Qwen3's
  own thinking distribution. Mitigated by the Experiment 030 gate; escalation path is Option B.
- **License review** of the chosen reasoning dataset must clear redistribution via our published
  artifacts' data statement before pinning.
- **Non-thinking regression:** shifting 40% of the mixture could nudge rank allocation away from
  chat traffic. The 030 acceptance criterion explicitly checks non-thinking parity with 028.
- **Calibration cost:** sample_count 256 → 384 raises calibration and tuning time ~1.5× for Qwen3
  runs only; adaptive memory planning (Doc 34) absorbs the memory side.
- **Open:** exact dataset + revision pin; final mixture ratio; final gate margin; whether future
  hybrid models (e.g. other reasoning-capable checkpoints) reuse this as a generic
  "reasoning-trace mixture" template rather than a Qwen-specific one — the design deliberately
  keys everything on a template flag, not on the model family, to keep that door open.

## 9. Implementation checklist

1. W2: per-mode partitions, comparison-report columns, llama.cpp thinking-corpus scoring, smoke
   test, gate wiring behind `evaluation.thinking_mode` (touches
   `src/nanoquant/application/evaluation.py`, `comparison_report.py`,
   `src/nanoquant/llamacpp_quality.py`, config schema).
2. W1: reasoning-dataset source mapping + `qwen-thinking-chat-plus-raw-text-v1` formatting +
   template-preservation unit test (touches `hf_calibration_dataset.py`,
   `application/dataset.py`, `tests/unit/test_hf_calibration_dataset.py`).
3. Recipe: Qwen template overrides for sources, sample_count, and `evaluation.thinking_mode`
   (touches `experiments/recipes/base_compression.py`).
4. Experiments 030/031 definitions following Doc 29 conventions.
5. W3: model-card updates for the two published repos (can proceed immediately, independent of
   code).
