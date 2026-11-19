# Experiment 070: primary/residual rank-split probes

Date: 2026-11-11

## Question and isolation

Does moving capacity from the first sign-factor stage into a later, independently
scaled stage improve quality at the same represented-bit budget? The resident
070 allocator and its completed artifacts are unchanged. These standalone
probes do not change a resident numerical path or its algorithm version.

Source: `evidence/070/070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it`.
A fresh complete artifact audit passed: 26 blocks, 874 immutable artifacts,
156 journal records, 1.9999743 represented BPW. The audit is retained at
`evidence/070/rank-split-source-validation.json`.

## Local screen

Tool: `tools/probe_residual_rank_split.py`.
Evidence: `evidence/070/rank-split-screen-20261111`.

- Block 25: down, gate, up, attention output, and stacked QKV.
- Seeds 0 and 1 for every split; no best-seed selection.
- Use the committed source-weight/Fisher objectives, including QKV member
  importance, and the actual allocated stage ranks from the 070 commits.
- Refit every arm from scratch: 800 outer ADMM iterations per stage, five inner
  iterations, the retained production numerical settings, and two scale-fit passes.
- Compare the default split, balanced split, and reversed split where distinct.
  For fixed stage count, all candidates have identical total rank and exact cost.
- For a single-stage owner, separately test a paid two-stage alternative:
  reduce total rank enough to charge the additional outer scale vectors.
  This is a stage-count test, not a pure equal-rank split test.
- Capture BF16 teacher inputs on eight calibration sequences truncated to 128
  tokens. Jointly train every stage's shadow signs and scales for eight epochs
  (64 optimizer steps). Binary LR 3e-5, scale LR 1e-5, AdamW without weight decay,
  gradient norm clip 1. This local linear-output objective is deliberately much
  smaller than production block tuning and model-level distillation.
- Score weighted weight NMSE and per-sequence linear-output NMSE on eight
  WikiText **validation** windows of 128 tokens, offsets 64 through 71.
- Reload hard-sign/BF16-scale payloads before final scoring. The diagnostic
  safetensors files hold signs as INT8; reported bit costs charge the production
  one-bit representation, not diagnostic file size. This is not native GGUF testing.

Validation inputs are disjoint from this probe's fitting inputs. They are an
exploratory split-selection set, not a new final quality test or a claim of
statistical significance. Reversing stages changes the sequential fitting
problem; equal rank does not imply the same learned solution.

## Full-model confirmation

Tool: `tools/score_residual_rank_split.py`.

For a split selected by the local screen, compare each seed against its fresh
default-split control using full-vocabulary teacher KL and NLL. Load the globally
tuned 070 model, preserve its remaining layers and auxiliary parameters, and
replace only the tested owner. Include unchanged 070 as an additional reference;
the matched comparison is the fresh candidate versus fresh default, not either
fresh fit versus the fully distilled reference.

Use 16 separate WikiText validation windows, offsets 128 through 143, at the
same 128-token length. Record per-sequence metrics and paired bootstrap intervals.
These are small dense-splice diagnostics, not the final benchmark or a native
packed-runtime acceptance gate. Do not promote an allocator change from a local
weight or activation error alone.

## Results

Completed 30 local fits (five owner types, three arms each, two seeds), about
476 seconds of fitting/tuning/scoring after input capture. The table reports
mean validation activation **NMSE**, not RMSE, across both seeds. Negative
changes are better. For the single-stage owners, the table shows the less bad
of the two paid second-stage alternatives; neither beat the one-stage control.

| Owner | Default ranks | Tested alternative | Default NMSE | Alternative NMSE | Change |
| --- | --- | --- | ---: | ---: | ---: |
| Down | 1152 + 608 | 608 + 1152 | 0.02801885 | 0.02601095 | -7.166% |
| Gate | 1152 + 864 | 864 + 1152 | 0.02349139 | 0.02296407 | -2.245% |
| Up | 1152 | 832 + 288 | 0.09954091 | 0.10912650 | +9.630% |
| Attention output | 960 | 672 + 256 | 0.01172460 | 0.01272566 | +8.538% |
| Stacked QKV | 896 | 640 + 224 | 0.06843249 | 0.07578899 | +10.750% |

Down and gate alternatives cost exactly 14,478,848 and 16,547,328 bits,
respectively, matching their controls. Up alternatives cost 9,307,648 versus
9,437,184 bits; attention output 2,103,808 versus 2,139,136; QKV 2,422,272
versus 2,465,792. These small savings result from rank-32 alignment after
charging the new outer scales, not a hidden free second stage.

The balanced down split (896 + 864) increased mean activation NMSE by 3.081%.
The balanced gate split (1024 + 992) improved it by 1.181%. Thus simply balancing
the stages is not supported, even by the local metric.

### Full-vocabulary confirmation in the distilled 070 model

Evidence directories:

- `evidence/070/rank-split-down-confirmation-20261111`
- `evidence/070/rank-split-gate-confirmation-20261111`

Each confirmation evaluated unchanged 070 and four matched fresh-fit arms
(default/candidate for each seed) on 16 disjoint validation sequences. Below,
deltas are candidate minus the same-seed fresh default. Intervals are paired
95% bootstrap intervals over the 16 sequences, not across independently trained
models or independent validation corpora.

| Owner / seed | KL change | KL delta, nats/token [95% interval] | NLL delta, nats/token [95% interval] |
| --- | ---: | --- | --- |
| Down / 0 | +2.258% | +0.011473 [+0.007675, +0.015534] | +0.011661 [+0.004750, +0.019607] |
| Down / 1 | +2.732% | +0.013716 [+0.010028, +0.017463] | -0.033324 [-0.045805, -0.020890] |
| Gate / 0 | -0.748% | -0.003631 [-0.005491, -0.001953] | +0.009266 [+0.003376, +0.014964] |
| Gate / 1 | -0.395% | -0.001919 [-0.004816, +0.001150] | +0.005221 [+0.001280, +0.008983] |

The unchanged 070 reference scored KL 0.52058589 and NLL 4.28366384 on this
specific diagnostic window. These are not directly comparable to the report's
different final WikiText test window. Both confirmation runs reproduced the
same unchanged-reference metrics.

### Disposition

**Do not change the production rank policy from these probes.** The split is a
real optimization variable, and local-error improvements are repeatable for
these down/gate cases. But the down improvement reverses under full-model KL,
while the small gate KL benefit comes with worse NLL. No tested split showed
a consistent improvement in both functional metrics across seeds.

A future allocator screen should measure candidate rank tuples at the tuned
operating point using functional KL **and** an NLL guard, not choose tuples from
weight or local activation NMSE alone. Repeat across earlier blocks before
generalizing. This experiment covers only block 25 and reduced local tuning;
it does not rule out benefits after full block tuning and model distillation.

## Implementation validation

Four focused CPU tests pass: equal-cost split accounting, paid-stage overhead,
explicit-rank fitting, joint updates to every stage, BF16/sign serialization
round-trip scoring, nonfinite rejection, and seed/budget-matched confirmation
pairing. Source-wide mypy passed (220 files) before unrelated concurrent source
edits. Full-suite and repository lint findings are recorded separately below.

The bounded full-suite recheck stopped after **1,266 passed, four failed, 53
deselected**. Three failures are Windows `os.link` access-denied errors in legacy
export adoption, run forking, and checkpoint hardlink tests. The fourth expects
a temporary launcher to be outside the repository; using a workspace-local
pytest temp directory (needed because safetensors could not write the default
sandbox temp path) makes that expectation false. Receipt:
`evidence/070/rank-split-regressions.xml`. The earlier unbounded attempt was
interrupted after the same four failures and a prolonged pause near its end.
No unrelated production behavior was changed to make these checks pass.

Focused Ruff passes for both tools and their tests. Repository-wide Ruff passed
earlier in this turn, but its final check found an import-order issue in the
concurrently modified, unrelated `src/nanoquant/kl_budget_workflow.py`; that
worktree edit was preserved. The repository-wide gates are therefore **not all
green**, even though the new CPU tests and both real functional probes completed.

## Reproduction

From the repository root, with no competing CUDA worker and new output paths:

```powershell
$splitRun = 'evidence/070/070-independent-residual-rank-2bpw-compress-and-benchmark-gemma-3-1b-it'
$splitSnapshot = 'C:/Users/pdykstra/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752'
.\.venv\Scripts\python.exe tools/validate_resident_run.py --run-output $splitRun --expected-blocks 26 --require-complete
.\.venv\Scripts\python.exe tools/probe_residual_rank_split.py --run-output $splitRun --snapshot $splitSnapshot --output evidence/070/rank-split-replay --blocks 25 --owners mlp.down_proj mlp.gate_proj mlp.up_proj self_attn.o_proj self_attn.attn_qkv
.\.venv\Scripts\python.exe tools/score_residual_rank_split.py --probe evidence/070/rank-split-replay --snapshot $splitSnapshot --output evidence/070/rank-split-down-replay --owner mlp.down_proj --ranks 608 1152
.\.venv\Scripts\python.exe tools/score_residual_rank_split.py --probe evidence/070/rank-split-replay --snapshot $splitSnapshot --output evidence/070/rank-split-gate-replay --owner mlp.gate_proj --ranks 864 1152
```

Run each command only after the preceding command completes successfully. Both
probe tools acquire the repository's cross-process CUDA lease. Existing output
directories are rejected rather than overwritten.
