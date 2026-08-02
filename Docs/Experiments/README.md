# Experiment archive

This directory is the durable record of NanoQuant Rewrite experiments 001 through 034. It records the question each
experiment asked, the method used, the retained result, and the lesson that should survive removal of the Python
launchers.

Statuses are evidence-based:

- **Completed** means the configured workflow produced its terminal artifacts and measurements. It does not mean the
  compressed model met BF16 quality.
- **Partial** means preparation or compression ran, but the experiment did not reach a valid terminal quality result.
- **Campaign** means the number covered multiple diagnostic arms rather than one launcher.
- **Not run** means a launcher was designed but no run evidence was retained. These pages document intent, not a result.

Metrics are only comparable when their evaluation protocols match. In particular, Experiment 024 used a longer
task-evaluation protocol, and Experiments 028 and 030 added Qwen-specific deployment or behavior checks.

| Experiment | Status | Model | Primary lesson |
| --- | --- | --- | --- |
| [001](001-gemma-3-1b-parity.md) | Completed campaign | Gemma 3 1B IT | Rewrite/legacy parity was reached, but the shared recipe remained far behind BF16 quality. |
| [002](002-gemma-3-1b-benchmark.md) | Completed | Gemma 3 1B IT | A completed candidate is not an accepted candidate; explicit baselines and gates matter. |
| [003](003-gemma-3-4b-baseline.md) | Completed | Gemma 3 4B IT | Bounded-memory 4B execution worked, while quality remained materially degraded. |
| [004](004-gemma-3-4b-vproj-plus30.md) | Completed | Gemma 3 4B IT | Better post-training reconstruction did not improve end quality. |
| [005](005-gemma-3-4b-vproj-max-rank.md) | Completed | Gemma 3 4B IT | A much larger v-projection rank still failed to improve perplexity. |
| [006](006-gemma-3-1b-projection-ranks.md) | Completed | Gemma 3 1B IT | Projection-specific ranks helped versus 002, but did not close the BF16 gap. |
| [007](007-gemma-3-270m-rank-transfer.md) | Completed | Gemma 3 270M IT | The 1B rank policy transferred operationally, not in quality. |
| [008](008-gemma-3-12b-scale-up.md) | Partial | Gemma 3 12B IT | Large-model execution still needed more adaptive memory management. |
| [009](009-gemma-3-270m-publish.md) | Completed | Gemma 3 270M IT | Compression/export/publication worked; run identity and rollover remained important. |
| [010](010-gemma-3-270m-cubic-update.md) | Completed | Gemma 3 270M IT | The cubic update improved the 270M result but left a large quality gap. |
| [011](011-gemma-3-1b-int8-outliers-0.2.md) | Completed | Gemma 3 1B IT | A small increase in INT8 outliers was an economical quality improvement. |
| [012](012-gemma-3-1b-int8-outliers-2.0.md) | Completed | Gemma 3 1B IT | More outliers improved perplexity at too large a BPW cost for a 1-BPW target. |
| [013](013-gemma-3-270m-stacked-qkv.md) | Completed | Gemma 3 270M IT | Shared-input QKV factorization was not a quality win by itself. |
| [014](014-gemma-3-270m-reconstruction-allocation.md) | Completed | Gemma 3 270M IT | Pure reconstruction allocation starved functionally important layers. |
| [015](015-gemma-3-270m-architecture-protection.md) | Completed | Gemma 3 270M IT | Architecture priors recovered much of the allocator regression. |
| [016](016-gemma-3-270m-stronger-protection.md) | Completed | Gemma 3 270M IT | Better protected-cohort reconstruction still did not guarantee better language quality. |
| [017](017-gemma-3-1b-tempered-allocation.md) | Completed | Gemma 3 1B IT | Tempered architecture-aware allocation established a strong low-BPW 1B baseline. |
| [018](018-gemma-3-4b-tempered-transfer.md) | Partial | Gemma 3 4B IT | The transfer run stopped too early to support a quality conclusion. |
| [019](019-llama-3-2-1b-first-transfer.md) | Partial | Llama 3.2 1B Instruct | The first cross-architecture run exposed execution work but produced no quality result. |
| [020](020-error-budget-and-kl-campaign.md) | Campaign | Gemma 3 270M IT | Corrected KL semantics found a real signal; several attractive local ideas failed equal-budget gates. |
| [021](021-gemma-3-270m-d2-kl.md) | Completed | Gemma 3 270M IT | D2 improved perplexity at lower BPW, but not the aggregate task score. |
| [022](022-gemma-3-1b-d2-kl.md) | Completed | Gemma 3 1B IT | Exact-unit D2 produced the clearest low-BPW 1B improvement. |
| [023](023-gemma-3-1b-d2-interaction.md) | Completed | Gemma 3 1B IT | Interaction correction added complexity without a clear net win. |
| [024](024-gemma-3-1b-best-methods.md) | Completed | Gemma 3 1B IT | Combining plausible improvements did not beat the simpler D2 recipe. |
| [025](025-llama-3-2-1b-replication.md) | Completed | Llama 3.2 1B Instruct | Cross-architecture compression worked end to end, but quality remained far from BF16. |
| [026](026-llama-3-2-3b-scale-up.md) | Partial | Llama 3.2 3B Instruct | Preparation completed, but no compression or quality conclusion followed. |
| [027](027-llama-3-8b-planned.md) | Not run | Llama 3 8B Instruct | This was a planned adaptive-memory scale gate, not empirical evidence. |
| [028](028-qwen3-0-6b-baseline.md) | Completed | Qwen3 0.6B | Deployment worked, while generic evaluation missed the thinking-mode failure. |
| [029](029-qwen3-8b-planned.md) | Not run | Qwen3 8B | Serial llama.cpp evaluation was designed, but never measured. |
| [030](030-qwen3-0-6b-thinking-recovery.md) | Completed | Qwen3 0.6B | Teacher traces alone did not recover either mode; the relative mode guard was insufficient. |
| [031](031-qwen3-8b-thinking-confirmation-planned.md) | Not run | Qwen3 8B | The planned scale confirmation has no empirical result. |
| [032](032-gemma-3-1b-raw-fisher-d2.md) | Completed | Gemma 3 1B IT | Raw Fisher won the static KL screen but regressed retained perplexity by 7.22%, so it was rejected. |
| [033](033-gemma-3-1b-covariance-refined-d2.md) | Completed | Gemma 3 1B IT | Pre-tuning covariance refinement improved its local objective but regressed retained perplexity by 19.26%, so it was rejected. |
| [034](034-gemma-3-1b-post-refit-qkv-covariance.md) | Completed | Gemma 3 1B IT | Selected post-refit QKV refinement passed resume/export validation but regressed pre-KD perplexity by 4.59% and final perplexity by 5.50%, so it was rejected. |
| [049](049-cyclic-scale-rank.md) | Rejected at screen | Gemma 3 1B IT | Cyclic pre/post scale banks added tiny fixed-rank capacity but regressed all nine equal-bit comparisons by 4.07–14.76%. |
| [050](050-tiny-factorization-optimality.md) | Completed diagnostic | Gemma 3 1B IT + synthetic | Exhaustive 3x3 search found large same-format optimizer gaps; exhaustive 10-bit row/column moves improved one of three real 10x10 crops by 3.12%. |

The original launchers were under `experiments/`. Retained measurements remain under `Results/` and `evidence/`;
architecture and decision records remain elsewhere in `Docs/`.
