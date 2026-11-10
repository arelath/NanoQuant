# Experiment 070 residual rank replacement for Experiment 060

Date: 2026-11-10

## Objective and matched recipe

Run a full Gemma 3 1B compression experiment matching 060, replacing its charged INT8 residual columns with **independently scaled binary residual stages**, as confirmed by the user. Total component rank may exceed `min(rows, columns)`; capacity is funded by the existing 2.0-BPW budget rather than stopped at the algebraic dimension.

The retained resolved 060 configuration is frozen in `experiments/recipes/references/experiment060.json`. Its normalized config hash is `sha256:3a48eafd5f6060c0e7617bec0ec94187310bee4990c7ea0c96cf147c05b2058a`; a regression check verifies exact equivalence with the retained reference. Keep its model/tokenizer revisions, 50/50 UltraChat–WikiText calibration mixture and formatting, 256-sample online Fisher/shrinkage, same-run tuned D2 allocation, 800-step ADMM, shared QKV/V importance, eight-epoch block binary tuning, post-block refit, and eight-epoch top-64 distillation. The recent full-KL continuation is not part of this experiment.

Keep 060's 2.0-BPW target, 0.02-BPW retry reserve, rank alignment 32, allocation floor and budget-relative measured response range. Remove its physical-rank ceiling and its 8.8% residual-column allocation, including column-increase retries. Reinvest the freed capacity in signed factors and every residual stage's own pre/mid/post scale vectors. The normal D2 allocation trust range remains budget-relative; there is no `min(rows,columns)` ceiling on combined component count.

## Numerical representation

For total allocated component count R, partition into consecutive stages, each no wider in rank than the aligned physical dimension, with any remainder in the final stage. This is a fitting decomposition, not a cap on total R: additional funded components create additional stages.

Fit stage 0 to the target, then each next stage to the residual after all preceding stages. Use the same importance objective and configured ADMM/scale fitting for each stage. Each stage has independent row, component and column scales. Execute the sum of the stages and tune them jointly under 060's existing block/global objectives. Do not concatenate independent outer scales into a single shared-scale factor.

Charge the exact sum of all stage factor costs, including all scale vectors and alignment. Crossing a stage boundary incurs a new pair of outer scale vectors. Planning, measured response allocation, retries, frozen statistics and exported payload accounting must agree. Existing single-stage configurations and artifacts retain their behavior.

## Integration and acceptance checklist

- [x] Freeze and verify the retained 060 numerical/data configuration.
- [x] Confirm independent residual stages as the requested representation.
- [x] Add typed stage state and exact budget/rank decomposition policies.
- [x] Integrate sequential fitting, joint tuning, shared QKV, freeze/load and resumable validation.
- [x] Extend logical/packed artifacts and checkpoint/GGUF export with explicit stage inventories.
- [x] Compose native residual outputs using existing NanoQuant operations and addition; preserve single-stage fast paths.
- [x] Add 070 launcher, matched D2 control/profile campaign and exact recipe-difference tests.
- [x] Pass focused numerical, accounting, persistence, architecture, runtime/export and resume tests; run the full test/lint/type gates.
- [x] Run a real block-bounded CUDA canary with fresh run-owned calibration and validate its commits.
- [ ] Complete the full compression through `execute_complete_compression`, fresh artifact validation, GGUF export and the matched quality protocol.
- [ ] Compare effective/stored BPW, per-task results, WikiText perplexity, memory and time against retained 060, documenting any execution-protocol differences.

The retained 060 baseline has WikiText perplexity 115.668625 and six-task primary-metric mean 50.2167% at task limit 1,000. Its run store is absent; its GGUF and quality report remain. Use per-task `primary_value` fields to derive the mean, rather than an unrelated aggregate. A new D2 control/profile is required for the new representation and provenance; do not relabel the old missing profile as a matched control.

A full run is not complete at a dense overlay, a tiny fixture, or a partial resident checkpoint. New residual artifacts must fail closed in old readers and validate all stages in updated readers. Increment the resident algorithm version before any new numerical path can produce or adopt commits.

## Numbering provenance

Experiment 070 is the next unused experiment number. Earlier analysis-only numbers have retained evidence without formal compression launchers: Experiment 061 covers held-out checkpoint selection (`evidence/061`), Experiment 062 covers coordinate binary QAT (`evidence/062`), Experiment 063 covers adaptive distillation targets (`evidence/063`), and Experiment 064 covers functional-proxy and structured-sidecar screening (`evidence/064`). Experiments 065–069 have their own analysis documents. These numbers are not reused by the new compression launcher.

## Implementation verification and run status

The resident algorithm version is 66. Configuration hashing omits the disabled residual policy, preserving the frozen 060 reference identity. Allocation and retry decisions charge every stage, including outer-scale overhead at stage boundaries. CPU tests cover ADMM residual targeting, joint gradients, independent scale persistence, shared QKV views, block interruption/resume, global distillation interruption/resume, packed model replay, and committed bit accounting.

Runtime 0.2.0 uses an explicit `nanoquant-residual-v2` logical specification with residual rank inventories. Packed words retain their existing per-term layout. Residual converter checkpoints use schema 2; GGUF uses NanoQuant version 3 with explicit stage counts and distinct stage-zero tensor names. The old native loader was tested against the fixture and rejected the missing legacy primary/dense contract. New native inference adds existing per-term operations. A CPU smoke test exposed and fixed the existing raw-I32/CPU-repack buffer mismatch.

Four real CUDA kernel checks pass for FP32/BF16 inputs at one and seven tokens. A synthetic one-block Gemma GGUF containing independent ranks 64 and 32 passes native CPU execution: full Python/native logit RMSE 0.00115805, maximum absolute difference 0.00574988, versus RMSE 0.06357598 when the residual term is omitted. Checkpoint, converter and GGUF tensor inventories and values validate exactly. These are correctness fixtures, not model-quality evidence. Compact receipts and a reproducible native source patch are in `evidence/070`; the actual native base is `2927569f318311aded148d76be6b1109664066f5`, separate from the historical per-term word-layout reference. No native-repository commit was made.

The first real control process completed its one-block canary on 2026-11-10 using fresh run-owned calibration. Its logs are `evidence/070/canary-control-block.*.log`. After the fresh hash and bit-accounting audits passed, `tools/run_experiment070_campaign.py` resumed the full campaign at 13:42 Pacific. Resume was observed training block 1 using the retained calibration and block-0 checkpoint. Controller logs are `evidence/070/campaign.*.log`; each process has separate `campaign-slice-NNN.*.log` files. The full campaign has no quality result yet; the retained 060 values above remain the comparison baseline.

The final pre-run CPU suite passed 1,549 tests (53 CUDA/opt-in tests deselected); the four residual CUDA tests were run separately and passed. Ruff and mypy passed (220 checked source files). The broader cold native CUDA rebuild was stopped after the native CPU and Python packed-CUDA correctness gates passed, keeping resources on the requested real-model campaign; native GPU model execution is not claimed by this canary.

The real control canary completed block 0 in 492.83 seconds after calibration. Fresh validation checked 34 immutable artifacts (3.04 GB), all six journal records, and the saved activation generation. The block represents 26,836,992 quantized weights at **1.9769345 BPW**: three MLP units have independent ranks **1,152 + 768**, QKV has **1,152 + 96**, and attention output uses one rank-1,024 term. Every frozen unit's cost independently matches the staged bit policy, including BF16 scales; combined rank is 8,032. The initial console events report primary rank only; subsequent workers and validation summaries report combined rank. No evidence log was rewritten. See `control-block0-validation.json` and `control-block0-stage-audit.json` in `evidence/070`.

The standard GGUF export helper was also exercised on the synthetic fixture through Q8 embedding quantization and its durable export receipt. Its six BF16 scale arrays and all stage words validated exactly, and the quantized GGUF executed in the native CPU probe. This verifies the export lifecycle rather than model quality.
