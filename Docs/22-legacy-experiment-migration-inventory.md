# Legacy numbered experiment migration inventory

## Scope and status language

This is the M9.1 inventory of every numbered top-level Python experiment in
`D:\dev\research\NanoQuant-OfficalCode` as inspected on 2026-07-15. The source hashes below freeze which legacy
files were classified. A replacement status describes behavior, not the presence of a migrated numbered runfile:
M9.2 remains open until thin, zero-argument rewrite runfiles exist.

- **Validated replacement:** the rewrite has real pinned-Gemma evidence for the principal behavior.
- **Implemented, needs recipe validation:** canonical configuration/math/runtime support exists, but the exact
  historical recipe has not completed a retained comparison.
- **Partial replacement:** important components exist, but a workflow/front end or model-size validation is missing.
- **Unsupported:** no safe claimed rewrite path exists yet.

No experiment is silently dropped. Historical ablations that are no longer recommended still receive an explicit
recipe/archive disposition so their chronology and intent remain discoverable.

## Inventory

| No. | Legacy purpose and distinguishing behavior | Rewrite replacement | Status | Remaining migration work |
| ---: | --- | --- | --- | --- |
| 001 | Baseline Gemma 3 1B compression; 256-sample UltraChat/WikiText online calibration and CUDA resident activations | Canonical historical recipe and zero-argument resident runfile over native immutable artifacts | Validated and migrated | Keep its original q/v/o/k/MLP order, 0.80/1.15 bounds, no-outlier path, and 1e-3 early-stop delta test green; the legacy executable pickle is deliberately not an output format. |
| 002 | Paired original-versus-NanoQuant short decode benchmark with CSV output | Runtime-bundle loader, `benchmark_runtime.py`, typed runtime metrics and comparison reports | Partial replacement | Add one paired benchmark application/CLI command and numbered runfile; current tools separately measure source, packed runtime, and llama.cpp. |
| 003 | Original-versus-quantized WikiText-2 plus small lm-eval smoke suite | Shared typed base/frozen evaluator, pinned task inputs, canonical recipe, zero-argument runfile, and retained real comparison | Validated and migrated | Keep the legacy eager-attention/BF16 scoring and choice-character `acc_norm` regression tests green; candidate quality is reported independently of evaluator parity. |
| 004 | Interactive Gemma chat over the custom GEMV path, chat-template history trimming, EOS/end-of-turn handling | Packed runtime generation, chat-template tokenizer assets, exact stopping and long-context cache | Partial replacement | Implement the interactive `chat` front end and history/context trimming policy; do not revive mutable train/runtime `NanoQuantLinear` mode switching. |
| 005 | Four-sample/128-token calibration-shrinkage proxy sweep | Canonical calibration statistics, shrinkage policies, cached calibration artifacts | Partial replacement | Add the reusable sweep driver and structured comparison output; no quantization rerun should be required per shrinkage point. |
| 006 | 1B compression with 0.1% BF16 Fisher-salient outliers outside the bit budget | Fisher outlier selector, typed outlier storage/cost, resident composition | Implemented, needs recipe validation | Add exact recipe/runfile and a real Fisher-outlier comparison; retained v28 evidence uses residual selection instead. |
| 007 | Cached WikiText-2 and six-task quality sweep across Gemma 1B/4B checkpoint families | Experiment 003's shared evaluator plus pinned six-task inputs, semantic caches, and campaign reporting | Partial replacement | Add the multi-model/cached sweep wrapper and complete real 4B evaluation; the validated 1B Experiment 003 path cannot satisfy the 4B gate. |
| 008 | 1B free residual-outlier recipe; BF16, all layers, 0.1%, no budget charge | Residual probe/outlier path, canonical recipe, and zero-argument runfile | Validated and migrated | Keep the exact 0.80/1.15 rank bounds, 1e-3 non-factorized early stop, and `charge_to_bit_budget=false` recipe-delta test green. |
| 009 | Budgeted residual outliers; selective attention/down projections, rowwise int8 values | Residual selector, layer patterns, charged bit cost, int8 outlier values/scales, packed runtime | Implemented, needs recipe validation | Run exact selective/int8 recipe and compare BPW/quality before promoting its runfile. |
| 010 | Gemma 3 270M free residual-outlier compression | Gemma3 adapter and size-independent resident pipeline | Implemented, needs recipe validation | Pin/download the 270M revision and run adapter, compression, quality, packing, and runtime canaries before claiming model support. |
| 011 | Standalone generation TPS benchmark with prompt, warmup, latency, memory, and JSON | Typed shared benchmark service, canonical recipe, production packed runtime, and zero-argument runfile | Validated and migrated | Keep the exact raw 12-token prompt, BF16 input/cache, one warmup, three 128-token repetitions, forced length, and generation-only timing test green. |
| 012 | Gemma 3 4B residual-outlier compression using CPU activations, small batches, cleanup, and top-k KD | Resource planner, pageable/mmap activation stores, block streaming/resume, top-k global distillation | Partial replacement | Complete a pinned 4B bounded-memory canary and verify quality/runtime; current 1B evidence cannot satisfy this model-size gate. |
| 013 | Improved 1B free-residual recipe with 0.90/1.10 rank bounds, MLP-first tuning, full batches, and no early stop | Complete resident parity mechanisms/evidence, canonical recipe, and zero-argument runfile | Validated and migrated | Keep its exact pre-Phase-1 tuning recipe and its documented delta from 008 under test. |
| 014 | Phase-1 tapered non-factorized tuning, post-block scale refit, 4K dense-Hessian sample | Per-position epoch schedule, post-block refit, dense-Hessian objective/workspace planning | Implemented, needs recipe validation | Execute the exact 4K-Hessian ablation against the diagonal v28 baseline and retain the structured weight/rank tables. |
| 015 | Phase-1 65K dense Hessian with sibling-input reuse | Dense-Hessian sampling/regularization, reuse policy, workspace rejection | Implemented, needs recipe validation | Validate exact 65,536-token recipe and memory plan; never silently fall back to a diagonal objective. |
| 016 | 65K Hessian safety correction: no sibling reuse, 20% diagonal blend, raw-error retry above cap | Hessian blend, independent sampling, retry thresholds/cap policy, diagnostics | Implemented, needs recipe validation | Migrate the recipe and run its safety comparison. Normalize the legacy filename containing ` copy` while preserving experiment number 016. |
| 017 | 256K Hessian safety recipe with sibling reuse | Dense-Hessian configuration and resource reservation | Implemented, needs recipe validation | Validate the 262,144-token objective under an explicit feasible resource plan. Legacy output/log constants incorrectly point at experiment 016 and must not be copied. |
| 018 | 1B phase-1 diagonal/no-Hessian recipe; closest retained quality/performance baseline | Complete v28 resident run, exact contemporary-legacy rank/trajectory/KD/PPL comparison, packed/runtime evidence; canonical recipe and zero-argument runfile | Validated and migrated | Keep the import-only recipe/request parity tests and shared resident/KD composition green. |
| 019 | Despite its filename, Gemma 3 **4B** phase-1 diagonal recipe with pageable CPU activations, bounded pinning, small batches, retry, reports, and KD | Streaming/resource architecture, activation retention/GC, phase-1 math/report contracts | Partial replacement | This is the critical 4B migration canary: pin the model/datasets, run interruption/resume and bounded-memory compression, evaluate, pack, and compare before adding a supported runfile. |

There are no unnumbered gaps between 001 and 019. Native rewrite runfiles now exist for validated Experiments 001,
003, 008, 011, 013, and 018. The `000_experiment_template.py` and frozen copies under `evidence/m0` are not migrations; every
other inventory row still needs either a tested runfile or an explicit unsupported/deprecated diagnostic.

The retained Experiment 011 migration result is `evidence/m9/011-generation-tps.json` (SHA-256
`e7933acba9014ae9adb9e2d456b9dd1c60a1e3bcd9ecf815192ce9c1327fe981`). It resolves the pinned Gemma revision,
loads the v28 production packed artifact, prepares all 182 linears with zero prefill/decode fallback, and reproduces
the historical 12-token prompt plus one-warmup/three-iteration/128-token BF16 protocol. Median complete-generation
throughput is 116.90 tokens/s and mean throughput is 110.18 tokens/s, versus the retained legacy GEMV mean of 22.50
tokens/s on the same named workload. This is a 4.90x mean-throughput improvement; it is a historical-workload
migration result, while M8's F32/F16 32-token campaign remains the release runtime comparison protocol.

The retained Experiment 003 result is `evidence/m9/003-gemma-3-1b-it-quality.json` (SHA-256
`a90f880b90dd91b957bf7a179d9941f0cbc8bf55ca061f8e0915347d5d1ee604`). On the base model, its 16×128
WikiText PPL is 94.8010 versus legacy 94.7989 (+0.0023%), while all primary 25-sample task metrics match exactly:
PIQA `acc_norm=0.72`, ARC-Easy `acc_norm=0.52`, and BoolQ `acc=0.76`. The comparison also revealed and fixed a
real evaluator bug: legacy lm-eval normalizes choice log-likelihood by character length, not token count. The v28
candidate result (PPL 396.573; task values 0.64/0.32/0.60) is retained as measured quality, not conflated with the
different legacy outlier checkpoint used by historical 003.

## Cross-cutting disposition

| Legacy mechanism | Rewrite disposition | Gate |
| --- | --- | --- |
| Repeated `.env`, tee, output-directory, load/save, and report code | Central run/session, event, environment, artifact, evaluation, and report services | M9.3 |
| Pickled `.pt` compressed checkpoints | Non-native and non-executable by policy; document import or require re-quantization | M9.5 |
| Custom GEMV/GEMM flags on mutable modules | Immutable prepared packed runtime with explicit workload plans | M9.17 |
| Modified llama.cpp NanoQuant GGUF | Validated export/converter and exact tensor/layout comparison already exist | M9.6 |
| CSV/Markdown weight and rank reports | Structured reconstruction/block results with derived reports | M9.16 |
| Inline Hugging Face orchestration | Model/dataset/tokenizer infrastructure adapters over application services | M9.10/M9.15 |

## Frozen source identities

| Experiment | Bytes | SHA-256 |
| ---: | ---: | --- |
| 001 | 4,617 | `5f2d6c6e4f83cf6e9a80a57d6d6bb61dd8e4c43b4fa159041de3b13bb9767dfc` |
| 002 | 5,520 | `d0b0c3c27cde752c88b126b8e75ab73216c5ee81d8a6cbdc1ed70ee6bf6126cf` |
| 003 | 8,825 | `7eb7844bff811158c3789da088110078ec751a807d59b0fb6cacce6f74a816ae` |
| 004 | 10,117 | `c085016d6f8376cf24512b8b17c64c5993462c662fa5cbf5a4bcef39f29cb207` |
| 005 | 4,548 | `d989d7ad8c676147f9e178be485ea6a267bb29a21a78426ab100f498392385be` |
| 006 | 4,953 | `b284a9856fe708e55c49b2c32216a3752ac8e0a9d9b9c1aa163f79db3d6ad7fb` |
| 007 | 20,011 | `b490a822d3ca29520814b0f4d2cbad5e7eb321255cc74b2fe45629f9158ca789` |
| 008 | 6,188 | `dae7693e8ad074ccb1ec8b9f5393b3d37efaefbf9d9f95eb2208b5d0169a9cc2` |
| 009 | 5,884 | `0b35f2f7aa373e15fd37359da34385353552ad93488375889fcda8c1512d3443` |
| 010 | 4,922 | `0ff13aaa08b989770342ff80a0e2e3ebf03fab2415e643852da96ceac174c145` |
| 011 | 10,484 | `2cf5ecf53b5dd61c9ba5d8547149756c16dede2998524f165769ef6675a03293` |
| 012 | 7,318 | `794f6154f38fc8771638fdead4a799058c61819caa4d6f92261ee136bdd075be` |
| 013 | 6,190 | `9dcc702d9768dc3ffb64c4a60ededd672ad963cb6ddd160e1670ec4789db6a84` |
| 014 | 7,853 | `f406a146353e07d10b671775ea3120adff832c717aa94ee6d71fffeee329ff23` |
| 015 | 8,153 | `2a452887226155cf54f1a5c4f29c84d8c4e2a5a36fe08ad0b79a774e59fed6f7` |
| 016 | 8,665 | `f5d2b569194dc319530ba1d3bfa3cda2fa612008cb85605b0545626d2343ccac` |
| 017 | 8,669 | `81fd6f702f38d34c62c4b7d80a646cb3781abbee43250718d4b01251d867490c` |
| 018 | 8,637 | `53606ddcba460e96c8dac2d2753e282b7f875020d26f2368f216b4e6fdcb134f` |
| 019 | 9,953 | `334146426e95f733d70c9fc0cd68a38d17841a4833a5ac19690911234870d407` |

## Migration order

1. Compose the 002 paired benchmark and finish 007 only with its required cached 1B/4B sweep.
2. Add the 005 sweep and 004 chat front ends without moving business logic into runfiles.
3. Validate and migrate the unproven 006/009 and 014–017 ablations only when their comparison is useful.
4. Run 010 (270M) and then the critical 012/019 4B bounded-memory canaries before declaring those model workflows
   supported.

The inventory is complete when every row either has a tested thin replacement or an explicit unsupported/deprecated
diagnostic. It does not authorize M9.GATE until the supported rows have those tested paths.
