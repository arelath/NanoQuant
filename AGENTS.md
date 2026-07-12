# NanoQuant Rewrite Agent Guide

## Objective

Bring this rewrite to measured behavioral and quality parity with the legacy NanoQuant implementation.
Passing the local test suite is necessary but not sufficient: parity must ultimately be demonstrated on the pinned
`google/gemma-3-1b-it` workload, including calibration, allocation, factorization, tuning, BPW, quality, memory,
resume behavior, artifacts, and relevant runtime behavior.

Do not mark the project or persistent goal complete while a required parity gate is supported only by a tiny fixture,
a reduced-iteration diagnostic, or an unfinished real-model run.

## Repository orientation

- `src/nanoquant/domain`: pure math, policies, typed states, and result contracts.
- `src/nanoquant/application`: orchestration-independent services and typed stages.
- `src/nanoquant/ports`: infrastructure interfaces.
- `src/nanoquant/infrastructure`: Hugging Face/model adapters, artifact stores, commits, execution, and resource control.
- `src/nanoquant/resident_quantization.py`: current real-model resident composition and resumable layer/block flow.
- `src/nanoquant/runtime`: deployment-only runtime surface; this remains less complete than the research pipeline.
- `tests`: unit, contract, and integration coverage.
- `tools/run_gemma_parity.py`: pinned Gemma resident parity launcher.
- `Docs/13-implementation-task-list.md`: authoritative milestone checklist; unchecked items are real remaining work unless
  later evidence explicitly supersedes them.
- `Docs/requirements-traceability.md`: requirements-to-milestone mapping.
- `Docs/14-artifact-retention-and-disk-usage.md`: measured design for bounded activation retention, shared stores,
  and store-aware garbage collection.
- `evidence/m4/README.md`: current Gemma resident-run evidence and comparison notes.

Read the relevant architecture/config/contract documents under `Docs/` before changing a boundary or persisted
schema. Preserve the dependency direction enforced by `tests/contract/test_architecture.py`.

## Legacy and reference implementations

- Legacy NanoQuant: `D:\dev\research\NanoQuant-OfficalCode`
- Modified llama.cpp reference: `D:\dev\research\llama.cpp`
- CUDA NanoQuant reference kernel: `D:\dev\research\llama.cpp\ggml\src\ggml-cuda\nanoquant.cu`
- Frozen legacy/reference provenance and Experiment 019 inputs: `evidence/m0/<LATEST>/`

Prefer direct old/new fixture or run comparisons over reimplementing behavior from memory. Experiment 018 is the
closest retained Gemma no-Hessian quality baseline; Experiment 019 and the implementation checklist define broader
replacement expectations.

## Python environment and validation

The global `python` may resolve to unsupported Python 3.7. Use the repository virtual environment explicitly:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src/nanoquant
```

The project requires Python 3.10 or newer. Before handing off code changes, run focused tests first, followed by all
three commands above when practical. Real CUDA findings need a CPU/tiny regression test whenever the failure can be
represented without weakening the real path.

## Pinned Gemma parity inputs

- Model: `google/gemma-3-1b-it`
- Revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Normal local snapshot:
  `C:\Users\pdykstra\.cache\huggingface\hub\models--google--gemma-3-1b-it\snapshots\dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Pinned calibration dataset: `evidence/m3/experiment018-calibration`
- Full Fisher state and preprocessing references are documented in `evidence/m4/README.md` and
  `evidence/m4/gemma-full-fisher-quantization/preprocessing.json`.
- The full tuned resumable run is under `evidence/m4/gemma-full-fisher-tuned-canary`.

The tuned run may be active in a detached, block-bounded process. Inspect its journal and process ownership before
starting any CUDA work:

```powershell
$j = 'evidence\m4\gemma-full-fisher-tuned-canary\state\journal.jsonl'
Get-Content $j -Tail 10
Get-CimInstance Win32_Process |
  Where-Object CommandLine -Like '*gemma-full-fisher-tuned-canary*' |
  Select-Object ProcessId, ParentProcessId, CommandLine
nvidia-smi
```

The journal, artifact descriptors, and completed block/layer commits are authoritative; console output is not.
Do not delete or rewrite valid evidence to make a rerun look clean.
Resident parity runs default to rolling activation retention: only the latest external resume generation is kept;
durable block results and frozen state remain valid after predecessor activation retirement.
When a semantic resident algorithm or numerical execution path changes, increment `RESIDENT_ALGORITHM_VERSION` in
`src/nanoquant/resident_quantization.py`; otherwise orphan discovery can adopt incompatible commits from a shared
artifact store.

## Long-running jobs and GPU safety

- NanoQuant calibration, factorization, tuning, evaluation, and real-model parity runs can take hours.
- When a known long-running job is healthy, prefer longer waits and less frequent polling instead of repeatedly
  checking it at short intervals. The user explicitly accepts response times longer than 60 seconds for these jobs.
- Preserve and monitor durable checkpoints/journals so interrupted work resumes without repeating completed stages.
- Never launch a second resident CUDA run until process inspection, the device lease, and `nvidia-smi` show that the
  intended device is free. Detached shell termination can leave Python child processes alive; verify descendants by
  command line rather than assuming the shell/tool result killed them.
- The cross-process lease is implemented in `src/nanoquant/infrastructure/device_lease.py`. Keep its true
  cross-process test when modifying it.
- Use `tools/cleanup_artifacts.py` for artifact reclamation. It is dry-run by default and preserves non-artifact
  evidence files; never manually delete content-addressed directories while a run is active.
- Preserve unrelated worktree changes. Much of the current dirty tree belongs to the ongoing parity effort.

## Git workflow

- Work directly on the current branch; do not create or switch branches for agent work.
- Use Git to commit each major feature after its implementation and proportionate validation are complete.
- Keep commits intentional and reviewable: inspect `git status` and the staged diff, stage only the feature being
  committed, and use a message that describes the completed behavior.
- Do not fold unrelated user changes into a feature commit. Never use destructive cleanup commands to manufacture a
  clean tree.
- Long-running evidence files may continue changing while source commits are made; commit only stable source,
  tests, tools, and documentation that belong to the completed feature.

## Current parity direction

The full-Fisher factor/outlier/scale run is complete but remains well behind legacy quality without tuning. The
resident path now implements legacy-style per-layer non-factorized tuning, factorized tuning, post-block refit,
mixed-precision trainable factor execution, deterministic replay on resume, and bounded block commits. The tuned
Gemma run is the current source of evidence for whether those changes close the quality gap.

After that run completes, the next required actions are:

1. Validate every committed artifact and assemble/load the frozen model.
2. Run the exact retained WikiText-2 limited protocol and compare with the BF16 and legacy baselines.
3. Compare ranks, effective BPW, layer/block losses, wall time, peak GPU/host memory, and artifact bytes.
4. Diagnose and implement any remaining algorithmic gap; model-level top-k KD is known to be represented in config
   but not yet implemented as a complete rewrite stage.
5. Update evidence and milestone gates only when the corresponding real comparison proves them.

## Performance work after parity

- Establish an accurate, protocol-matched Gemma parity result before starting broad performance optimization.
- Once correctness parity is demonstrated, run a dedicated profiling and performance pass. The current resident
  implementation is approximately 30% slower than the legacy reference on the comparable workload; treat that as
  the initial performance baseline, not an accepted final state.
- Compare identical model, calibration, ADMM, tuning, batch, device, and retention settings. Record wall time and
  stage-level timing so optimizations target measured bottlenecks and preserve numerical/quality parity.
