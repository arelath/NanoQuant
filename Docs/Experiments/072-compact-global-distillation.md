# Experiment 072: compact global distillation on 12 GiB

The control committed all 34 blocks and all eight teacher-cache epochs before
slices 42 and 43 failed at the initial student transfer. The fixed signs alone
would occupy approximately 15.8 GiB in BF16, before any training workspace.

Global CUDA distillation now keeps immutable plain sign factors in one-bit
buffers when their expanded BF16 footprint alone reaches the physical device
capacity. Each term decodes to the input dtype for the existing factorized
arithmetic and recomputes on backward rather than retaining decoded signs for
all terms. Scale and norm training, parameter names, optimizer, ranks, teacher
cache identity, and existing committed block identities are unchanged. The
freezer decodes the original signs and publishes the usual logical artifacts;
there is no change to the persisted format or the resident quantization algorithm.
072 keeps its existing dense packed-model quality replay.

A fresh pre-resume audit validated all 34 blocks and 1,183 artifacts
(`evidence/072/pre-compact-distillation-validation.json`). The first
resume attempt (slice 44) stopped with `VRAM002` because the WDDM
shared-memory meter was unavailable in the launch environment. It did
not reach distillation. The guard was not disabled, and the real-model
CUDA training peak remains unverified. Once the meter is available,
resume with the unchanged `tools/run_experiment072_campaign.py` command
after confirming the device and run lease are free.

Validation: 44 focused tests passed, including bitwise CPU forward and
gradient comparisons and compact-path checkpoint/resume with residual
stages. Ruff and mypy passed. Repository pytest stopped at the known
Windows hardlink access denial in legacy GGUF export adoption, after
296 passes and 54 deselections.

## Host activation offload throughput follow-up

**Superseded by the VRAM-first correction below.** This section records the
rejected implementation and its misleading isolated performance result.

Slice 45 reached training but was interrupted at step 83/2048 after 8m33s.
The original compact path nested factor-term checkpointing inside HF decoder
checkpointing: backward recomputed the block and then the individual terms.

Compact global KD now disables HF block checkpointing and wraps the student
hidden-state forward in `torch.autograd.graph.save_on_cpu(pin_memory=True)` on
CUDA. Attention/MLP saved tensors and term-checkpoint inputs live in host RAM.
Term checkpointing remains enabled so decoded sign matrices are not retained
across the model. Ordinary noncompact KD is unchanged. This trades host RAM
and activation transfers for whole-block recomputation, without changing ranks,
loss, optimizer, batches, teacher cache or stored checkpoint representation.
It does not change resident quantization, so its algorithm version is unchanged.

`tools/benchmark_compact_distillation.py` acquires the device lease and compares
one 512-token BF16 2560->2560->10240 factor term, with three warmups and ten
timed forward/backward iterations. On the 12 GiB RTX 4000 Ada:

| Execution | Mean milliseconds | Peak allocated bytes |
| --- | ---: | ---: |
| Nested block/term checkpoints | 6.927 | 155,253,760 |
| Block checkpoint only (diagnostic) | 4.903 | 155,274,240 |
| Host activations + term checkpoint | 5.181 | 147,389,440 |

Outputs and input/scale gradients matched bitwise. Host activation offload was
25.2% faster in this isolated probe; this is **not** a full-model speedup claim.
An exploratory decoded-sign host cache was slower (12.944 ms) and was removed.
The full 4B host-memory peak and sustained throughput still require a real run.
No existing worker was killed and no production campaign was started here.

Focused tests: 11 passed, covering FP32/BF16 forward/gradient/export parity
with CPU saved-tensor offload and compact KD checkpoint/resume. The initial
sandbox temporary-directory access failures were avoided with a fresh
repository-local pytest base directory.
Ruff and mypy passed. The broad repository pytest run was manually stopped
while still progressing, without a reported failure; it is not a full-suite
pass. Git staging was denied at `.git/index.lock`, leaving this work uncommitted.

## VRAM-first correction after slice 46

Real-model evidence rejected automatic host activation offload: slice 46
progressed from step 5 at 57s to step 41 at 402s (~9.58s/step), whereas slice 45
progressed from step 26 at 137s to step 56 at 310s (~5.77s/step). Both were
interrupted; neither is a completed quality result. The single-layer speedup
did not predict model-wide transfer overhead.

Global KD no longer calls `save_on_cpu`. Small models keep ordinary resident
factors and the existing configured checkpoint behavior; compact eligibility
is unchanged. Large compact students use reentrant HF block checkpointing
when configured, with per-term checkpointing disabled only after confirming
the model enabled block checkpointing. Initial block forwards run without a
saved graph; only the block being recomputed for backward retains its decoded
factors. This keeps activations on the GPU and removes nested recomputation.

After student transfer, the policy compares the largest block's expanded BF16
sign footprint (including residual terms) plus a conservative 2 GiB workspace
margin against free CUDA memory. Insufficient headroom retains nested
checkpointing, without CPU activation transfers. This margin is a heuristic,
not a measured peak guarantee. It does not attempt to fill VRAM with a global
decoded-factor cache. No rank, objective, optimizer, cache identity or resident
algorithm change is made.

Regression tests cover small-model exclusion, unsupported block checkpointing,
headroom boundaries, BF16/FP32 block-only output and gradient parity including
residual terms, and compact HF checkpoint/resume. Real 072 throughput and peak
memory for this correction are still unmeasured; no production restart was
performed during this correction.

## Measured step profile and throughput changes after slice 47

Slice 47 ran the VRAM-first path at about 3.5 s/step for its first ten steps
and about 4.3 s/step once the 110 W GPU had warmed up, reserving 5.2 GiB of
12 GiB. A read-only profile of the real 072 student (2048 tokens, batch 1,
synchronised timers) showed the step is compute-bound, not memory-bound:

| Part of a 3.77 s step | Seconds |
| --- | ---: |
| Factor GEMMs (forward, checkpoint re-forward, backward) | ~1.9 |
| Sign decode, twice per step | 0.76 |
| Eager attention, norms, autograd | ~1.1 |
| Loss and optimizer | <0.01 |

Three changes followed. Timings are from the same profile on the real student.

| Change | Step seconds | Numerics |
| --- | ---: | --- |
| Baseline (slice 47 code) | 3.77 | - |
| Two-kernel sign decode | 3.46 | bitwise identical |
| Resident decoded signs in measured spare VRAM | 3.28 | bitwise identical |
| One shared-input group evaluation per input | 2.80 | forward identical, gradient rounding differs |

Without timers the final configuration ran at 2.60-2.73 s/step over twelve
steps. Sustained, thermally throttled throughput is still unmeasured, and no
production slice has been run with these changes.

**Sign decode.** `unpack_sign_bits` lifts each inverted bit into the sign
position of its own byte and lets `copysign` emit +/-1: two passes over the
output instead of five. Losses matched to every digit over six real steps.

**Resident signs.** After the student is on the device, one rehearsed training
step (parameters, gradients, trainability, mode and RNG all restored) measures
the real working set. Spare memory beyond it, less 1.5 GiB of headroom, keeps
decoded signs resident for the smallest packed terms first. On 072 this was 158
of 340 terms (4.4 GiB), a 9.6 GiB reserved peak and 1.25 GiB left free. This
replaces the earlier statement that the policy does not fill VRAM with decoded
factors; the 2 GiB block-expansion heuristic above still only selects block
versus nested checkpointing. The headroom was raised from 1 GiB after training
left 0.85 GiB free where the rehearsal predicted 1 GiB. Other processes growing
by more than the headroom plus the 0.75 GiB WDDM allowance would still abort
the run, which then resumes from its last epoch checkpoint.

**Shared-input group evaluation.** `q_proj`, `k_proj` and `v_proj` views each
evaluated their stacked owner, so the QKV factor ran three times per pass and
was back-propagated three times. Global distillation now shares one evaluation
among the members that receive the same input tensor. Forward values and the
step-0 loss are unchanged. Parameter gradients are the same sum, but accumulated
in one backward pass, so BF16 rounding differs and later losses diverge from the
per-member path. On a synthetic float64 comparison the two agreed to 7.5e-16,
and the shared BF16 gradient was slightly closer to the float64 reference (0.52%
against 0.62% relative error on the mid scale). The switch is opt-in on
`install_trainable_group`: block tuning and the probe tools keep the per-member
path, so `RESIDENT_ALGORITHM_VERSION` is unchanged. Distillation checkpoints
written before this change would resume under the new rounding; 072 has none.

Validation: 1726 tests passed with 54 deselected; Ruff and mypy passed.
