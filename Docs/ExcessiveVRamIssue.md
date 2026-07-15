## Root cause

The VRAM blowout isn't in the factorization/tuning math — that path is carefully microbatched and streams activations through pinned CPU buffers. It's the **inline quality evaluation** in [resident_quantization.py](../src/nanoquant/resident_quantization.py), which is on by default (`evaluate_inline_quality: bool = True`):

1. **Setup** ran `model(input_ids=quality_tokens)` over the *entire* quality token set in a single forward, materializing full-vocabulary logits of shape `samples × sequence × vocab`. For a 1B model this dwarfs everything else: 16 samples × 2048 tokens × 128K vocab in fp32 is **≈16.8GB in one allocation** — which matches the 16GB you're seeing. (`quality_token_ids` also defaults to the whole calibration token set, so this grows with your calibration data.)
2. That `reference_logits` tensor was then **held on the GPU for the entire block-compression loop**, so even when the forward fit, several GB of your 12GB card were squatted on while every block factorized and tuned.
3. Finalize repeated the same single-shot forward for the compressed model, then made *additional* full-size fp32 copies inside `_nll`/`_mse`.

## Fix (no meaningful speed cost)

- New `_run_quality_logits_batched` — the reference forward now runs one sequence at a time and parks logits in **pageable CPU memory**, so the GPU holds at most one sequence of logits, and nothing quality-related occupies VRAM during block compression.
- New `_streamed_quality_metrics` — finalize runs the compressed forward per sequence and computes all four metrics (reference/compressed NLL, logit MSE, argmax agreement) in 256-position fp32 chunks, so peak device memory is independent of sample count and vocabulary size. Worst case per iteration is now ~3GB of transients instead of multiples of the full logits tensor.
- Removed the now-dead `_nll`/`_mse` helpers.

The quality forwards happen twice per run and are dominated by the same matmuls as before, so wall-clock impact is negligible against a multi-hour compression; the block loop itself is untouched (and now has more free VRAM). Expected peak during compression drops to roughly model shell + one forward batch + ADMM workspace — comfortably inside 12GB for a 1B model.

## Verification

- All 7 integration tests in `test_resident_quantization.py` pass — including the ones that independently recompute NLL from the packed model and compare against my streamed accumulation at `rel=1e-6`, and the resume-parity checks.
- `test_global_distillation.py`, `test_resident_batching.py`, `test_resident_config_identity.py`: 20 passed. `mypy` clean.

## You should know

The reference logits now cost **host RAM** instead of VRAM (`samples × seq × vocab × 2` bytes). Since `quality_token_ids` defaults to the full calibration set, a large calibration run should still pass an explicitly smaller quality set (as `run_gemma_parity.py` already does) — otherwise you'd trade a VRAM OOM for tens of GB of host RAM.

## Follow-up: WDDM shared-memory pressure

The inline-quality fix remained correct, but a later performance optimization introduced a separate Windows
memory problem. Commit `875e165` pinned every complete resident activation stream (about 1.2 GiB each for the
Gemma parity workload), and `20003b1` made every block round-trip its boundary through the artifact store and pin
the reloaded streams again. PyTorch caches freed pinned-host blocks. WDDM continues to charge those cached,
GPU-addressable blocks as shared GPU memory, so Task Manager showed roughly 18.7 GiB total GPU memory with about
10 GiB shared even while CUDA's own reservation remained bounded.

Version 28 fixes the problem at its source:

- complete teacher/compressed activation streams are pageable CPU tensors;
- only two batch-sized host slots are pinned for ordered H2D prefetch;
- shuffled tuning uses the same bounded staging policy for pageable inputs;
- unused pinned-host cache is released after every durable block commit;
- Windows resource samples include per-process WDDM dedicated/shared current and peak bytes.

A controlled 512 MiB pinned allocation stayed at 0.57 GiB WDDM shared after its tensor was deleted and fell to
0.07 GiB after the new cache release. With two full Gemma-sized pageable streams (2.25 GiB total), the bounded
batch path used 0.326 GiB WDDM shared before release and 0.074 GiB after release. CUDA bitwise tests cover block
forward, block loss, ordered device batching, and factorized tuning under the new policy.

## Real-model regression result

The v28 four-block pinned-Gemma canary confirmed the fix under the production workload. Per-process WDDM shared
memory peaked at 622,854,144 bytes (0.580 GiB), and the explicit release after each of blocks 0--3 returned it to
83,886,080 bytes (80 MiB). Peak host working set fell from 15.153 GiB in v27 to 10.654 GiB; peak CUDA reservation
was unchanged at 6,236,930,048 bytes. The artifact/journal validator passed all 153 reachable artifacts, and the
four block losses were within 0.80% of contemporary legacy at every boundary.

This is now guarded at three levels: unit tests exercise the pinned-host cache release and WDDM counters, a CUDA
integration test requires multi-block resident activation sources to remain pageable and emits one release event
per block, and real runs retain current/peak WDDM dedicated/shared bytes in their resource events.

The subsequent full 26-block continuation preserved the same bound: shared memory still peaked at 0.580 GiB and
all 26 cache-release events returned it to 80 MiB. Peak CUDA reservation rose normally with later, larger blocks to
7.107 GiB, while peak host working set remained 11.908 GiB. The complete 979-artifact graph passed strict validation
after unreachable training scratch was collected.
