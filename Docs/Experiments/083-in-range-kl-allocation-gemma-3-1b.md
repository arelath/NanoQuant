# Experiment 083: in-range KL rank allocation for Gemma 3 1B IT

083 tests the in-range solution from the 082 isolated-KL rerank. It uses the
same checkpoint, IMatrix, shared QKV, 256 × 2,048 calibration tokens, ADMM,
tuning, block refit, Q8_0 embeddings and disabled global KD as 082. The
decoder factor budget is 219,368,448 bytes. The pinned 130-unit plan spends
219,368,384 bytes. All 182 members stay within their sensitivity sweep's
error ranges under the fitted model.

The rerank predicts 2.805514 nats/token summed isolated KL, compared with the
measured 082 isolated sum of 4.075018. The comparison scores 080, 082 and 083
against the identical FP16 teacher on 4,080 token positions. Score joint
decoder KL with the teacher embeddings and full model KL with the shared
Q8_0 embedding. Verify the exported embedding payloads match byte for byte.
Compare WikiText-2 perplexity, six tasks and full MMLU under the same dense
benchmark protocol.

The launcher selects the content-addressed `Datasets/gemma-3-1b-it-kl-rank-plan-083`
dataset. Recreate it using `tools/rerank_staged_kl_budget.py` with
`--publish-in-range-dataset`. The experiment's run-local calibration receipt
materializes the 082 tokens under the 083 configuration identity.

```powershell
.\.venv\Scripts\python.exe experiments/083-in-range-kl-allocation-compress-and-benchmark-gemma-3-1b-it.py
.\.venv\Scripts\python.exe tools/compare_staged_rank_runs.py Results/082/082-imatrix-error-attribution.json 080 082 083 --output Results/083/083-quality-and-joint-kl-comparison.md
```
