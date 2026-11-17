# Curated sensitivity comparison after Experiment 083

Measure whether the sensitivity estimates and joint ranking change with a larger,
curated text sample. Use the existing curated Gemma IMatrix calibration corpus
at `Datasets/gemma-3-imatrix-calibration-v1/calibration.txt`, with SHA-256
`3d8fa1e99d56ca7ba8455a1aabd5b6d10a82ca1a7d502f7fb15450cba290e301`.
Its manifest records 1,000,294 tokens across 15,449 records. Tokenization through
the pinned Gemma tokenizer preserves the existing native turn and BOS/EOS tokens.

Sample 160 disjoint windows, each consisting of a BOS and 255 corpus tokens,
with seed zero. This gives 40,800 scored positions per perturbation, exactly
ten times the prior 4,080. Windows can cross record boundaries, matching the
existing stream-window protocol. The new teacher cache has its own dataset
identity and approximately 20 GiB of retained FP16 log probabilities.

First compare joint decoder KL for the existing 080, 082 and 083 exports on the
curated 160 windows and a nested 16-window subset selected with seed 83.
Compare their rankings with the saved original-corpus results.

Then run the full sensitivity sweep with all 182 individual projections,
relative Gaussian noise levels 0.01, 0.025, 0.05, 0.10, 0.20 and 0.35, and
seeds 0, 1 and 2. All 3,276 trials persist per-window KL. Fit the usual
power-law curves to the 160-window means and to the nested 16-window means.
This tests sample-size stability within the curated corpus and provides a
comparison with the original 16-window corpus. The sample-size and corpus
changes are both present when comparing the old sweep to the new full sweep.

The model, precision, noise generation, embeddings, arithmetic and 256-token
context length follow the prior sensitivity protocol. Corpus provenance and
coverage limitations are documented in [the corpus report](../gemma-calibration-corpus.md).

```powershell
.\.venv\Scripts\python.exe tools/build_gemma_3_1b_sensitivity_dataset.py --calibration-text Datasets/gemma-3-imatrix-calibration-v1/calibration.txt --samples 160 --sequence-length 256 --teacher-dataset Datasets/gemma-3-1b-it-fp16-sensitivity-curated-160x256 --local-files-only
```

Reuse this invocation to resume completed noise trials. The full sweep can take
many hours. Logs are under `outputs/curated-sensitivity-160x256`.

## Initial joint-model comparison

The curated teacher dataset completed on 2026-11-17. Joint decoder KL is measured
with clean embeddings on the same windows for all three existing exports.

| Experiment | Original corpus, 16 windows | Curated nested 16 | Curated 160 |
| --- | ---: | ---: | ---: |
| 080 | 1.369504 | 0.764553 | 0.747412 |
| 082 | 1.413458 | 0.756072 | 0.761960 |
| 083 | 1.379170 | 0.770809 | 0.786708 |

080 leads on the full curated sample. Its ranking relative to 082 reverses in
the nested 16-window subset. On 160 windows, 083 minus 080 is +0.039297
nats/token, with a paired window standard error of 0.007966. Absolute KL values
across the original and curated corpora reflect their different token distributions.
These windows provide within-corpus variation, not independent corpus replications.
The [result JSON](../../Results/083/083-curated-160-joint-kl.json) retains every window.
The 3,276-trial sensitivity sweep is running separately.
