# Experiment 027: Planned Llama 3 8B scale gate

## Status

**Not run.**

- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Launcher: `experiments/027-compress-and-benchmark-meta-llama-3-8b-instruct.py`
- Retained results: none

## Question

Could adaptive memory execution, a logical batch size of 32, and llama.cpp-based quality evaluation make an 8B
Llama run practical on the available hardware?

## Intended method

The launcher was configured as a large-model scale gate with conservative resident execution and deployment-runtime
evaluation. It was meant to test both completion and quality after the smaller Llama experiments.

## Results

The experiment was never run, so there are no measurements or empirical lessons.

## What we learned

Only a protocol-design lesson survives: an 8B attempt needed adaptive memory policy and a serialized deployment
evaluation path. The existence of a launcher must not be cited as proof that the repository supported the model.

## Disposition

Planned work only.
