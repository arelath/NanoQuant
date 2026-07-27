# Experiment 008: Gemma 3 12B scale-up

## Status

**Partial; no valid quality result.**

- Model: `google/gemma-3-12b-it`
- Launcher: `experiments/008-compress-and-benchmark-gemma-3-12b-it.py`
- Retained state: [`Results/008`](../../Results/008/)

## Question

Could the resident workflow scale from 4B to 12B using CPU offload, forward-only execution, and conservative memory
guards?

## What we did

We attempted several bounded-memory configurations. The furthest retained run reached **161 of 336 layers** and **23
of 48 blocks**. An earlier forward-only attempt failed with CUDA out-of-memory.

## Results

No completed compressed model or comparable quality report exists. The retained `running` state is historical and
must not be interpreted as an active or successful run.

## What we learned

Static guards and offload were insufficient for reliable 12B execution on the available hardware. Scale-up needed
more adaptive residency/streaming, clearer heartbeat and terminal-state reporting, and careful process/GPU ownership
checks. A partial block prefix does not support a quality conclusion.

## Disposition

Stopped. Preserve only as scale and failure-mode evidence.
