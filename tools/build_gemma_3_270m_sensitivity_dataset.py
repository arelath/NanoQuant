"""Build the pinned Gemma 3 270M IT sensitivity dataset, including plots."""

import torch
from kl_sensitivity_dataset import ModelSpec, main

GEMMA_270M = ModelSpec(
    "google/gemma-3-270m-it",
    "ac82b4e820549b854eebf28ce6dedaf9fdfa17b3",
    "gemma3_text",
    18,
    (("model.safetensors", "700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab"),),
    # FP16 overflows on the clean 270M model, and BF16 rounding alone adds about 0.004 nats/token of KL
    # whenever any weight changes, which hides the response below roughly 5% weight error. FP32 removes
    # that floor; the earlier BF16 measurements remain under gemma-3-270m-it-bf16-sensitivity-16x256.
    dtype=torch.float32,
)

if __name__ == "__main__":
    raise SystemExit(main(spec=GEMMA_270M, dataset_reports=True))
