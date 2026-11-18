"""Build the pinned Gemma 3 1B IT sensitivity dataset, including plots."""

from kl_sensitivity_dataset import GEMMA_1B, main

if __name__ == "__main__":
    raise SystemExit(main(spec=GEMMA_1B, dataset_reports=True))
