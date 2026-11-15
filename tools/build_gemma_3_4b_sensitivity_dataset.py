"""Build the pinned Gemma 3 4B IT sensitivity dataset, including plots."""

from kl_sensitivity_dataset import GEMMA_4B, main

if __name__ == "__main__":
    raise SystemExit(main(spec=GEMMA_4B, dataset_reports=True))
