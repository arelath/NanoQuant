# Copyright (c) 2026 Samsung Electronics Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 Gemma 3 1B 65k-token Hessian safety correction experiment.

This follow-up to experiment 015 keeps the 65,536-token sampling and tapered Phase-1
recipe, disables sibling reuse, blends the full Hessian toward the all-calibration
diagonal, and restores a raw-error safety gate for rank retries.

The output checkpoint is discovered automatically by
``007-evaluate-all-gemma-quality.py``.
"""

from __future__ import annotations

import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
MODEL_ID = "google/gemma-3-4b-it"
OUTPUT_DIR = ROOT / "outputs"
QMODEL_PATH = OUTPUT_DIR / "gemma-3-4b-it-nq-chat-phase1-no-hessian.pt"
LOG_DIR = ROOT / "logs"
LOG_PATH = LOG_DIR / "019-compress-gemma-3-4b-it-phase1.log"

# Keep the inherited 013 recipe explicit for experiment provenance.
OUTLIER_FRAC = 0.001
OUTLIER_DTYPE = "bf16"
OUTLIER_LAYERS = "all"
OUTLIER_METRIC = "residual"
OUTLIER_BUDGET_COMPENSATE = False
OUTLIER_COUNT_MULTIPLE = 1
OUTLIER_I_NORM_MODE = "zero"
OUTLIER_RESIDUAL_PROBE_ITERS = 80
QUANT_LAYER_ORDER = "mlp_first"
RANK_FLOOR_FRAC = 0.90
RANK_CEIL_FRAC = 1.10
# Gemma 3 4B's full 256x2048 activation tensors do not fit alongside the
# current block, optimizer state, and float32 reconstruction-loss workspace on
# a 12 GiB GPU. Keep the full calibration set on pageable CPU memory and stage
# only the current minibatch on CUDA. Pinning each 2.5 GiB full-store buffer is
# reported by Windows/WDDM as Shared GPU memory and causes severe pressure.
BLOCK_ACTIVATION_DEVICE = "cpu"
BLOCK_ACTIVATION_GPU_CACHE = "auto"
BLOCK_ACTIVATION_GPU_RESERVE_GIB = 6.0
PIN_CPU_ACTIVATION_MAX_GIB = 1.0
BLOCK_FORWARD_BATCH_SIZE = 4
NONFACT_BATCH_SIZE = 4
FACT_BATCH_SIZE = 1
CLEANUP_PER_LAYER = True
NONFACT_EARLY_STOP_REL_TOL = 0.0

# Phase-1 treatment inherited from 015; safety blending/retry behavior is the ablation.
NONFACT_EPOCH_SCHEDULE = "8,4,3,2,2,2,2"
POST_BLOCK_SCALE_EPOCHS = 2
HESSIAN_WHITENING = False
HESSIAN_MAX_TOKENS = 262144
HESSIAN_MAX_SEQUENCES = 256
HESSIAN_BATCH_SIZE = 8
HESSIAN_REUSE_SIBLINGS = True
HESSIAN_DAMP_PERCENT = 0.01
HESSIAN_SHRINKAGE = 0.0
HESSIAN_DIAGONAL_BLEND = 0.20
RAW_RETRY_NORM_ERROR_THRESHOLD = 0.40
RANK_RETRY_ALLOW_ABOVE_CAP = True
WEIGHT_ERROR_LOG_PATH = OUTPUT_DIR / "019-phase1-weight-errors.csv"
WEIGHT_ERROR_TABLE_PATH = OUTPUT_DIR / "019-phase1-weight-errors.md"
RANK_UTILITY_LOG_PATH = OUTPUT_DIR / "019-phase1-rank-utility.csv"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

    if "HF_TOKEN" in os.environ:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", os.environ["HF_TOKEN"])


def main() -> None:
    sys.path.insert(0, str(SRC))
    load_dotenv(ROOT / ".env")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    from nanoquant.modules.hub import NanoQuantConfigDataclass, NanoQuantModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for NanoQuant compression.")

    quant_config = NanoQuantConfigDataclass(
        model_id=MODEL_ID,
        bits=1.0,
        seed=0,
        device_map="cuda",
        eval_block_ppl=False,
        block_forward_batch_size=BLOCK_FORWARD_BATCH_SIZE,
        block_activation_device=BLOCK_ACTIVATION_DEVICE,
        block_activation_gpu_cache=BLOCK_ACTIVATION_GPU_CACHE,
        block_activation_gpu_reserve_gib=BLOCK_ACTIVATION_GPU_RESERVE_GIB,
        pin_cpu_activation_max_gib=PIN_CPU_ACTIVATION_MAX_GIB,
        cleanup_per_layer=CLEANUP_PER_LAYER,
        quant_layer_order=QUANT_LAYER_ORDER,
        num_calib_samples=256,
        calib_dataset="ultrachat_200k,wikitext2",
        calib_shrinkage=0.6,
        calib_strategy="online",
        hessian_whitening=HESSIAN_WHITENING,
        hessian_max_tokens=HESSIAN_MAX_TOKENS,
        hessian_max_sequences=HESSIAN_MAX_SEQUENCES,
        hessian_batch_size=HESSIAN_BATCH_SIZE,
        hessian_reuse_siblings=HESSIAN_REUSE_SIBLINGS,
        hessian_damp_percent=HESSIAN_DAMP_PERCENT,
        hessian_shrinkage=HESSIAN_SHRINKAGE,
        hessian_diagonal_blend=HESSIAN_DIAGONAL_BLEND,
        rank_allocation_strategy="sensitivity",
        rank_sensitivity_alpha=0.5,
        rank_edge_boost=0.15,
        rank_floor_frac=RANK_FLOOR_FRAC,
        rank_ceil_frac=RANK_CEIL_FRAC,
        rank_retry_norm_error_threshold=0.35,
        rank_retry_raw_norm_error_threshold=RAW_RETRY_NORM_ERROR_THRESHOLD,
        rank_retry_allow_above_cap=RANK_RETRY_ALLOW_ABOVE_CAP,
        rank_retry_bump_frac=0.25,
        rank_retry_max_attempts=2,
        rank_retry_bits_budget_frac=0.02,
        weight_error_log_path=str(WEIGHT_ERROR_LOG_PATH),
        weight_error_table_path=str(WEIGHT_ERROR_TABLE_PATH),
        rank_utility_profile_path=None,
        rank_utility_log_path=str(RANK_UTILITY_LOG_PATH),
        outlier_frac=OUTLIER_FRAC,
        outlier_dtype=OUTLIER_DTYPE,
        outlier_layers=OUTLIER_LAYERS,
        outlier_metric=OUTLIER_METRIC,
        outlier_budget_compensate=OUTLIER_BUDGET_COMPENSATE,
        outlier_count_multiple=OUTLIER_COUNT_MULTIPLE,
        outlier_i_norm_mode=OUTLIER_I_NORM_MODE,
        outlier_residual_probe_iters=OUTLIER_RESIDUAL_PROBE_ITERS,
        embed_tokens_weight_bits=8,
        nonfact_batch_size=NONFACT_BATCH_SIZE,
        nonfact_epochs=8,
        nonfact_early_stop_rel_tol=NONFACT_EARLY_STOP_REL_TOL,
        nonfact_epoch_schedule=NONFACT_EPOCH_SCHEDULE,
        admm_type="nanoquant",
        admm_outer_iters=800,
        admm_penalty_scheduler="cubic",
        ls_scale_fit=True,
        ls_scale_fit_iters=2,
        fact_batch_size=FACT_BATCH_SIZE,
        fact_epochs=8,
        post_block_scale_epochs=POST_BLOCK_SCALE_EPOCHS,
        model_kd_gradient_checkpointing=True,
        # Top-k KD uses the memory-safe teacher-target cache. Full-KL requires
        # the 4B teacher and student to coexist on CUDA and does not fit in 12 GiB.
        model_kd_loss="topk",
        model_kd_topk=64,
        model_kd_max_tokens_per_batch=512,
        model_kd_vocab_chunk_size=8192,
        model_kd_token_chunk_size=128,
        model_kd_temperature=1.0,
        tune_eval_summaries=True,
    )

    print("Experiment 016: Phase-1 65k-token Hessian safety correction")
    print(f"Compressing {MODEL_ID}")
    print(f"Output checkpoint: {QMODEL_PATH}")
    print(f"Log file: {LOG_PATH}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print("Calibration dataset: UltraChat rendered with Gemma's chat template, mixed with Wikitext2.")
    print("Embedding checkpoint storage: model.embed_tokens.weight row-wise int8.")
    print("Per-block PPL checks are disabled for this run.")
    print(f"Quantization layer order: {QUANT_LAYER_ORDER}")
    print(f"Rank allocation clamp: [{RANK_FLOOR_FRAC:.2f}, {RANK_CEIL_FRAC:.2f}] x uniform base rank.")
    print("Rank utility profile: cold start; this run writes a reusable utility CSV.")
    print(
        "Full input-Hessian whitening: "
        f"enabled={HESSIAN_WHITENING}, max_tokens={HESSIAN_MAX_TOKENS}, "
        f"max_sequences={HESSIAN_MAX_SEQUENCES}, "
        f"batch_size={HESSIAN_BATCH_SIZE}, reuse_siblings={HESSIAN_REUSE_SIBLINGS}, "
        f"damp={HESSIAN_DAMP_PERCENT:.2%}, shrinkage={HESSIAN_SHRINKAGE:.2%}, "
        f"diagonal_blend={HESSIAN_DIAGONAL_BLEND:.2%}, "
        f"raw_retry_threshold={RAW_RETRY_NORM_ERROR_THRESHOLD:.2f}, "
        f"retry_above_allocation_cap={RANK_RETRY_ALLOW_ABOVE_CAP}."
    )
    print(
        "Tuning profile: "
        f"block_activation_device={BLOCK_ACTIVATION_DEVICE}, "
        f"block_activation_gpu_cache={BLOCK_ACTIVATION_GPU_CACHE}, "
        f"block_activation_gpu_reserve_gib={BLOCK_ACTIVATION_GPU_RESERVE_GIB}, "
        f"pin_cpu_activation_max_gib={PIN_CPU_ACTIVATION_MAX_GIB}, "
        f"block_forward_batch_size={BLOCK_FORWARD_BATCH_SIZE}, "
        f"nonfact_batch_size={NONFACT_BATCH_SIZE}, fact_batch_size={FACT_BATCH_SIZE}, "
        f"nonfact_early_stop_rel_tol={NONFACT_EARLY_STOP_REL_TOL}, "
        f"nonfact_epoch_schedule={NONFACT_EPOCH_SCHEDULE}, "
        f"post_block_scale_epochs={POST_BLOCK_SCALE_EPOCHS}."
    )
    print(
        "Salient outliers: "
        f"frac={OUTLIER_FRAC}, dtype={OUTLIER_DTYPE}, layers={OUTLIER_LAYERS}, "
        f"metric={OUTLIER_METRIC}, budget_compensate={OUTLIER_BUDGET_COMPENSATE}, "
        f"count_multiple={OUTLIER_COUNT_MULTIPLE}, i_norm_mode={OUTLIER_I_NORM_MODE}."
    )

    nanoquant_model = NanoQuantModel.from_pretrained_quantize(
        model_id=MODEL_ID,
        qmodel_path=str(QMODEL_PATH),
        quant_config=quant_config,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    print("Compression complete.")
    print(f"Saved checkpoint: {QMODEL_PATH}")
    print(f"Model type: {nanoquant_model.model.config.model_type}")


if __name__ == "__main__":
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            main()
