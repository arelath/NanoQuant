"""Experiment 015: architecture-protected reconstruction ranks for Gemma 3 270M."""

from dataclasses import replace

from recipes import (
    GEMMA_3_270M_COMPRESSION_TEMPLATE,
    RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    BaselineRef,
    ExperimentIdentity,
    ExperimentRef,
    define_compression_quality_experiment,
)

from nanoquant.compression_quality_workflow import run_compression_quality_experiment
from nanoquant.config.schema import LayerRankBudgetConfig, ReconstructionImportanceConfig

PARENT = ExperimentRef(14, "compress-and-benchmark-gemma-3-270m-it")

HISTORICAL_CONFIG = replace(
    RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE,
    model=GEMMA_3_270M_COMPRESSION_TEMPLATE.model,
    allocation=replace(
        RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation,
        reconstruction=replace(
            RECONSTRUCTION_AWARE_STACKED_QKV_COMPRESSION_TEMPLATE.allocation.reconstruction,
            sensitivity_strength=0.25,
            importance=ReconstructionImportanceConfig(
                layer_multipliers=(
                    LayerRankBudgetConfig("self_attn.q_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.k_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.v_proj", 1.25),
                    LayerRankBudgetConfig("self_attn.o_proj", 1.25),
                    LayerRankBudgetConfig("mlp.down_proj", 1.25),
                ),
                protected_layer_patterns=(
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                    "mlp.down_proj",
                ),
                edge_block_multiplier=1.25,
                protected_edge_block_count=1,
            ),
        ),
    ),
)

EXPERIMENT = define_compression_quality_experiment(
    ExperimentIdentity(
        number=15,
        name="compress-and-benchmark-gemma-3-270m-it",
        purpose=(
            "Allocate the reconstruction-informed fixed rank budget while explicitly protecting "
            "Q/K/V/O/down projections and the first and last transformer blocks."
        ),
        hypothesis=(
            "Architectural importance priors move rank toward Q/K/V/O/down and edge blocks, "
            "lowering their reconstruction error and improving quality relative to Experiment 014."
        ),
        baseline=BaselineRef.experiment(PARENT),
        tags=(
            "gemma-3-270m-it",
            "compression",
            "quality",
            "shared-input-qkv",
            "reconstruction-aware-ranks",
            "architecture-protected-ranks",
            "edge-block-protection",
            "wikitext2",
            "ultrachat",
        ),
    ),
    HISTORICAL_CONFIG,
    expected_blocks=18,
    maximum_wddm_shared_gib=0.75,
)


if __name__ == "__main__":
    raise SystemExit(
        run_compression_quality_experiment(
            EXPERIMENT.config,
            EXPERIMENT.workflow,
            launcher_path=__file__,
        )
    )
