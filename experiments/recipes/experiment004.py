"""Experiment 004: selectively add 30% packed bits to Experiment 003 v_proj layers."""

from dataclasses import replace
from pathlib import Path

from nanoquant.config.schema import IntentConfig
from nanoquant.rank_expansion_experiment import RankExpansionExperiment

from .experiment003 import EXPERIMENT_003_CONFIG

EXPERIMENT_004_CONFIG = replace(
    EXPERIMENT_003_CONFIG,
    intent=IntentConfig(
        experiment_number=4,
        name="004-gemma-3-4b-it-vproj-plus30",
        purpose="Measure whether additive v_proj rank improves Experiment 003 reconstruction and quality.",
        hypothesis=(
            "Adding 30% packed bits only to final Experiment 003 v_proj states lowers weighted reconstruction "
            "error and improves matched WikiText/task quality while every non-v_proj tensor remains exact."
        ),
        baseline_run="003-compress-and-benchmark-gemma-3-4b-it-v5",
        tags=("rank-allocation", "v-proj", "selective-replay", "gemma-3-4b-it"),
    ),
)

_ROOT = Path("outputs/004-gemma-3-4b-it-vproj-plus30")
EXPERIMENT_004 = RankExpansionExperiment(
    parent_run=Path("evidence/m10/003-compress-and-benchmark-gemma-3-4b-it-v5"),
    source_packed=Path("outputs/003-gemma-3-4b-it/packed"),
    output_packed=_ROOT / "packed",
    checkpoint_output=_ROOT / "llamacpp-checkpoint",
    gguf_output=_ROOT / "gemma-3-4b-it-vproj-plus30-nanoquant.gguf",
    expansion_report=Path("evidence/m11/004-gemma-3-4b-it-vproj-plus30-expansion.json"),
    quality_output=Path("evidence/m11/004-gemma-3-4b-it-vproj-plus30-quality.json"),
    quality_markdown_output=Path("evidence/m11/004-gemma-3-4b-it-vproj-plus30-quality.md"),
    summary_output=Path("evidence/m11/004-gemma-3-4b-it-vproj-plus30-summary.json"),
    baseline_quality=Path("Results/003/003-gemma-3-4b-it-quality.json"),
    llama_cpp_root=Path(r"D:\dev\research\llama.cpp"),
)

__all__ = ["EXPERIMENT_004", "EXPERIMENT_004_CONFIG"]
