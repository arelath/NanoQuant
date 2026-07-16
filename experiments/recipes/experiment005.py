"""Experiment 005: request 2x v_proj bits, saturating all layers at maximum rank."""

from dataclasses import replace
from pathlib import Path

from nanoquant.config.schema import IntentConfig
from nanoquant.rank_expansion_experiment import RankExpansionExperiment

from .experiment003 import EXPERIMENT_003_CONFIG

EXPERIMENT_005_CONFIG = replace(
    EXPERIMENT_003_CONFIG,
    intent=IntentConfig(
        experiment_number=5,
        name="005-gemma-3-4b-it-vproj-double-request",
        purpose="Upper-bound the Experiment 003 v_proj allocation hypothesis at maximum physical rank.",
        hypothesis=(
            "Requesting twice the packed v_proj bits, capped at rank 1024, may produce enough downstream quality "
            "gain to overturn the negative Experiment 004 result."
        ),
        baseline_run="003-compress-and-benchmark-gemma-3-4b-it-v5",
        tags=("rank-allocation", "v-proj", "maximum-rank", "gemma-3-4b-it"),
    ),
)

_ROOT = Path("outputs/005-gemma-3-4b-it-vproj-double-request")
EXPERIMENT_005 = RankExpansionExperiment(
    parent_run=Path("evidence/m10/003-compress-and-benchmark-gemma-3-4b-it-v5"),
    source_packed=Path("outputs/003-gemma-3-4b-it/packed"),
    output_packed=_ROOT / "packed",
    checkpoint_output=_ROOT / "llamacpp-checkpoint",
    gguf_output=_ROOT / "gemma-3-4b-it-vproj-maxrank-nanoquant.gguf",
    expansion_report=Path("evidence/m12/005-gemma-3-4b-it-vproj-maxrank-expansion.json"),
    quality_output=Path("evidence/m12/005-gemma-3-4b-it-vproj-maxrank-quality.json"),
    quality_markdown_output=Path("evidence/m12/005-gemma-3-4b-it-vproj-maxrank-quality.md"),
    summary_output=Path("evidence/m12/005-gemma-3-4b-it-vproj-maxrank-summary.json"),
    baseline_quality=Path("Results/003/003-gemma-3-4b-it-quality.json"),
    llama_cpp_root=Path(r"D:\dev\research\llama.cpp"),
    bit_multiplier=2.0,
)

__all__ = ["EXPERIMENT_005", "EXPERIMENT_005_CONFIG"]
