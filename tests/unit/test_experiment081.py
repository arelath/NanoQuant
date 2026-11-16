import runpy
from pathlib import Path

from tests.support.experiments import config_diff_paths, load_experiment


def test_081_retains_074_recipe_except_stage_ceiling_and_global_kd() -> None:
    before, after = [runpy.run_path(str(next(Path("experiments").glob(f"{number:03d}-*.py"))))["CONFIG"]
                     for number in (74, 81)]
    assert config_diff_paths(before, after) == {
        "allocation.bounds.overcomplete_rank_ceiling_fraction", "distillation.enabled",
    }
    experiment = load_experiment(81)
    assert experiment.config.allocation.bounds.overcomplete_rank_ceiling_fraction == 3.0
    assert experiment.config.allocation.maximum_rank_layer_patterns == ("*",)
    assert experiment.config.factorization.residual_stages.enabled
    assert not experiment.config.distillation.enabled
    assert experiment.workflow.include_mmlu and experiment.workflow.task_limit == 1000
    assert experiment.workflow.export.gguf_output == Path("Results/081/gemma-3-270m-it-nanoquant.gguf")
