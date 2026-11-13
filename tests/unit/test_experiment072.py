import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import run_experiment070_campaign as campaign
from recipes import GEMMA_3_4B_COMPRESSION_TEMPLATE

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from nanoquant.infrastructure.model_adapters import decoder_block_count_from_config, vocabulary_size_from_config
from tests.support.experiments import config_diff_paths, load_experiment


def test_072_retains_071_recipe_with_explicit_4b_adaptations():
    namespace = runpy.run_path(str(next(Path("experiments").glob("072-*.py"))))
    parent, config = namespace["REFERENCE"], namespace["CONFIG"]
    assert config_diff_paths(parent, config) == {
        "model.source", "model.revision", "model.tokenizer_revision",
        "allocation.kl_profile_artifact", "allocation.kl_profile_key",
        "block_tuning.non_factorized.loop.batch_size", "block_tuning.factorized.loop.batch_size",
        "block_tuning.post_block_refit.batch_size", "block_tuning.microbatch_size",
        "runtime.block_forward_batch_size", "evaluation.inline_quality",
    }
    assert config.model == GEMMA_3_4B_COMPRESSION_TEMPLATE.model
    assert config.factorization == load_experiment(71).config.factorization
    assert config.factorization.admm.outer_iterations == 600
    assert config.factorization.admm.inner_iterations == 2
    assert config.factorization.residual_stages.enabled
    assert config.block_tuning.compile_forward
    assert config.distillation == parent.distillation
    assert config.calibration == parent.calibration
    assert config.allocation.maximum_rank_layer_patterns == ("*",)
    assert config.allocation.bounds.overcomplete_rank_ceiling_fraction == 2.0
    assert config.allocation.reconstruction.response_source.value == "measured"
    assert config.allocation.reconstruction.allocation_dataset is None
    assert namespace["KL_TEACHER_DATASET"].name == "gemma-3-4b-it-kl-48x512"
    experiment = namespace["EXPERIMENT"]
    assert not experiment.workflow.restore_completed_blocks
    assert not config.evaluation.inline_quality
    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.interrupt_after_block_commits == 1
    assert experiment.workflow.quality_backend == "dense"
    assert experiment.workflow.local_files_only
    assert experiment.workflow.export.gguf_output.name == "gemma-3-4b-it-nanoquant.gguf"
    raise_for_issues(validate(experiment.config, ValidationPhase.RESOLVED))


def test_072_plan_does_not_load_datasets_or_start_preparation(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("plan-only must not load targets or start computation")

    monkeypatch.setattr(workflow, "load_kl_teacher_dataset", forbidden)
    monkeypatch.setattr(workflow, "config_with_allocation_dataset", forbidden)
    monkeypatch.setattr(workflow, "_prepare_automatic_kl_inputs", forbidden)
    monkeypatch.setattr(workflow, "run_compression_quality_experiment", forbidden)
    assert workflow.run_self_measured_d2_experiment(
        load_experiment(72), launcher_path=tmp_path / "experiments/072.py", arguments=["--plan-only"],
        profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True),
        kl_teacher_dataset=tmp_path / "not-built-yet",
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["plan_only"] and not report["dataset_contents_validated"]
    assert report["automatic_control_and_profile"]
    assert report["profile_options"]["wikitext_samples"] == 48
    assert not tuple(tmp_path.iterdir())


def test_072_campaign_has_enough_slices_and_forwards_plan_only(tmp_path, monkeypatch):
    namespace = runpy.run_path("tools/run_experiment072_campaign.py")
    calls = []
    monkeypatch.setitem(namespace["main"].__globals__, "run_campaign", lambda **kwargs: calls.append(kwargs) or 0)
    assert namespace["main"](["--plan-only"]) == 0
    options = calls[0]
    assert options["maximum_slices"] > 2 * 34
    monkeypatch.setattr(campaign, "__file__", str(tmp_path / "tools/run_experiment070_campaign.py"))
    commands = []
    monkeypatch.setattr(campaign.subprocess, "run", lambda command, **kwargs:
                        commands.append(command) or SimpleNamespace(returncode=0))
    assert campaign.main(**options) == 0
    assert commands[0][-1] == "--plan-only"
    assert Path(commands[0][1]).name == "072-two-full-rank-stages-compress-and-benchmark-gemma-3-4b-it.py"
    assert not tuple(tmp_path.iterdir())


def test_4b_text_config_resolves_omitted_vocabulary_default_without_weights():
    config = {"model_type": "gemma3", "text_config": {
        "model_type": "gemma3_text", "hidden_size": 2560, "intermediate_size": 10240, "num_hidden_layers": 34,
    }, "torch_dtype": "bfloat16"}
    assert vocabulary_size_from_config(config) == 262208
    assert decoder_block_count_from_config(config) == 34


@pytest.mark.parametrize("config", [
    {"model_type": "llama", "vocab_size": 11},
    {"model_type": "gemma3", "text_config": {"num_hidden_layers": 34}},
])
def test_vocabulary_resolution_preserves_raw_config_identity(config):
    before = json.dumps(config, sort_keys=True)
    vocabulary_size_from_config(config)
    assert json.dumps(config, sort_keys=True) == before


def test_072_reuses_control_identity_and_publishes_numbered_results(tmp_path, monkeypatch):
    experiment = load_experiment(72)
    launcher = tmp_path / "experiments/072.py"
    options = workflow.SelfMeasuredD2ProfileOptions(48, 512, True)
    expected = workflow._uniform_control_config(
        experiment.config, tmp_path / "evidence/072", tuned_operating_point=True,
    )
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("fixed ranks must skip KL data loading, profiling and candidate compression")

    monkeypatch.setattr(workflow, "load_kl_teacher_dataset", forbidden)
    monkeypatch.setattr(workflow, "_prepare_automatic_kl_inputs", forbidden)
    monkeypatch.setattr(workflow, "run_compression_quality_experiment", forbidden)
    sentinel = object()
    monkeypatch.setattr(workflow, "resolve_compression_quality_experiment",
                        lambda config, publication, **kwargs: sentinel)
    monkeypatch.setattr(workflow, "execute_compression_quality_experiment",
                        lambda config, publication, resolved: calls.append((config, publication, resolved)))
    assert workflow.run_self_measured_d2_experiment(
        experiment, launcher_path=launcher, arguments=[], profile_options=options,
        reuse_maximum_rank_control=True, kl_teacher_dataset=tmp_path / "teacher",
    ) == 0
    config, publication, resolved = calls[0]
    assert config == expected
    assert config.distillation == experiment.config.distillation
    assert publication.export.publication_experiment_number == 72
    assert publication.export.gguf_output == experiment.workflow.export.gguf_output
    assert resolved is sentinel


def test_072_control_reuse_dry_run_needs_no_allocation_dataset(tmp_path, capsys):
    assert workflow.run_self_measured_d2_experiment(
        load_experiment(72), launcher_path=tmp_path / "experiments/072.py", arguments=["--dry-run"],
        profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True), reuse_maximum_rank_control=True,
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config"]["allocation"]["strategy"] == "uniform"
    assert report["config"]["allocation"]["maximum_rank_layer_patterns"] == ["*"]
    assert report["workflow"]["export"]["publication_experiment_number"] == 72
    assert not tuple(tmp_path.iterdir())


def test_control_reuse_rejects_variable_ranks(tmp_path):
    with pytest.raises(ValueError, match="maximum rank"):
        workflow.run_self_measured_d2_experiment(
            load_experiment(70), launcher_path=tmp_path / "experiments/070.py", arguments=["--plan-only"],
            profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True), reuse_maximum_rank_control=True,
        )
