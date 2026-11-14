import json
import runpy
from pathlib import Path

import pytest
from recipes import BASE_COMPRESSION_TEMPLATE

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.config.codec import config_hash
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from tests.support.experiments import config_diff_paths, load_experiment


def test_073_changes_model_imatrix_journal_pin_and_remaining_block_batches():
    namespace = runpy.run_path(str(next(Path('experiments').glob('073-*.py'))))
    parent, config = namespace['REFERENCE'], namespace['CONFIG']
    assert parent == load_experiment(72).config
    assert config_diff_paths(parent, config) == {
        'model.source', 'model.revision', 'model.tokenizer_revision',
        'calibration.imatrix', 'block_tuning.factorized.optimizer.continuation_journal_sha256',
        'block_tuning.batch_schedule.enabled', 'block_tuning.batch_schedule.start_block',
        'block_tuning.batch_schedule.continuation_journal_sha256',
    }
    assert config.model == BASE_COMPRESSION_TEMPLATE.model
    assert config.factorization == parent.factorization
    assert config.factorization.admm.outer_iterations == 600
    assert config.factorization.admm.inner_iterations == 2
    assert config.factorization.residual_stages.enabled
    assert config.block_tuning.compile_forward
    optimizer = config.block_tuning.factorized.optimizer
    assert optimizer.continuation_journal_sha256 is None
    assert optimizer.start_block == 19
    assert optimizer.name == 'fused_adamw_scales_fp32'
    assert optimizer.layer_patterns == ('mlp.gate_proj', 'mlp.down_proj', 'self_attn.o_proj')
    schedule = config.block_tuning.batch_schedule
    assert schedule.enabled and schedule.start_block == 11
    assert schedule.batch_size is None and schedule.maximum_batch_size == 8
    assert schedule.continuation_journal_sha256 == '10ecd0a406bc1efa1ae1543494595cc73aca5300bcb7adbb1838325989bdca99'
    # Blocks before the schedule keep 072's committed batch sizes.
    assert config.block_tuning.factorized.loop.batch_size == parent.block_tuning.factorized.loop.batch_size == 1
    assert config.allocation.maximum_rank_layer_patterns == ('*',)
    assert config.allocation.bounds.overcomplete_rank_ceiling_fraction == 2.
    assert config.calibration.imatrix.sha256 == '86a81192de49a6480f8dc19c5a7fd3f4377af59172fd4c7b8d93d296a0a18515'
    assert Path(config.calibration.imatrix.path).name == 'imatrix_unsloth.dat'
    assert namespace['KL_TEACHER_DATASET'].name == 'gemma-3-1b-it-kl-48x512'
    experiment = namespace['EXPERIMENT']
    assert experiment.config.intent.experiment_number == 73
    assert experiment.config.intent.baseline_run == '071-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it'
    assert not experiment.workflow.restore_completed_blocks
    assert experiment.workflow.interrupt_after_block_commits == 1
    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.quality_backend == 'dense'
    assert experiment.workflow.local_files_only
    assert experiment.workflow.export.gguf_output.name == 'gemma-3-1b-it-nanoquant.gguf'
    assert experiment.workflow.export.gguf_output.parent == Path('Results/073')
    assert config_hash(experiment.config) != config_hash(parent)
    raise_for_issues(validate(experiment.config, ValidationPhase.RESOLVED))


@pytest.mark.parametrize('flag', ['--plan-only', '--dry-run'])
def test_073_inspection_starts_no_stages(tmp_path, monkeypatch, capsys, flag):
    def forbidden(*args, **kwargs):
        pytest.fail('setup inspection must not load datasets or execute a model')

    for name in ('load_kl_teacher_dataset', 'config_with_allocation_dataset', '_prepare_automatic_kl_inputs',
                 'run_compression_quality_experiment', 'resolve_compression_quality_experiment'):
        monkeypatch.setattr(workflow, name, forbidden)
    assert workflow.run_self_measured_d2_experiment(
        load_experiment(73), launcher_path=tmp_path / 'experiments/073.py', arguments=[flag],
        profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True),
        reuse_maximum_rank_control=True, kl_teacher_dataset=tmp_path / 'missing-teacher',
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['config']['calibration']['imatrix']['sha256'].startswith('86a81192')
    assert report['config']['model']['source'] == 'google/gemma-3-1b-it'
    assert report['config']['block_tuning']['factorized']['optimizer']['continuation_journal_sha256'] is None
    if flag == '--dry-run':
        assert report['workflow']['export']['publication_experiment_number'] == 73
        assert report['config']['allocation']['strategy'] == 'uniform'
        assert report['config']['intent']['name'] == '073-d2-uniform-control-gemma-3-1b-it'
    assert not tuple(tmp_path.iterdir())


def test_073_campaign_forwards_read_only_options(monkeypatch):
    namespace = runpy.run_path('tools/run_experiment073_campaign.py')
    calls = []
    monkeypatch.setitem(namespace['main'].__globals__, 'run_campaign', lambda **kwargs: calls.append(kwargs) or 0)
    assert namespace['main'](['--plan-only']) == 0
    call = calls[0]
    assert call['number'] == 73
    assert call['control_name'] == '073-d2-uniform-control-gemma-3-1b-it'
    assert call['maximum_slices'] > 26
    assert call['arguments'] == ['--plan-only']
