import json
import runpy
from pathlib import Path

import pytest
from recipes import GEMMA_3_270M_COMPRESSION_TEMPLATE

import nanoquant.self_measured_d2_workflow as workflow
from nanoquant.config.codec import config_hash
from nanoquant.config.schema import ActivationGpuCacheMode
from nanoquant.config.validation import ValidationPhase, raise_for_issues, validate
from tests.support.experiments import config_diff_paths, load_experiment


def test_074_changes_model_imatrix_and_only_speed_settings():
    namespace = runpy.run_path(str(next(Path('experiments').glob('074-*.py'))))
    parent, config = namespace['REFERENCE'], namespace['CONFIG']
    assert parent == load_experiment(73).config
    assert config_diff_paths(parent, config) == {
        'model.source', 'model.revision', 'model.tokenizer_revision',
        'calibration.imatrix.path', 'calibration.imatrix.sha256',
        'block_tuning.factorized.optimizer.start_block',
        'block_tuning.batch_schedule.start_block',
        'block_tuning.batch_schedule.continuation_journal_sha256',
        'runtime.block_forward_batch_size', 'runtime.activations.gpu_cache',
    }
    assert config.model == GEMMA_3_270M_COMPRESSION_TEMPLATE.model
    assert config.model.source == 'unsloth/gemma-3-270m-it'
    assert config.factorization == parent.factorization
    assert config.distillation == parent.distillation
    assert config.allocation == parent.allocation
    schedule = config.block_tuning.batch_schedule
    assert schedule.enabled and schedule.start_block == 0
    assert schedule.batch_size is None and schedule.maximum_batch_size == 8
    assert schedule.continuation_journal_sha256 is None
    optimizer = config.block_tuning.factorized.optimizer
    assert optimizer.name == 'fused_adamw_scales_fp32' and optimizer.start_block == 0
    assert optimizer.layer_patterns == ('mlp.gate_proj', 'mlp.down_proj', 'self_attn.o_proj')
    assert optimizer.continuation_journal_sha256 is None
    assert config.block_tuning.compile_forward
    assert config.runtime.block_forward_batch_size == 8
    assert config.runtime.activations.gpu_cache is ActivationGpuCacheMode.AUTO
    assert config.calibration.imatrix.sha256 == '2f88200eae0588cf7b3030f34e5469a529758a1c69f7a13453b7d3a91de017f3'
    assert Path(config.calibration.imatrix.path).parent.name == 'gemma-3-270m-it-imatrix-unsloth'
    assert Path(config.calibration.imatrix.path).name == 'imatrix_unsloth.gguf_file'
    experiment = namespace['EXPERIMENT']
    assert experiment.config.intent.experiment_number == 74
    assert experiment.config.intent.baseline_run == (
        '073-imatrix-two-full-rank-stages-compress-and-benchmark-gemma-3-1b-it'
    )
    assert not experiment.workflow.restore_completed_blocks
    assert experiment.workflow.interrupt_after_block_commits is None
    assert experiment.workflow.task_limit == 1000
    assert experiment.workflow.quality_backend == 'dense'
    assert experiment.workflow.local_files_only
    assert experiment.workflow.export.gguf_output.name == 'gemma-3-270m-it-nanoquant.gguf'
    assert experiment.workflow.export.gguf_output.parent == Path('Results/074')
    assert config_hash(experiment.config) != config_hash(parent)
    raise_for_issues(validate(experiment.config, ValidationPhase.RESOLVED))


@pytest.mark.parametrize('flag', ['--plan-only', '--dry-run'])
def test_074_inspection_starts_no_stages(tmp_path, monkeypatch, capsys, flag):
    def forbidden(*args, **kwargs):
        pytest.fail('setup inspection must not load datasets or execute a model')

    for name in ('load_kl_teacher_dataset', 'config_with_allocation_dataset', '_prepare_automatic_kl_inputs',
                 'run_compression_quality_experiment', 'resolve_compression_quality_experiment'):
        monkeypatch.setattr(workflow, name, forbidden)
    assert workflow.run_self_measured_d2_experiment(
        load_experiment(74), launcher_path=tmp_path / 'experiments/074.py', arguments=[flag],
        profile_options=workflow.SelfMeasuredD2ProfileOptions(48, 512, True),
        reuse_maximum_rank_control=True,
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['config']['model']['source'] == 'unsloth/gemma-3-270m-it'
    assert report['config']['calibration']['imatrix']['sha256'].startswith('2f88200e')
    assert report['config']['block_tuning']['batch_schedule']['start_block'] == 0
    if flag == '--dry-run':
        assert report['workflow']['export']['publication_experiment_number'] == 74
        assert report['config']['allocation']['strategy'] == 'uniform'
        assert report['config']['intent']['name'] == '074-d2-uniform-control-gemma-3-270m-it'
    assert not tuple(tmp_path.iterdir())
