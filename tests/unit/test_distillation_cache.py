from pathlib import Path

import pytest
import torch

from nanoquant.application.distillation import TopKTeacherBatch
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.distillation_cache import (
    TeacherCacheIdentity,
    commit_teacher_epoch,
    load_teacher_cache_journal,
    materialize_teacher_cache,
    record_teacher_epoch,
)


def _batch(sample: int, offset: float) -> TopKTeacherBatch:
    return TopKTeacherBatch(
        (sample,),
        torch.tensor([0, 2]),
        torch.tensor(((1.0 + offset, 0.5), (0.8, 0.2 + offset))),
        torch.tensor(((3, 4), (5, 6)), dtype=torch.int32),
    )


def test_teacher_cache_epochs_commit_resume_and_materialize(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    identity = TeacherCacheIdentity("sha256:protocol", "sha256:tokens")
    journal = load_teacher_cache_journal(tmp_path, identity, 2)
    assert journal.epochs == (None, None)
    with pytest.raises(ValueError, match="incomplete"):
        materialize_teacher_cache(journal, artifacts)

    first = commit_teacher_epoch(0, (_batch(0, 0.0),), identity, artifacts)
    journal = record_teacher_epoch(tmp_path, journal, 0, first.reference)
    resumed = load_teacher_cache_journal(tmp_path, identity, 2)
    assert resumed == journal
    assert resumed.epochs == (first.reference, None)

    second = commit_teacher_epoch(1, (_batch(1, 0.25),), identity, artifacts)
    complete = record_teacher_epoch(tmp_path, resumed, 1, second.reference)
    cache = materialize_teacher_cache(complete, artifacts)

    assert cache.epochs[0][0].sample_indices == (0,)
    assert cache.epochs[1][0].sample_indices == (1,)
    assert torch.equal(cache.epochs[1][0].top_values, _batch(1, 0.25).top_values)
    assert cache.bytes == first.bytes + second.bytes
    with pytest.raises(ValueError, match="does not match"):
        load_teacher_cache_journal(tmp_path, TeacherCacheIdentity("different", "sha256:tokens"), 2)
    replacement_identity = TeacherCacheIdentity("different", "sha256:tokens")
    replacement = load_teacher_cache_journal(
        tmp_path,
        replacement_identity,
        3,
        replace_mismatched=True,
    )
    assert replacement.identity == replacement_identity
    assert replacement.epochs == (None, None, None)
