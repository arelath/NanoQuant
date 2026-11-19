"""Check simultaneous initialisation, backfitting, and paired checkpoint selection."""

import pytest
import torch
from probe_joint_independent_stages import alternate_start, select, simultaneous_start, tune_checkpoints

from nanoquant.domain.factorization import AdmmParameters


def test_both_stages_start_active_and_receive_joint_updates():
    rng = torch.Generator().manual_seed(7)
    target = torch.randn(8, 6, generator=rng) * 0.1
    model = simultaneous_start(target, torch.ones(6), torch.ones(8), (3, 2), 1)
    terms = (model, *model.residual_terms)
    initial = [term.scale_mid.detach().clone() for term in terms]
    assert all(t.abs().sum() > 0 for t in initial)
    x = torch.randn(2, 3, 6, generator=rng)
    seen = []
    tune_checkpoints(model, x, x @ target.mT, lambda epoch, _: seen.append(epoch), (0, 1, 3))
    assert seen == [0, 1, 3]
    assert all(not torch.equal(term.scale_mid, start) for term, start in zip(terms, initial, strict=True))


def test_alternation_checks_both_stages_and_never_accepts_worse_error():
    target = torch.randn(8, 6, generator=torch.Generator().manual_seed(5)) * 0.1
    model, moves = alternate_start(target, torch.ones(6), torch.ones(8), (3, 2), 1,
                                   AdmmParameters(outer_iterations=4))
    assert model.scale_mid.numel() == 3 and model.residual_terms[0].scale_mid.numel() == 2
    assert [m["stage"] for m in moves] == [0, 1]
    assert all(m["accepted"] == (m["candidate"] < m["before"]) for m in moves)


def test_checkpoint_selection_cannot_cherry_pick_seeds():
    rows = [dict(owner="up", seed=seed, method="warm", epoch=epoch, after={"activation_nmse": loss})
            for seed, epoch, loss in ((0, 8, 0.1), (1, 8, 0.4), (0, 32, 0.2), (1, 32, 0.25))]
    assert {r["epoch"] for r in select(rows, "up", ("warm",))} == {32}
    with pytest.raises(ValueError, match="unmatched"):
        select(rows[:-1], "up", ("warm",))
