import torch

from nanoquant.domain.binary_qat import (
    align_shadow_to_persisted_signs,
    binary_sign,
    hamming_changes,
    project_hamming_budget,
)


def test_binary_sign_maps_zero_to_positive() -> None:
    assert torch.equal(
        binary_sign(torch.tensor([-2.0, -0.0, 0.0, 3.0])),
        torch.tensor([-1.0, 1.0, 1.0, 1.0]),
    )


def test_hamming_projection_keeps_deepest_crossings_globally() -> None:
    entry = (torch.tensor([1.0, 1.0, 1.0]), torch.tensor([-1.0, -1.0]))
    latents = (torch.tensor([-0.2, -0.9, 0.3]), torch.tensor([0.7, -0.2]))

    retained = project_hamming_budget(latents, entry, 0.4)

    assert retained == 2
    assert hamming_changes(latents, tuple(binary_sign(value) for value in entry)) == 2
    assert torch.equal(latents[0], torch.tensor([1.0, -0.9, 0.3]))
    assert torch.equal(latents[1], torch.tensor([0.7, -0.2]))


def test_shadow_alignment_preserves_magnitudes_under_frozen_signs() -> None:
    aligned = align_shadow_to_persisted_signs(
        torch.tensor([-0.2, 0.0, 0.7]),
        torch.tensor([1.0, -1.0, -1.0]),
    )

    assert torch.equal(binary_sign(aligned), torch.tensor([1.0, -1.0, -1.0]))
    assert torch.allclose(aligned.abs(), torch.tensor([0.2, 1e-7, 0.7]))


def test_zero_hamming_budget_restores_exact_entry_latents() -> None:
    entry = (torch.tensor([0.25, -0.5]),)
    latents = (torch.tensor([-0.1, 0.2]),)

    assert project_hamming_budget(latents, entry, 0.0) == 0
    assert torch.equal(latents[0], entry[0])
