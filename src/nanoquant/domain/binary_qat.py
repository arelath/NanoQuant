"""Pure trust-region helpers for binary-factor QAT."""

from __future__ import annotations

import math

import torch


def binary_sign(value: torch.Tensor) -> torch.Tensor:
    """Return the persisted NanoQuant sign convention (zero maps to +1)."""

    return torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))


def align_shadow_to_persisted_signs(
    shadow: torch.Tensor,
    persisted_signs: torch.Tensor,
    *,
    minimum_margin: float = 1e-7,
) -> torch.Tensor:
    """Reuse continuous margin magnitudes under an authoritative sign state."""

    if shadow.shape != persisted_signs.shape or not math.isfinite(minimum_margin) or minimum_margin <= 0:
        raise ValueError("binary QAT shadow alignment inputs are invalid")
    signs = binary_sign(persisted_signs).to(device=shadow.device, dtype=shadow.dtype)
    return signs * shadow.abs().clamp_min(minimum_margin)


def hamming_changes(
    latents: tuple[torch.Tensor, ...],
    entry_signs: tuple[torch.Tensor, ...],
) -> int:
    if len(latents) != len(entry_signs) or not latents:
        raise ValueError("binary QAT latent and entry-sign inventories must match")
    total = 0
    for latent, entry_sign in zip(latents, entry_signs, strict=True):
        if latent.shape != entry_sign.shape:
            raise ValueError("binary QAT latent and entry-sign shapes must match")
        total += int((binary_sign(latent) != entry_sign).sum())
    return total


@torch.no_grad()
def project_hamming_budget(
    latents: tuple[torch.Tensor, ...],
    entry_latents: tuple[torch.Tensor, ...],
    maximum_change_fraction: float,
) -> int:
    """Project shadow latents to one deterministic global Hamming ball.

    The deepest sign crossings survive. Excess crossings are restored to their
    exact entry latents, which keeps the persisted state within the hard budget
    without inventing a second continuous representation.
    """

    if (
        len(latents) != len(entry_latents)
        or not latents
        or not math.isfinite(maximum_change_fraction)
        or not 0 <= maximum_change_fraction <= 1
    ):
        raise ValueError("binary QAT Hamming projection inputs are invalid")
    entry_signs = tuple(binary_sign(value) for value in entry_latents)
    changed_flat_indices: list[torch.Tensor] = []
    strengths: list[torch.Tensor] = []
    total_elements = 0
    for latent, entry, entry_sign in zip(latents, entry_latents, entry_signs, strict=True):
        if latent.shape != entry.shape or latent.device != entry.device:
            raise ValueError("binary QAT latent and entry tensors must align")
        total_elements += latent.numel()
        changed = (binary_sign(latent) != entry_sign).reshape(-1)
        indices = torch.nonzero(changed, as_tuple=False).flatten()
        changed_flat_indices.append(indices)
        if indices.numel():
            signed_margin = (-(latent * entry_sign)).reshape(-1).index_select(0, indices)
            strengths.append(signed_margin.float())
    maximum_changes = int(math.floor(total_elements * maximum_change_fraction))
    current_changes = sum(int(value.numel()) for value in changed_flat_indices)
    if current_changes <= maximum_changes:
        return current_changes
    if maximum_changes == 0:
        for latent, entry in zip(latents, entry_latents, strict=True):
            latent.copy_(entry)
        return 0
    joined = torch.cat(strengths)
    keep_joined = torch.zeros(joined.numel(), dtype=torch.bool, device=joined.device)
    # ``sorted=True`` plus the stable concatenation order makes ties reproducible.
    keep_joined[torch.topk(joined, maximum_changes, sorted=True).indices] = True
    offset = 0
    for latent, entry, indices in zip(latents, entry_latents, changed_flat_indices, strict=True):
        count = indices.numel()
        local_keep = keep_joined[offset : offset + count]
        restore = indices[~local_keep]
        if restore.numel():
            latent.reshape(-1)[restore] = entry.reshape(-1)[restore]
        offset += count
    return maximum_changes
