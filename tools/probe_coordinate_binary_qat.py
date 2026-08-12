"""Bounded coordinate binary-QAT probe on one retained NanoQuant owner.

The probe compares an immutable-sign continuous control with a matched arm that
also trains FP32 shadow sign latents.  The binary arm is projected after every
optimizer step into a hard Hamming ball around the exact retained entry state.
Both arms consume the same retained top-k-plus-tail teacher batches and are
judged on disjoint WikiText full-vocabulary KL and NLL.  No source-run pointer
or content-addressed artifact is mutated.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_global_foldable_mlp_multipliers import (
    _hidden_states,
    _hidden_states_width,
    _lm_head,
    _load_calibration,
    _load_training_cache,
)
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION
from probe_topk_tail_mass import TailMassSums, _means, _tail_mass_sums
from safetensors import safe_open
from torch import nn

from nanoquant.application.distillation import (
    TopKDistillationConfig,
    TopKTeacherBatch,
    topk_tail_distillation_loss,
)
from nanoquant.application.layers import TrainableFactorizedLinear
from nanoquant.application.parity_adamw import ParityAdamW
from nanoquant.config.codec import to_dict
from nanoquant.domain.binary_qat import (
    align_shadow_to_persisted_signs,
    binary_sign,
    hamming_changes,
    project_hamming_budget,
)
from nanoquant.domain.models import ArtifactRef, ArtifactTypes
from nanoquant.global_distillation import _thaw_frozen_layers
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.commits import latest_complete_identity, load_committed_block
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json
from nanoquant.infrastructure.reproducibility import deterministic_torch_execution
from nanoquant.infrastructure.tensor_store import LocalTensorStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--expected-blocks", type=int, default=26)
    parser.add_argument("--owner", default="mlp.gate_proj")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--maximum-batches-per-epoch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--binary-learning-rate", type=float, default=3e-5)
    parser.add_argument("--hamming-fraction", type=float, default=0.008)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--monitor-offset", type=int, default=128)
    parser.add_argument("--monitor-samples", type=int, default=8)
    parser.add_argument("--monitor-sequence-length", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _config(run_output: Path, epochs: int, maximum_batches: int) -> TopKDistillationConfig:
    manifest = json.loads((run_output / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["resolved_config"]["canonical_run_config"]["distillation"]
    names = {field.name for field in fields(TopKDistillationConfig)}
    values = {name: source[name] for name in names if name in source}
    values.update(
        {
            "objective": "top_k_tail",
            "epochs": epochs,
            "maximum_batches_per_epoch": maximum_batches,
            "gradient_checkpointing": True,
        }
    )
    return TopKDistillationConfig(**values)


def _load_shadow_initialization(
    run_output: Path,
    block: int,
    owner: str,
    expected_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    store = LocalArtifactStore(run_output / "artifacts")
    records = [
        json.loads(line)
        for line in (run_output / "state" / "journal.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identity, block_records = latest_complete_identity(records, expected_blocks)
    committed = load_committed_block(
        ArtifactRef(
            ArtifactTypes.BLOCK_RESULT,
            str(block_records[block]["artifact_id"]),
            1,
        ),
        store,
        identity,
    ).result
    try:
        result = next(value for value in committed.layers if value.layer.path == owner)
    except StopIteration as exc:
        raise ValueError("coordinate binary QAT currently requires an independent layer owner") from exc
    factorization_id = result.factorization.artifact_id
    tensor_path = store.path_for(factorization_id) / "tensors.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        required = {"left_latent", "right_latent"}
        if not required <= set(handle.keys()):
            raise ValueError("retained factorization is missing pre-hardening shadow latents")
        left = handle.get_tensor("left_latent").float()
        right = handle.get_tensor("right_latent").float()
    return left, right, factorization_id


def _set_shadow_latents(
    module: TrainableFactorizedLinear,
    shadow_initialization: tuple[torch.Tensor, torch.Tensor],
) -> None:
    module.left_latent.data = shadow_initialization[0].to(module.left_latent.device).clone()
    module.right_latent.data = shadow_initialization[1].to(module.right_latent.device).clone()


def _selected_parameters(
    module: TrainableFactorizedLinear,
    *,
    binary: bool,
) -> list[tuple[str, nn.Parameter]]:
    names = {"scale_pre", "scale_mid", "scale_post", "outlier_values", "bias"}
    if binary:
        names.update({"left_latent", "right_latent"})
    return [(name, parameter) for name, parameter in module.named_parameters() if name in names]


def _loss(
    model: nn.Module,
    tokens: torch.Tensor,
    target: TopKTeacherBatch,
    config: TopKDistillationConfig,
    device: str,
) -> torch.Tensor:
    sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
    batch = tokens.index_select(0, sample_indices).to(device)
    selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
    hidden = _hidden_states(model, batch).reshape(-1, _hidden_states_width(model))
    hidden = hidden.index_select(0, selected_tokens)
    if target.teacher_log_normalizers is None:
        raise ValueError("coordinate binary QAT requires retained tail normalizers")
    return topk_tail_distillation_loss(
        hidden,
        target.top_values.to(device),
        target.top_indices.to(device=device, dtype=torch.long),
        target.teacher_log_normalizers.to(device),
        _lm_head(model),
        temperature=config.temperature,
        vocabulary_chunk_size=config.vocabulary_chunk_size,
        token_chunk_size=config.token_chunk_size,
        mass_loss_weight=config.tail_mass_weight,
        token_weights=None if target.token_weights is None else target.token_weights.to(device),
    )


@torch.no_grad()
def _monitor(
    student: nn.Module,
    teacher: nn.Module,
    tokens: torch.Tensor,
    *,
    top_k: int,
    device: str,
) -> dict[str, object]:
    student.eval()
    teacher.eval()
    total = TailMassSums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    for row in tokens:
        batch = row.unsqueeze(0).to(device)
        teacher_logits = cast(Any, teacher)(input_ids=batch, use_cache=False).logits[:, :-1]
        student_logits = cast(Any, student)(input_ids=batch, use_cache=False).logits[:, :-1]
        total += _tail_mass_sums(teacher_logits, student_logits, batch[:, 1:], top_k=top_k)
        del batch, teacher_logits, student_logits
    return _means(total)


def _run_arm(
    args: argparse.Namespace,
    config: TopKDistillationConfig,
    calibration: torch.Tensor,
    training_batches: tuple[tuple[TopKTeacherBatch, ...], ...],
    monitor_tokens: torch.Tensor,
    teacher: nn.Module,
    shadow_initialization: tuple[torch.Tensor, torch.Tensor],
    *,
    binary: bool,
) -> dict[str, object]:
    loaded = load_frozen_run(
        args.run_output,
        args.snapshot,
        source_name=MODEL_SOURCE,
        revision=PINNED_MODEL_REVISION,
        device="cpu",
        verify_hashes=True,
        backend="factorized",
        use_global_tuning=False,
    )
    modules = _thaw_frozen_layers(loaded, LocalTensorStore(LocalArtifactStore(args.run_output / "artifacts")))
    key = (args.block, args.owner)
    if key not in modules:
        raise ValueError(f"retained run has no coordinate owner {key}")
    owner = modules[key]
    entry_signs = (
        binary_sign(owner.left_latent.detach()).float(),
        binary_sign(owner.right_latent.detach()).float(),
    )
    if binary:
        if tuple(value.shape for value in shadow_initialization) != (
            owner.left_latent.shape,
            owner.right_latent.shape,
        ):
            raise ValueError("retained shadow initialization differs from the frozen owner")
        shadow_initialization = tuple(
            align_shadow_to_persisted_signs(shadow, entry)
            for shadow, entry in zip(shadow_initialization, entry_signs, strict=True)
        )
        _set_shadow_latents(owner, shadow_initialization)
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    selected = _selected_parameters(owner, binary=binary)
    for _name, parameter in selected:
        parameter.requires_grad_(True)
    student = loaded.model.to(args.device)
    cast(Any, student).config.use_cache = False
    enable_checkpointing = getattr(student, "gradient_checkpointing_enable", None)
    if callable(enable_checkpointing):
        enable_checkpointing()
    enable_inputs = getattr(student, "enable_input_require_grads", None)
    if callable(enable_inputs):
        enable_inputs()
    owner = modules[key]
    entry_device = tuple(
        value.to(args.device)
        for value in (shadow_initialization if binary else entry_signs)
    )
    before = _monitor(student, teacher, monitor_tokens, top_k=config.top_k, device=args.device)
    groups = []
    continuous = [parameter for name, parameter in selected if name not in {"left_latent", "right_latent"}]
    if continuous:
        groups.append({"params": continuous, "lr": args.learning_rate})
    if binary:
        groups.append(
            {
                "params": [owner.left_latent, owner.right_latent],
                "lr": args.binary_learning_rate,
            }
        )
    optimizer = ParityAdamW(groups, lr=args.learning_rate, weight_decay=0.0)
    total_steps = sum(len(epoch) for epoch in training_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    trajectory = []
    started = time.perf_counter()
    for epoch_index, epoch in enumerate(training_batches):
        total_loss = 0.0
        for target in epoch:
            student.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(student, calibration, target, config, args.device)
            if binary and args.anchor_weight:
                anchor = sum(
                    (latent - entry_value).square().mean()
                    for latent, entry_value in zip(
                        (owner.left_latent, owner.right_latent), entry_device, strict=True
                    )
                )
                loss = loss + args.anchor_weight * anchor
            loss.backward()
            optimizer.step()
            scheduler.step()
            if binary:
                retained = project_hamming_budget(
                    (owner.left_latent, owner.right_latent),
                    entry_device,
                    args.hamming_fraction,
                )
            else:
                retained = 0
            total_loss += float(loss.detach())
        trajectory.append(
            {
                "epoch": epoch_index + 1,
                "training_loss": total_loss / len(epoch),
                "hamming_changes": retained,
            }
        )
    after = _monitor(student, teacher, monitor_tokens, top_k=config.top_k, device=args.device)
    final_signs = (
        binary_sign(owner.left_latent.detach()),
        binary_sign(owner.right_latent.detach()),
    )
    result = {
        "arm": "coordinate_binary" if binary else "immutable_sign_control",
        "before": before,
        "after": after,
        "trajectory": trajectory,
        "selected_parameters": [name for name, _parameter in selected],
        "hamming_changes": hamming_changes(
            final_signs, tuple(binary_sign(value) for value in entry_device)
        ),
        "hamming_fraction": hamming_changes(
            final_signs, tuple(binary_sign(value) for value in entry_device)
        )
        / sum(value.numel() for value in entry_device),
        "left_hamming_changes": int((final_signs[0] != binary_sign(entry_device[0])).sum()),
        "right_hamming_changes": int((final_signs[1] != binary_sign(entry_device[1])).sum()),
        "wall_seconds": time.perf_counter() - started,
        "persisted_sign_dtype": "bfloat16",
    }
    student.cpu()
    del student, loaded, modules, owner, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> int:
    if (
        args.block < 0
        or args.epochs <= 0
        or args.maximum_batches_per_epoch <= 0
        or args.learning_rate <= 0
        or args.binary_learning_rate <= 0
        or not 0 <= args.hamming_fraction < 1
        or args.anchor_weight < 0
        or args.monitor_offset < 0
        or args.monitor_samples <= 0
        or args.monitor_sequence_length <= 1
    ):
        raise ValueError("coordinate binary QAT protocol is invalid")
    config = _config(args.run_output, args.epochs, args.maximum_batches_per_epoch)
    left_shadow, right_shadow, factorization_id = _load_shadow_initialization(
        args.run_output,
        args.block,
        args.owner,
        args.expected_blocks,
    )
    calibration = _load_calibration(args.run_output)
    cache = _load_training_cache(args.run_output, epochs=args.epochs)
    training_batches = tuple(
        epoch[: args.maximum_batches_per_epoch] for epoch in cache.epochs
    )
    all_monitor, fingerprint, bos = _split_tokens(
        args.snapshot,
        split="validation",
        samples=args.monitor_offset + args.monitor_samples,
        sequence_length=args.monitor_sequence_length,
        local_files_only=args.local_files_only,
    )
    monitor = all_monitor[args.monitor_offset : args.monitor_offset + args.monitor_samples]
    output: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "role": "analysis-only coordinate binary-QAT functional gate",
        "protocol": {
            "source_run": str(args.run_output.resolve()),
            "block": args.block,
            "owner": args.owner,
            "shadow_factorization_artifact": factorization_id,
            "shadow_initialization": (
                "retained-fp32-pre-hardening-margin-magnitudes-aligned-to-frozen-signs"
            ),
            "epochs": args.epochs,
            "maximum_batches_per_epoch": args.maximum_batches_per_epoch,
            "continuous_learning_rate": args.learning_rate,
            "binary_learning_rate": args.binary_learning_rate,
            "hamming_fraction": args.hamming_fraction,
            "anchor_weight": args.anchor_weight,
            "monitor_offset": args.monitor_offset,
            "monitor_samples": args.monitor_samples,
            "monitor_sequence_length": args.monitor_sequence_length,
            "monitor_dataset_fingerprint": fingerprint,
            "monitor_bos_token_id": bos,
            "distillation": to_dict(config),
        },
        "arms": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, output)
    with acquire_device_lease(args.device), deterministic_torch_execution(config.seed, args.device):
        model_config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
        teacher = load_causal_language_model(
            args.snapshot,
            torch_dtype=torch.bfloat16,
            attention_implementation="eager",
            local_files_only=args.local_files_only,
        ).to(args.device)
        cast(Any, teacher).config.use_cache = False
        for binary in (False, True):
            result = _run_arm(
                args,
                config,
                calibration,
                training_batches,
                monitor,
                teacher,
                (left_shadow, right_shadow),
                binary=binary,
            )
            cast(list[object], output["arms"]).append(result)
            atomic_write_json(args.output, output)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        teacher.cpu()
        del teacher, model_config
        gc.collect()
        torch.cuda.empty_cache()
    output["status"] = "completed"
    atomic_write_json(args.output, output)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
