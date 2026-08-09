"""Globally tune foldable MLP row/column multipliers against cached top-k targets."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import _paths  # noqa: F401
import torch
from probe_mlp_overlays_kl import _split_tokens
from probe_mlp_policy_frozen_transfer import (
    MODEL_SOURCE,
    PINNED_MODEL_REVISION,
    _evaluate_per_sequence,
    _paired_nll_payload,
)
from torch import nn
from transformers.models.auto.tokenization_auto import AutoTokenizer

from nanoquant.application.distillation import (
    TopKDistillationConfig,
    TopKTeacherBatch,
    TopKTeacherCache,
    cache_topk_teacher_targets,
    topk_distillation_loss,
)
from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.linear_math import functional_factorized_linear, rescale_factorized_terms
from nanoquant.domain.models import ArtifactRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.distillation_cache import (
    TeacherCacheJournal,
    load_teacher_epoch,
)
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.packed_model_loader import load_packed_model
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.kl_budget_workflow import _token_hash
from nanoquant.runtime import (
    OpenProductCodebookArtifact,
    ProductCodebookLayerState,
    write_product_codebook_artifact,
)

MLP_PATHS = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def _module_parent(block: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    current = block
    for part in parts[:-1]:
        child = current[part] if isinstance(current, nn.ModuleDict) else getattr(current, part, None)
        if not isinstance(child, nn.Module):
            raise KeyError(f"module path not found: {path}")
        current = child
    return current, parts[-1]


def _child(parent: nn.Module, name: str) -> nn.Module:
    value = parent[name] if isinstance(parent, nn.ModuleDict) else getattr(parent, name, None)
    if not isinstance(value, nn.Module):
        raise KeyError(f"module child not found: {name}")
    return value


def _set_child(parent: nn.Module, name: str, value: nn.Module) -> None:
    if isinstance(parent, nn.ModuleDict):
        parent[name] = value
    else:
        setattr(parent, name, value)


def _decoder(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose a mutable decoder stack")
    return layers


class FoldableMultiplierLinear(nn.Module):
    """Apply positive input/output multipliers without changing frozen payloads."""

    def __init__(
        self,
        base: FactorizedReferenceLinear,
        *,
        input_family: str | None = None,
        output_family: str | None = None,
    ) -> None:
        super().__init__()
        if input_family is None and output_family is None:
            raise ValueError("foldable multiplier wrapper needs at least one axis")
        self.base = base
        self.input_family = input_family
        self.output_family = output_family
        device = base.scale_pre.device
        self.log_input_multiplier = (
            None
            if input_family is None
            else nn.Parameter(torch.zeros(base.scale_pre.shape, device=device, dtype=torch.float32))
        )
        self.log_output_multiplier = (
            None
            if output_family is None
            else nn.Parameter(torch.zeros(base.scale_post.shape, device=device, dtype=torch.float32))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        # Materialize the *stored* BF16 payload terms dynamically, rather than
        # multiplying BF16 activations by a near-identity value. The latter has
        # a wide quantization dead zone and does not replay the result of folding
        # the same FP32 multiplier into BF16 scales/outliers/patches.
        scaled = rescale_factorized_terms(
            self.base.scale_pre,
            self.base.scale_post,
            input_multiplier=(None if self.log_input_multiplier is None else torch.exp(self.log_input_multiplier)),
            output_multiplier=(None if self.log_output_multiplier is None else torch.exp(self.log_output_multiplier)),
            outlier_indices=self.base.outlier_indices,
            outlier_values=self.base.outlier_values,
            patch_left=self.base.patch_left,
            patch_right=self.base.patch_right,
        )
        return functional_factorized_linear(
            value,
            self.base.left_binary,
            self.base.right_binary,
            scaled.scale_pre,
            self.base.scale_mid,
            scaled.scale_post,
            self.base.bias,
            self.base.outlier_indices,
            scaled.outlier_values,
            self.base.outlier_scales,
            scaled.patch_left,
            scaled.patch_right,
            scale_left_before_linear=True,
        )


@dataclass(frozen=True, slots=True)
class InstalledMultipliers:
    wrappers: dict[tuple[int, str], FoldableMultiplierLinear]
    families: dict[str, tuple[nn.Parameter, ...]]

    @property
    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for values in self.families.values() for parameter in values)


def install_global_mlp_multipliers(model: nn.Module) -> InstalledMultipliers:
    wrappers: dict[tuple[int, str], FoldableMultiplierLinear] = {}
    families: dict[str, list[nn.Parameter]] = defaultdict(list)
    for block_index, block in enumerate(_decoder(model)):
        for path in MLP_PATHS:
            parent, name = _module_parent(block, path)
            base = _child(parent, name)
            if not isinstance(base, FactorizedReferenceLinear):
                raise TypeError(f"global MLP multiplier target is not factorized: {block_index}:{path}")
            if path == "mlp.gate_proj":
                wrapper = FoldableMultiplierLinear(base, output_family="gate_output")
            elif path == "mlp.up_proj":
                wrapper = FoldableMultiplierLinear(base, output_family="up_output")
            else:
                wrapper = FoldableMultiplierLinear(
                    base,
                    input_family="down_input",
                    output_family="down_output",
                )
            _set_child(parent, name, wrapper)
            wrappers[(block_index, path)] = wrapper
            if wrapper.log_input_multiplier is not None:
                assert wrapper.input_family is not None
                families[wrapper.input_family].append(wrapper.log_input_multiplier)
            if wrapper.log_output_multiplier is not None:
                assert wrapper.output_family is not None
                families[wrapper.output_family].append(wrapper.log_output_multiplier)
    return InstalledMultipliers(wrappers, {name: tuple(values) for name, values in families.items()})


def family_identity_penalty(families: dict[str, tuple[nn.Parameter, ...]]) -> torch.Tensor:
    if not families or any(not values for values in families.values()):
        raise ValueError("identity penalty requires non-empty multiplier families")
    means = [torch.cat(tuple(value.reshape(-1) for value in values)).square().mean() for values in families.values()]
    return torch.stack(means).mean()


def multiplier_summary(installed: InstalledMultipliers, log_limit: float) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in sorted(installed.families.items()):
        values = torch.cat(tuple(parameter.detach().float().cpu().reshape(-1) for parameter in parameters))
        multipliers = values.exp()
        result[name] = {
            "count": values.numel(),
            "minimum": float(multipliers.min()),
            "q01": float(torch.quantile(multipliers, 0.01)),
            "median": float(torch.quantile(multipliers, 0.5)),
            "q99": float(torch.quantile(multipliers, 0.99)),
            "maximum": float(multipliers.max()),
            "lower_bound_count": int((values <= -log_limit).sum()),
            "upper_bound_count": int((values >= log_limit).sum()),
        }
    return result


def gradient_summary(installed: InstalledMultipliers) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, parameters in sorted(installed.families.items()):
        gradients = [parameter.grad for parameter in parameters]
        present = [gradient for gradient in gradients if isinstance(gradient, torch.Tensor)]
        result[name] = {
            "parameter_tensors": len(parameters),
            "gradient_tensors": len(present),
            "missing_gradient_tensors": len(parameters) - len(present),
            "nonfinite_gradient_tensors": sum(int(not bool(torch.isfinite(gradient).all())) for gradient in present),
            "zero_gradient_tensors": sum(int(not bool(torch.count_nonzero(gradient))) for gradient in present),
            "gradient_norm": math.sqrt(sum(float(gradient.detach().float().square().sum()) for gradient in present)),
        }
    return result


def _component_values(prefix: str, module: FactorizedReferenceLinear) -> dict[str, torch.Tensor]:
    values = {
        f"{prefix}.scale_pre": module.scale_pre,
        f"{prefix}.scale_post": module.scale_post,
    }
    for name in ("outlier_values", "patch_left", "patch_right"):
        value = getattr(module, name)
        if isinstance(value, torch.Tensor):
            values[f"{prefix}.{name}"] = value
    return {name: value.detach().cpu().contiguous() for name, value in values.items()}


def fold_global_mlp_multipliers(
    model: nn.Module,
    installed: InstalledMultipliers,
) -> tuple[dict[str, torch.Tensor], int]:
    tensors: dict[str, torch.Tensor] = {}
    replaced_bytes = 0
    for (block_index, path), wrapper in sorted(installed.wrappers.items()):
        input_multiplier = None if wrapper.log_input_multiplier is None else wrapper.log_input_multiplier.detach().exp()
        output_multiplier = (
            None if wrapper.log_output_multiplier is None else wrapper.log_output_multiplier.detach().exp()
        )
        base = wrapper.base
        scaled = rescale_factorized_terms(
            base.scale_pre,
            base.scale_post,
            input_multiplier=input_multiplier,
            output_multiplier=output_multiplier,
            outlier_indices=base.outlier_indices,
            outlier_values=base.outlier_values,
            patch_left=base.patch_left,
            patch_right=base.patch_right,
        )
        with torch.no_grad():
            base.scale_pre.copy_(scaled.scale_pre)
            base.scale_post.copy_(scaled.scale_post)
            if base.outlier_values is not None and scaled.outlier_values is not None:
                base.outlier_values.copy_(scaled.outlier_values)
            if base.patch_left is not None and scaled.patch_left is not None:
                base.patch_left.copy_(scaled.patch_left)
            if base.patch_right is not None and scaled.patch_right is not None:
                base.patch_right.copy_(scaled.patch_right)
        parent, name = _module_parent(_decoder(model)[block_index], path)
        _set_child(parent, name, base)
        prefix = f"model.layers.{block_index}.{path}"
        values = _component_values(prefix, base)
        tensors.update(values)
        replaced_bytes += sum(value.numel() * value.element_size() for value in values.values())
    return tensors, replaced_bytes


def _export_component_overlay(
    destination: Path,
    tensors: dict[str, torch.Tensor],
    *,
    frozen_identity: dict[str, str],
    global_tuning: ArtifactRef | None,
    source_overlay: Path,
    replaced_bytes: int,
) -> dict[str, object]:
    if not tensors or any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError("global multiplier component overlay tensors are invalid")
    replacement_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    source_hash = hash_file(source_overlay / "components.safetensors")
    with atomic_workspace(destination) as temporary:
        tensor_path = temporary / "components.safetensors"
        SAFETENSORS.save(tensors, tensor_path)
        manifest = {
            "schema_version": 2,
            "semantics": "replace-existing-factorized-components",
            "source_dense_tensor_sha256": f"phase-c-source-component:{source_hash}",
            "frozen_identity": frozen_identity,
            "global_tuning": None if global_tuning is None else to_dict(global_tuning),
            "policy": {
                str(index): "global-foldable-multipliers" for index in sorted(_decoder_blocks_from_names(tensors))
            },
            "tensor_sha256": hash_file(tensor_path),
            "tensor_count": len(tensors),
            "replaced_payload_bytes": replaced_bytes,
            "replacement_payload_bytes": replacement_bytes,
            "payload_byte_delta": replacement_bytes - replaced_bytes,
            "tensors": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
                for name, value in sorted(tensors.items())
            },
        }
        atomic_write_json(temporary / "manifest.json", manifest)
    return {"directory": str(destination), **manifest}


def fold_product_codebook_states(
    artifact: OpenProductCodebookArtifact,
    installed: InstalledMultipliers,
) -> dict[int, tuple[ProductCodebookLayerState, ...]]:
    """Map folded physical MLP terms back into compact factorization orientation."""

    replacements: dict[int, list[ProductCodebookLayerState]] = defaultdict(list)
    for entry in artifact.manifest.layers:
        prefix = f"blocks.{entry.block}."
        if not entry.spec.name.startswith(prefix):
            raise ValueError(f"product-codebook layer name differs from block: {entry.spec.name}")
        path = entry.spec.name.removeprefix(prefix)
        wrapper = installed.wrappers.get((entry.block, path))
        if wrapper is None:
            raise ValueError(f"product-codebook multiplier target is absent: {entry.spec.name}")
        module = wrapper.base
        state = artifact.load_compact_layer(entry.spec.name)
        if state.factorization_transposed:
            factor_scale_pre = module.scale_post
            factor_scale_post = module.scale_pre
        else:
            factor_scale_pre = module.scale_pre
            factor_scale_post = module.scale_post
        if (state.outlier_values is None) != (module.outlier_values is None):
            raise ValueError(f"product-codebook outlier terms differ: {entry.spec.name}")
        replacements[entry.block].append(
            replace(
                state,
                factor_scale_pre=factor_scale_pre.detach()
                .to(dtype=state.factor_scale_pre.dtype, device="cpu")
                .contiguous(),
                factor_scale_mid=module.scale_mid.detach()
                .to(dtype=state.factor_scale_mid.dtype, device="cpu")
                .contiguous(),
                factor_scale_post=factor_scale_post.detach()
                .to(dtype=state.factor_scale_post.dtype, device="cpu")
                .contiguous(),
                outlier_values=(
                    None
                    if state.outlier_values is None
                    else cast(torch.Tensor, module.outlier_values)
                    .detach()
                    .to(dtype=state.outlier_values.dtype, device="cpu")
                    .contiguous()
                ),
            )
        )
    return {block: tuple(states) for block, states in replacements.items()}


def product_codebook_discrete_payload_equal(
    first: ProductCodebookLayerState,
    second: ProductCodebookLayerState,
) -> bool:
    """Return whether every bit-defining, non-tunable product term is unchanged."""

    return (
        first.spec == second.spec
        and first.factorization_transposed == second.factorization_transposed
        and first.free_rows == second.free_rows
        and torch.equal(first.factor_left_words, second.factor_left_words)
        and torch.equal(first.factor_right_free_words, second.factor_right_free_words)
        and torch.equal(first.factor_right_coded_payload, second.factor_right_coded_payload)
        and torch.equal(first.first_half_words, second.first_half_words)
        and torch.equal(first.second_half_words, second.second_half_words)
        and (
            (first.outlier_indices is None and second.outlier_indices is None)
            or (
                first.outlier_indices is not None
                and second.outlier_indices is not None
                and torch.equal(first.outlier_indices, second.outlier_indices)
            )
        )
    )


def _export_product_codebook_overlay(
    destination: Path,
    source: OpenProductCodebookArtifact,
    replacements: dict[int, tuple[ProductCodebookLayerState, ...]],
    *,
    protocol: dict[str, object],
) -> dict[str, object]:
    by_name = {
        state.spec.name: state for states in replacements.values() for state in states
    }
    if set(by_name) != source.replacement_names:
        raise ValueError("distilled product-codebook layer inventory differs")
    if any(
        not product_codebook_discrete_payload_equal(
            source.load_compact_layer(name), by_name[name]
        )
        for name in sorted(by_name)
    ):
        raise ValueError("distillation modified a discrete product-codebook payload")
    replay = {key: json.loads(value) for key, value in source.manifest.replay}
    replay["scale_axis_distillation"] = protocol
    opened = write_product_codebook_artifact(
        destination,
        source.base,
        replacements,
        allocation_sha256=source.manifest.allocation_sha256,
        allocation_total_bits=source.manifest.allocation_total_bits,
        effective_bpw=source.manifest.effective_bpw,
        correction_source_sha256=hash_file(source.root / "components.safetensors"),
        replay=replay,
    )
    if (
        opened.manifest.compact_mlp_bits != source.manifest.compact_mlp_bits
        or opened.manifest.allocation_total_bits != source.manifest.allocation_total_bits
        or opened.manifest.effective_bpw != source.manifest.effective_bpw
    ):
        raise ValueError("distillation changed the product-codebook bit budget")
    return {
        "directory": str(opened.root),
        "descriptor_sha256": hash_file(
            opened.root / "nanoquant-product-codebook-overlay.json"
        ),
        "tensor_sha256": opened.manifest.tensor_sha256,
        "layer_count": opened.manifest.layer_count,
        "compact_mlp_bits": opened.manifest.compact_mlp_bits,
        "allocation_total_bits": opened.manifest.allocation_total_bits,
        "effective_bpw": opened.manifest.effective_bpw,
        "discrete_payload_unchanged": True,
    }


def _decoder_blocks_from_names(tensors: dict[str, torch.Tensor]) -> set[int]:
    return {int(name.split(".")[2]) for name in tensors}


def _hidden_states(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    text_stack = getattr(model, "model", None)
    if not isinstance(text_stack, nn.Module):
        language_model = getattr(model, "language_model", None)
        text_stack = getattr(language_model, "model", None)
    if not isinstance(text_stack, nn.Module):
        raise TypeError("model does not expose a supported text stack")
    outputs = cast(Any, text_stack)(input_ids=token_ids, use_cache=False)
    value = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, "last_hidden_state", None)
    if not isinstance(value, torch.Tensor):
        raise TypeError("model text stack did not return hidden states")
    return value


def _lm_head(model: nn.Module) -> nn.Module:
    value = getattr(model, "lm_head", None)
    if not isinstance(value, nn.Module):
        raise TypeError("model does not expose an LM head")
    return value


def _load_calibration(run_output: Path) -> torch.Tensor:
    receipt = json.loads((run_output / "calibration-input.json").read_text(encoding="utf-8"))
    reference = ArtifactRef("calibration-dataset-manifest", str(receipt["artifact_id"]), 1)
    return load_pinned_calibration(run_output, reference).input_ids


def _load_training_cache(
    run_output: Path,
    *,
    epochs: int,
) -> TopKTeacherCache:
    payload = json.loads((run_output / "global-distillation-cache.json").read_text(encoding="utf-8"))
    journal = from_dict(TeacherCacheJournal, payload, path="teacher_cache_journal")
    if epochs <= 0 or epochs > len(journal.epochs):
        raise ValueError("requested multiplier epochs exceed the retained teacher cache")
    artifacts = LocalArtifactStore(run_output / "artifacts")
    committed = []
    for reference in journal.epochs[:epochs]:
        if reference is None:
            raise ValueError("retained teacher cache is incomplete")
        committed.append(load_teacher_epoch(reference, journal.identity, artifacts))
    return TopKTeacherCache(tuple(item.batches for item in committed), sum(item.bytes for item in committed))


def _checkpoint_dtype(config: dict[str, object]) -> torch.dtype:
    value = config.get("torch_dtype")
    if not isinstance(value, str):
        return torch.float32
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(value, torch.float32)


def _teacher_topk_entropy(batch: TopKTeacherBatch) -> float:
    probabilities = torch.softmax(batch.top_values.float(), dim=-1)
    entropy = -(probabilities * torch.log_softmax(batch.top_values.float(), dim=-1)).sum(dim=-1)
    if batch.token_weights is None:
        return float(entropy.mean())
    weights = batch.token_weights.float()
    return float((entropy * weights).sum() / weights.sum())


@torch.no_grad()
def _evaluate_topk_kl(
    model: nn.Module,
    tokens: torch.Tensor,
    cache: TopKTeacherCache,
    *,
    device: str,
    token_chunk_size: int,
) -> dict[str, object]:
    losses = []
    entropies = []
    model.eval()
    for target in cache.epochs[0]:
        indices = torch.tensor(target.sample_indices, dtype=torch.long)
        batch = tokens.index_select(0, indices).to(device)
        selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
        hidden = _hidden_states(model, batch).reshape(-1, _hidden_states_width(model))
        hidden = hidden.index_select(0, selected_tokens)
        loss = topk_distillation_loss(
            hidden,
            target.top_values.to(device),
            target.top_indices.to(device=device, dtype=torch.long),
            _lm_head(model),
            temperature=1.0,
            token_chunk_size=token_chunk_size,
            token_weights=None if target.token_weights is None else target.token_weights.to(device),
        )
        losses.append(float(loss))
        entropies.append(_teacher_topk_entropy(target))
    cross_entropy = statistics.fmean(losses)
    entropy = statistics.fmean(entropies)
    return {
        "cross_entropy": cross_entropy,
        "teacher_entropy": entropy,
        "topk_kl": cross_entropy - entropy,
        "batch_count": len(losses),
    }


def _hidden_states_width(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    width = getattr(config, "hidden_size", None)
    if not isinstance(width, int):
        raise TypeError("model config does not expose hidden size")
    return width


def _capture_state(installed: InstalledMultipliers) -> tuple[torch.Tensor, ...]:
    return tuple(parameter.detach().cpu().clone() for parameter in installed.parameters)


def _restore_state(installed: InstalledMultipliers, state: tuple[torch.Tensor, ...]) -> None:
    if len(state) != len(installed.parameters):
        raise ValueError("multiplier checkpoint differs from installed parameter inventory")
    with torch.no_grad():
        for parameter, value in zip(installed.parameters, state, strict=True):
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _monitor(
    model: nn.Module,
    tokens: torch.Tensor,
    cache: TopKTeacherCache,
    *,
    device: str,
    token_chunk_size: int,
) -> dict[str, object]:
    nll = _evaluate_per_sequence(model, tokens, device)
    topk = _evaluate_topk_kl(model, tokens, cache, device=device, token_chunk_size=token_chunk_size)
    return {"nll": nll, "teacher_topk": topk}


def _paired_standard_error(first: dict[str, object], second: dict[str, object]) -> float:
    first_sequences = first.get("sequences")
    second_sequences = second.get("sequences")
    if (
        not isinstance(first_sequences, list)
        or not isinstance(second_sequences, list)
        or len(first_sequences) != len(second_sequences)
        or len(first_sequences) < 2
    ):
        return 0.0
    differences = [
        float(left["mean_negative_log_likelihood"]) - float(right["mean_negative_log_likelihood"])
        for left, right in zip(first_sequences, second_sequences, strict=True)
    ]
    return statistics.stdev(differences) / math.sqrt(len(differences))


def _select_one_se(checkpoints: list[dict[str, object]]) -> int:
    def _mean_nll(index: int) -> float:
        monitor = cast(dict[str, Any], checkpoints[index]["monitor"])
        return float(monitor["nll"]["mean_negative_log_likelihood"])

    best = min(range(len(checkpoints)), key=_mean_nll)
    best_monitor = cast(dict[str, Any], checkpoints[best]["monitor"])
    baseline_kl = float(cast(dict[str, Any], checkpoints[0]["monitor"])["teacher_topk"]["topk_kl"])
    for index, checkpoint in enumerate(checkpoints):
        monitor = cast(dict[str, Any], checkpoint["monitor"])
        candidate_nll = cast(dict[str, object], monitor["nll"])
        best_nll = cast(dict[str, object], best_monitor["nll"])
        paired_error = _paired_standard_error(candidate_nll, best_nll)
        if (
            float(candidate_nll["mean_negative_log_likelihood"])
            <= float(best_nll["mean_negative_log_likelihood"]) + paired_error
            and float(monitor["teacher_topk"]["topk_kl"]) <= baseline_kl + 1e-6
        ):
            return index
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--component-overlay", type=Path)
    parser.add_argument("--output-component-overlay", type=Path)
    parser.add_argument("--base-packed", type=Path)
    parser.add_argument("--product-codebook-overlay", type=Path)
    parser.add_argument("--output-product-codebook-overlay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--maximum-steps-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--identity-penalty", type=float, default=100.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--multiplier-limit", type=float, default=4.0)
    parser.add_argument("--monitor-split", choices=("test", "validation"), default="validation")
    parser.add_argument("--monitor-offset", type=int, default=72)
    parser.add_argument("--monitor-samples", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--monitor-topk-tokens", type=int, default=512)
    parser.add_argument("--monitor-every-steps", type=int)
    parser.add_argument("--select-checkpoint", choices=("one_se", "final"), default="one_se")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    component_mode = args.component_overlay is not None
    product_mode = args.product_codebook_overlay is not None
    if (
        args.epochs <= 0
        or args.learning_rate <= 0
        or args.identity_penalty < 0
        or args.gradient_clip <= 0
        or args.multiplier_limit <= 1
        or args.monitor_offset < 0
        or args.monitor_samples <= 0
        or args.sequence_length <= 1
        or args.monitor_topk_tokens <= 0
        or (args.maximum_steps_per_epoch is not None and args.maximum_steps_per_epoch <= 0)
        or (args.monitor_every_steps is not None and args.monitor_every_steps <= 0)
        or component_mode == product_mode
        or component_mode != (args.output_component_overlay is not None)
        or product_mode
        != (
            args.base_packed is not None
            and args.output_product_codebook_overlay is not None
        )
    ):
        raise ValueError("global foldable multiplier protocol is invalid")
    started = time.perf_counter()
    device = args.device
    config = json.loads((args.snapshot / "config.json").read_text(encoding="utf-8"))
    adapter = adapter_for_config(config)
    all_tokens, fingerprint, bos_token_id = _split_tokens(
        args.snapshot,
        split=args.monitor_split,
        samples=args.monitor_offset + args.monitor_samples,
        sequence_length=args.sequence_length,
        local_files_only=args.local_files_only,
    )
    monitor_tokens = all_tokens[args.monitor_offset : args.monitor_offset + args.monitor_samples].contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=args.local_files_only)
    monitor_config = TopKDistillationConfig(
        epochs=1,
        batch_size=1,
        learning_rate=args.learning_rate,
        top_k=64,
        maximum_tokens_per_batch=args.monitor_topk_tokens,
        gradient_checkpointing=False,
    )
    print("capturing held-out teacher top-k targets", flush=True)
    teacher = load_causal_language_model(
        args.snapshot,
        torch_dtype=_checkpoint_dtype(config),
        attention_implementation=adapter.attention_implementation,
    ).to(device)
    cast(Any, teacher).config.use_cache = False
    monitor_cache = cache_topk_teacher_targets(
        teacher,
        monitor_tokens,
        _lm_head(teacher),
        _hidden_states,
        monitor_config,
        device=device,
        pad_token_id=tokenizer.pad_token_id,
    )
    teacher.cpu()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    training_tokens = _load_calibration(args.run_output)
    training_cache = _load_training_cache(args.run_output, epochs=args.epochs)
    if product_mode:
        print("loading packed product-codebook student", flush=True)
        loaded_packed = load_packed_model(
            args.base_packed,
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device=device,
            backend="factorized",
            use_global_tuning=True,
            product_codebook_overlay=args.product_codebook_overlay,
        )
        if loaded_packed.product_codebook is None:
            raise ValueError("packed product-codebook student did not retain its overlay")
        student = loaded_packed.model
        loaded_identity = loaded_packed.identity
        loaded_global_tuning = loaded_packed.global_tuning
        product_artifact = loaded_packed.product_codebook
    else:
        print("loading factorized student and component initialization", flush=True)
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=MODEL_SOURCE,
            revision=args.model_revision,
            device=device,
            backend="factorized",
            use_global_tuning=True,
            component_overlay=args.component_overlay,
        )
        student = loaded.model
        loaded_identity = loaded.identity
        loaded_global_tuning = loaded.global_tuning
        product_artifact = None
    cast(Any, student).config.use_cache = False
    installed = install_global_mlp_multipliers(student)
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in installed.parameters:
        parameter.requires_grad_(True)
    if not args.disable_gradient_checkpointing:
        enable_checkpointing = getattr(student, "gradient_checkpointing_enable", None)
        if callable(enable_checkpointing):
            enable_checkpointing()
        enable_input_gradients = getattr(student, "enable_input_require_grads", None)
        if callable(enable_input_gradients):
            enable_input_gradients()

    expected_count = sum(
        (0 if wrapper.log_input_multiplier is None else wrapper.base.scale_pre.numel())
        + (0 if wrapper.log_output_multiplier is None else wrapper.base.scale_post.numel())
        for wrapper in installed.wrappers.values()
    )
    actual_count = sum(parameter.numel() for parameter in installed.parameters)
    if actual_count != expected_count:
        raise ValueError(f"unexpected global MLP multiplier count: {actual_count} != {expected_count}")
    log_limit = math.log(args.multiplier_limit)
    baseline_monitor = _monitor(
        student,
        monitor_tokens,
        monitor_cache,
        device=device,
        token_chunk_size=monitor_config.token_chunk_size,
    )
    checkpoints: list[dict[str, object]] = [
        {
            "epoch": 0,
            "steps": 0,
            "state": _capture_state(installed),
            "monitor": baseline_monitor,
            "multiplier_summary": multiplier_summary(installed, log_limit),
        }
    ]
    parameters = list(installed.parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    steps_per_epoch = [min(len(epoch), args.maximum_steps_per_epoch or len(epoch)) for epoch in training_cache.epochs]
    total_steps = sum(steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    cpu_tokens = training_tokens.detach().cpu()
    step = 0
    epoch_losses = []
    gradient_norms = []
    gradient_checks: list[dict[str, object]] = []
    student.train()
    for epoch_index, epoch in enumerate(training_cache.epochs):
        student.train()
        total_loss = 0.0
        for target in epoch[: steps_per_epoch[epoch_index]]:
            sample_indices = torch.tensor(target.sample_indices, dtype=torch.long)
            batch = cpu_tokens.index_select(0, sample_indices).to(device)
            selected_tokens = target.token_indices.to(device=device, dtype=torch.long)
            hidden = (
                _hidden_states(student, batch)
                .reshape(-1, _hidden_states_width(student))
                .index_select(0, selected_tokens)
            )
            kd_loss = topk_distillation_loss(
                hidden,
                target.top_values.to(device),
                target.top_indices.to(device=device, dtype=torch.long),
                _lm_head(student),
                temperature=1.0,
                token_chunk_size=128,
                token_weights=None if target.token_weights is None else target.token_weights.to(device),
            )
            penalty = family_identity_penalty(installed.families)
            loss = kd_loss + args.identity_penalty * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            coverage = gradient_summary(installed)
            if any(
                int(cast(dict[str, object], family)["missing_gradient_tensors"]) > 0
                or int(cast(dict[str, object], family)["nonfinite_gradient_tensors"]) > 0
                for family in coverage.values()
            ):
                raise FloatingPointError(f"global multiplier gradient coverage is invalid: {coverage}")
            if step == 0 or step + 1 == total_steps:
                gradient_checks.append({"step": step + 1, "families": coverage})
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                raise FloatingPointError("global multiplier training produced a non-finite loss or gradient")
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for parameter in parameters:
                    parameter.clamp_(min=-log_limit, max=log_limit)
            total_loss += float(kd_loss.detach())
            gradient_norms.append(float(gradient_norm.detach()))
            step += 1
            if step == 1 or step % 32 == 0 or step == total_steps:
                print(
                    f"step {step}/{total_steps}: kd={float(kd_loss.detach()):.6f} "
                    f"penalty={float(penalty.detach()):.6g} "
                    f"grad={float(gradient_norm.detach()):.6g}",
                    flush=True,
                )
            del batch, hidden, kd_loss, penalty, loss
            if args.monitor_every_steps is not None and step % args.monitor_every_steps == 0 and step < total_steps:
                interval_monitor = _monitor(
                    student,
                    monitor_tokens,
                    monitor_cache,
                    device=device,
                    token_chunk_size=monitor_config.token_chunk_size,
                )
                checkpoints.append(
                    {
                        "epoch": epoch_index + 1,
                        "epoch_batch": step - sum(steps_per_epoch[:epoch_index]),
                        "steps": step,
                        "state": _capture_state(installed),
                        "monitor": interval_monitor,
                        "multiplier_summary": multiplier_summary(installed, log_limit),
                    }
                )
                print(
                    f"monitor step {step}/{total_steps}: "
                    f"nll={interval_monitor['nll']['mean_negative_log_likelihood']:.6f} "  # type: ignore[index]
                    f"topk_kl={interval_monitor['teacher_topk']['topk_kl']:.6f}",  # type: ignore[index]
                    flush=True,
                )
                student.train()
        epoch_loss = total_loss / steps_per_epoch[epoch_index]
        epoch_losses.append(epoch_loss)
        if checkpoints[-1]["steps"] == step:
            monitor = cast(dict[str, object], checkpoints[-1]["monitor"])
        else:
            monitor = _monitor(
                student,
                monitor_tokens,
                monitor_cache,
                device=device,
                token_chunk_size=monitor_config.token_chunk_size,
            )
            checkpoints.append(
                {
                    "epoch": epoch_index + 1,
                    "steps": step,
                    "state": _capture_state(installed),
                    "monitor": monitor,
                    "multiplier_summary": multiplier_summary(installed, log_limit),
                }
            )
        print(
            f"epoch {epoch_index + 1}/{args.epochs}: kd={epoch_loss:.6f} "
            f"heldout_nll={monitor['nll']['mean_negative_log_likelihood']:.6f} "  # type: ignore[index]
            f"heldout_topk_kl={monitor['teacher_topk']['topk_kl']:.6f}",  # type: ignore[index]
            flush=True,
        )

    selected_index = len(checkpoints) - 1 if args.select_checkpoint == "final" else _select_one_se(checkpoints)
    selected_checkpoint = checkpoints[selected_index]
    _restore_state(installed, cast(tuple[torch.Tensor, ...], selected_checkpoint["state"]))
    unfolded_monitor = _monitor(
        student,
        monitor_tokens,
        monitor_cache,
        device=device,
        token_chunk_size=monitor_config.token_chunk_size,
    )
    tensors, replaced_bytes = fold_global_mlp_multipliers(student, installed)
    folded_monitor = _monitor(
        student,
        monitor_tokens,
        monitor_cache,
        device=device,
        token_chunk_size=monitor_config.token_chunk_size,
    )
    frozen_identity = {
        "model_hash": loaded_identity.model_hash,
        "config_hash": loaded_identity.config_hash,
        "plan_hash": loaded_identity.plan_hash,
    }
    distillation_protocol = {
        "epochs": args.epochs,
        "steps": step,
        "selected_epoch": selected_checkpoint["epoch"],
        "selected_steps": selected_checkpoint["steps"],
        "learning_rate": args.learning_rate,
        "identity_penalty": args.identity_penalty,
        "gradient_clip": args.gradient_clip,
        "multiplier_limit": args.multiplier_limit,
        "teacher_objective": "top_k",
        "top_k": 64,
    }
    if product_artifact is None:
        assert args.output_component_overlay is not None
        assert args.component_overlay is not None
        overlay = _export_component_overlay(
            args.output_component_overlay,
            tensors,
            frozen_identity=frozen_identity,
            global_tuning=loaded_global_tuning,
            source_overlay=args.component_overlay,
            replaced_bytes=replaced_bytes,
        )
        overlay_kind = "component_overlay"
    else:
        assert args.output_product_codebook_overlay is not None
        replacements = fold_product_codebook_states(product_artifact, installed)
        overlay = _export_product_codebook_overlay(
            args.output_product_codebook_overlay,
            product_artifact,
            replacements,
            protocol=distillation_protocol,
        )
        overlay_kind = "product_codebook_overlay"
    report_checkpoints = [
        {key: value for key, value in checkpoint.items() if key != "state"} for checkpoint in checkpoints
    ]
    report = {
        "schema_version": 1,
        "role": "analysis-only global foldable MLP multiplier tuning",
        "source": {
            "run_output": str(args.run_output),
            "component_overlay": (
                None if args.component_overlay is None else str(args.component_overlay)
            ),
            "base_packed": None if args.base_packed is None else str(args.base_packed),
            "product_codebook_overlay": (
                None
                if args.product_codebook_overlay is None
                else str(args.product_codebook_overlay)
            ),
            "global_tuning": (
                None if loaded_global_tuning is None else to_dict(loaded_global_tuning)
            ),
            "frozen_identity": frozen_identity,
        },
        "protocol": {
            "epochs": args.epochs,
            "steps_per_epoch": steps_per_epoch,
            "learning_rate": args.learning_rate,
            "identity_penalty": args.identity_penalty,
            "gradient_clip": args.gradient_clip,
            "multiplier_limit": args.multiplier_limit,
            "selected_parameter_tensors": len(installed.parameters),
            "selected_parameter_count": actual_count,
            "teacher_cache_bytes_loaded": training_cache.bytes,
            "monitor_split": args.monitor_split,
            "monitor_offset": args.monitor_offset,
            "monitor_samples": args.monitor_samples,
            "sequence_length": args.sequence_length,
            "monitor_token_hash": _token_hash(monitor_tokens),
            "monitor_dataset_fingerprint": fingerprint,
            "monitor_bos_token_id": bos_token_id,
            "monitor_every_steps": args.monitor_every_steps,
            "selection": args.select_checkpoint,
            "gradient_checkpointing": not args.disable_gradient_checkpointing,
        },
        "training": {
            "epoch_losses": epoch_losses,
            "steps": step,
            "gradient_norm_minimum": min(gradient_norms),
            "gradient_norm_maximum": max(gradient_norms),
            "gradient_checks": gradient_checks,
            "checkpoints": report_checkpoints,
            "selected_checkpoint_index": selected_index,
            "selected_epoch": selected_checkpoint["epoch"],
            "selected_steps": selected_checkpoint["steps"],
        },
        "selection_comparison": {
            "baseline_to_unfolded": _paired_nll_payload(baseline_monitor["nll"], unfolded_monitor["nll"]),  # type: ignore[arg-type]
            "unfolded_to_folded": _paired_nll_payload(unfolded_monitor["nll"], folded_monitor["nll"]),  # type: ignore[arg-type]
            "unfolded_monitor": unfolded_monitor,
            "folded_monitor": folded_monitor,
        },
        overlay_kind: overlay,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected_epoch": selected_checkpoint["epoch"],
                "selected_steps": selected_checkpoint["steps"],
                "overlay": overlay["directory"],
            },
            indent=2,
        )
    )
    return 0


def main(arguments: list[str] | None = None) -> int:
    args = _parser().parse_args(arguments)
    if args.device.startswith("cuda"):
        with acquire_device_lease(args.device):
            return run(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
