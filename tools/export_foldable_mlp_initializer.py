"""Convert an accepted dense/component MLP refit into a portable multiplier seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import _paths  # noqa: F401
import torch
from probe_factor_compatible_mlp_refit import _axis_scales, _fit_axes, _module_at_path
from probe_mlp_policy_frozen_transfer import MODEL_SOURCE, PINNED_MODEL_REVISION, _load_overlay
from torch import nn

from nanoquant.application.layers import FactorizedReferenceLinear
from nanoquant.config.codec import to_dict
from nanoquant.domain.linear_math import rescale_factorized_terms
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.factorized_component_overlay import load_factorized_component_overlay
from nanoquant.infrastructure.frozen_model_loader import load_frozen_run
from nanoquant.infrastructure.io_utils import atomic_workspace, atomic_write_json, hash_file
from nanoquant.infrastructure.safetensors_io import SAFETENSORS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--dense-overlay", type=Path, required=True)
    parser.add_argument("--component-overlay", type=Path, required=True)
    parser.add_argument("--output-initializer", type=Path, required=True)
    parser.add_argument("--model-source", default=MODEL_SOURCE)
    parser.add_argument("--model-revision", default=PINNED_MODEL_REVISION)
    parser.add_argument("--fit-iterations", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _decoder(model: nn.Module) -> nn.ModuleList:
    base = getattr(model, "model", None)
    layers = getattr(base, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("model does not expose decoder blocks")
    return layers


def _scaled_components(
    module: FactorizedReferenceLinear,
    *,
    rows: torch.Tensor,
    columns: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scaled = rescale_factorized_terms(
        module.scale_pre,
        module.scale_post,
        input_multiplier=columns,
        output_multiplier=rows,
        outlier_indices=module.outlier_indices,
        outlier_values=module.outlier_values,
        patch_left=module.patch_left,
        patch_right=module.patch_right,
    )
    values = {
        "scale_pre": scaled.scale_pre,
        "scale_post": scaled.scale_post,
        "outlier_values": scaled.outlier_values,
        "patch_left": scaled.patch_left,
        "patch_right": scaled.patch_right,
    }
    return {
        name: value.detach().cpu()
        for name, value in values.items()
        if isinstance(value, torch.Tensor)
    }


def run(args: argparse.Namespace) -> int:
    if args.fit_iterations <= 0:
        raise ValueError("initializer export fit iterations must be positive")
    dense_tensors, dense_manifest = _load_overlay(args.dense_overlay)
    with acquire_device_lease(args.device):
        loaded = load_frozen_run(
            args.run_output,
            args.snapshot,
            source_name=args.model_source,
            revision=args.model_revision,
            device=args.device,
            backend="factorized",
            use_global_tuning=True,
        )
        identity = {
            "model_hash": loaded.identity.model_hash,
            "config_hash": loaded.identity.config_hash,
            "plan_hash": loaded.identity.plan_hash,
        }
        component = load_factorized_component_overlay(
            args.component_overlay,
            frozen_identity=identity,
            global_tuning=loaded.global_tuning,
        )
        policy = {int(block): str(choice) for block, choice in component.manifest["policy"].items()}
        if {int(name.split(".")[2]) for name in dense_tensors} != set(policy):
            raise ValueError("dense overlay blocks differ from the accepted component policy")
        decoder = _decoder(loaded.model)
        initializer: dict[str, torch.Tensor] = {}
        replay: dict[str, object] = {}
        for tensor_name, target in sorted(dense_tensors.items()):
            logical = tensor_name.removeprefix("model.layers.").removesuffix(".weight")
            block_text, path = logical.split(".", maxsplit=1)
            block = int(block_text)
            module = _module_at_path(decoder[block], path)
            if not isinstance(module, FactorizedReferenceLinear):
                raise TypeError(f"initializer source is not factorized: {block}:{path}")
            fit_rows, fit_columns = _fit_axes(path, policy[block])
            source = module.dense_weight().detach()
            rows, columns = _axis_scales(
                source,
                target.to(device=source.device, dtype=source.dtype),
                fit_rows=fit_rows,
                fit_columns=fit_columns,
                iterations=args.fit_iterations,
            )
            prefix = tensor_name.removesuffix(".weight")
            if fit_columns:
                initializer[f"{prefix}.input_log_multiplier"] = columns.log().detach().cpu()
            if fit_rows:
                initializer[f"{prefix}.output_log_multiplier"] = rows.log().detach().cpu()
            scaled = _scaled_components(module, rows=rows, columns=columns)
            expected = {
                name.removeprefix(prefix + "."): value
                for name, value in component.tensors.items()
                if name.startswith(prefix + ".")
            }
            if set(scaled) != set(expected):
                raise ValueError(f"accepted component inventory differs for {prefix}")
            differences = {
                name: float((scaled[name].float() - expected[name].float()).abs().max())
                for name in scaled
            }
            exact = all(torch.equal(scaled[name], expected[name]) for name in scaled)
            replay[prefix] = {
                "exact": exact,
                "maximum_absolute_error": max(differences.values(), default=0.0),
                "components": differences,
            }
            if not exact:
                raise ValueError(f"initializer does not exactly replay accepted components: {prefix}")

        with atomic_workspace(args.output_initializer, replace_existing=True) as temporary:
            tensor_path = temporary / "multipliers.safetensors"
            SAFETENSORS.save(initializer, tensor_path)
            digest = hash_file(tensor_path)
            multipliers = torch.cat(tuple(value.exp() for value in initializer.values()))
            atomic_write_json(
                temporary / "manifest.json",
                {
                    "schema_version": 1,
                    "semantics": "foldable-mlp-log-multiplier-initializer",
                    "model": {"source": args.model_source, "revision": args.model_revision},
                    "tensor_sha256": digest,
                    "tensor_count": len(initializer),
                    "tensors": {
                        name: {"shape": list(value.shape), "dtype": "float32"}
                        for name, value in sorted(initializer.items())
                    },
                    "policy": {str(block): choice for block, choice in sorted(policy.items())},
                    "multiplier_summary": {
                        "count": multipliers.numel(),
                        "minimum": float(multipliers.min()),
                        "median": float(torch.quantile(multipliers, 0.5)),
                        "maximum": float(multipliers.max()),
                    },
                    "source": {
                        "run_output": str(args.run_output.resolve()),
                        "frozen_identity": identity,
                        "global_tuning": to_dict(loaded.global_tuning),
                        "dense_overlay_sha256": dense_manifest["tensor_sha256"],
                        "component_overlay_sha256": component.manifest["tensor_sha256"],
                        "fit_iterations": args.fit_iterations,
                    },
                    "accepted_component_replay": replay,
                },
            )
    print(hash_file(args.output_initializer / "multipliers.safetensors"), flush=True)
    return 0


def main() -> int:
    return run(_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
