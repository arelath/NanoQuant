"""Blockwise fixed-sign scale refit of a diagnostic mixed GGUF.

Uses the production post-block refit selector and loss. Standard tensors, signs,
norms and embeddings are immutable. Teacher and student streams are regenerated
from pinned calibration tokens; each completed block has an atomic scale checkpoint.
This is a research continuation, not a resident compression/export workflow.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import gguf
import numpy as np
import torch
from materialize_nanoquant_gguf import _gemma_qkv_sizes, iter_dense_nanoquant_projections
from measure_gguf_kl import NORMS, PROJECTIONS
from safetensors.torch import load_file, save_file
from weight_image_common import _float_tensor, unpack_sign_words

from nanoquant.application.fused_tuning_optimizer import fp32_scale_masters
from nanoquant.application.layers import (
    SharedGroupEvaluation,
    SharedInputProjectionView,
    TrainableFactorizedLinear,
    TrainableSharedInputFactorGroup,
)
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.tuning import TuningRequest, post_block_refit
from nanoquant.config.codec import from_dict, to_dict
from nanoquant.domain.models import ArtifactRef, TensorRef
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import (
    _forward_metadata_to_device,
    _legacy_cuda_numerics,
    _run_block_batched,
    _run_prefix_batched,
)


def token_digest(tokens):
    return hashlib.sha256(tokens.contiguous().numpy().tobytes()).hexdigest()


def block_owners(reader, tensors, index):
    """Decode independent stages, preserving one owner for shared QKV."""
    owners = {}
    prefix = f"blk.{index}."
    for name in sorted(tensors):
        if not name.startswith(prefix) or not name.endswith(".nq_stage0_v"):
            continue
        base = name.removesuffix(".nq_stage0_v")
        count = int(reader.fields[f"quantization.nanoquant.{base}.residual_stage_count"].contents())
        terms = ["nq_stage0", *(f"nq_residual_{i}" for i in range(count))]
        stages = []
        for term in terms:
            stem = f"{base}.{term}"
            scales = [torch.from_numpy(_float_tensor(tensors[f"{stem}_scale_{axis}"], gguf).copy()).flatten()
                      for axis in ("pre", "mid", "post")]
            if any(tensors[f"{stem}_scale_{axis}"].tensor_type != gguf.GGMLQuantizationType.BF16
                   for axis in ("pre", "mid", "post")):
                raise ValueError("this canary requires BF16 scale storage")
            pre, mid, post = scales
            left = torch.from_numpy(unpack_sign_words(tensors[stem + "_u"].data, len(post), len(mid)))
            right = torch.from_numpy(unpack_sign_words(tensors[stem + "_v"].data, len(mid), len(pre)))
            cls = TrainableSharedInputFactorGroup if base.endswith(".attn_qkv") else TrainableFactorizedLinear
            stages.append(cls(left, right, *scales, immutable_binary_factors=True).bfloat16())
        owner = stages[0]
        owner.residual_stages = torch.nn.ModuleList(stages[1:])
        for stage in stages:
            stage.coupled_shared_input_scales = owner.coupled_shared_input_scales
        owners[base] = (owner, terms)
        expected = {f"{base}.{term}_{suffix}" for term in terms
                    for suffix in ("u", "v", "scale_pre", "scale_mid", "scale_post")}
        if {key for key in tensors if key.startswith(base + ".nq_")} != expected:
            raise ValueError("unsupported sidepaths in fixed-sign canary")
    return owners


def scale_state(owners):
    return {f"{base}.{term}_scale_{axis}": getattr(stage, f"scale_{axis}").detach().cpu().bfloat16().contiguous()
            for base, (owner, terms) in owners.items()
            for term, stage in zip(terms, (owner, *owner.residual_terms), strict=True)
            for axis in ("pre", "mid", "post")}


def scale_payloads(tensors, values):
    result = dict(tensors)
    for name, value in values.items():
        tensor = tensors[name]
        if "_scale_" not in name or not torch.isfinite(value).all() or value.numel() != tensor.n_elements:
            raise ValueError(f"invalid scale replacement: {name}")
        data = gguf.quantize(value.float().numpy(), tensor.tensor_type).reshape(tensor.data.shape)
        result[name] = SimpleNamespace(name=name, data=data, tensor_type=tensor.tensor_type,
                                       shape=tensor.shape, n_bytes=tensor.n_bytes, n_elements=tensor.n_elements)
    return result


def install_block(block, reader, tensors, index, owners, originals):
    for base, (owner, _) in owners.items():
        kind = base.split(".")[2]
        owner.to("cuda")
        if kind == "attn_qkv":
            block.add_module("_refit_qkv", owner)
            evaluation = SharedGroupEvaluation(3)
            start = 0
            for suffix, size in zip(("q", "k", "v"), _gemma_qkv_sizes(reader.fields, index), strict=True):
                path = f"self_attn.{suffix}_proj"
                block.set_submodule(path, SharedInputProjectionView(owner, start, start + size,
                                                                    originals[path].in_features, evaluation))
                start += size
        else:
            block.set_submodule(PROJECTIONS[kind], owner)
    with torch.no_grad():
        for kind, path in PROJECTIONS.items():
            name = f"blk.{index}.{kind}.weight"
            if name in tensors:
                value = torch.from_numpy(_float_tensor(tensors[name], gguf).copy())
                originals[path].weight.copy_(value)


def restore_dense(block, reader, tensors, index, owners, originals, scales):
    payloads = scale_payloads(tensors, scales)
    selected = {name: value for name, value in payloads.items() if name.startswith(f"blk.{index}.")}
    if owners:
        with torch.no_grad():
            for name, projection in iter_dense_nanoquant_projections(reader.fields, selected, gguf):
                path = PROJECTIONS[name.split(".")[2]]
                originals[path].weight.copy_(torch.from_numpy(projection.weight.copy()))
    for path, module in originals.items():
        block.set_submodule(path, module)
    if hasattr(block, "_refit_qkv"):
        del block._refit_qkv


def export_scales(source, output, values):
    """Rewrite only existing BF16 scales; all other raw payloads are checked exactly."""
    if output.exists():
        raise ValueError("preserving existing GGUF output")
    reader = gguf.GGUFReader(source)
    original = {tensor.name: tensor for tensor in reader.tensors}
    payloads = scale_payloads(original, values)
    writer = gguf.GGUFWriter(None, reader.fields["general.architecture"].contents(), endianess=reader.endianess)
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        if name == "general.alignment":
            writer.add_custom_alignment(int(field.contents()))
        else:
            writer.add_key_value(name, field.contents(), field.types[0],
                                 sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, payloads[tensor.name].data, raw_dtype=tensor.tensor_type,
                          tensor_endianess=reader.endianess)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".refit-export-") as temporary:
        path = Path(temporary) / output.name
        writer.write_header_to_file(path)
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
        actual = {tensor.name: tensor for tensor in gguf.GGUFReader(path).tensors}
        if set(actual) != set(payloads):
            raise ValueError("refit changed the tensor inventory")
        for name, expected in payloads.items():
            got = actual[name]
            if (got.tensor_type != expected.tensor_type or not np.array_equal(got.shape, expected.shape)
                    or not np.array_equal(got.data, expected.data)):
                raise ValueError(f"refit export mismatch: {name}")
        if path.stat().st_size != source.stat().st_size:
            raise ValueError("scale-only refit changed GGUF bytes")
        del actual, got
        gc.collect()
        path.replace(output)
    changed = sum(not np.array_equal(original[name].data, payloads[name].data) for name in values)
    return {"sha256": hash_file(output), "bytes": output.stat().st_size, "changed_scale_tensors": changed,
            "all_other_payloads_exact": True, "embedding_exact": True, "signs_exact": True,
            "standard_weights_exact": True}


def loss_metrics(prediction, targets, train_count):
    result = {}
    for name, sl in (("train", slice(None, train_count)), ("validation", slice(train_count, None))):
        error = (prediction[sl].float() - targets[sl].float()).square().mean().item()
        energy = targets[sl].float().square().mean().item()
        result[name] = {"mse": error, "normalized_mse": error / max(energy, 1e-12)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-after-blocks", type=int)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    source, run = Path(plan["sources"][args.arm]), Path(plan["run_output"])
    audit = json.loads(Path(plan["audit"]).read_text(encoding="utf-8"))
    if not audit.get("complete") or audit["completed_blocks"] != list(range(26)):
        raise ValueError("requires complete source artifact audit")
    receipt = json.loads((run / "calibration-input.json").read_text())
    calibration = load_pinned_calibration(run, ArtifactRef("calibration-dataset-manifest", receipt["artifact_id"], 1))
    train_rows, validation_rows = plan["train_rows"], plan["validation_rows"]
    if set(train_rows) & set(validation_rows):
        raise ValueError("training and validation rows overlap")
    tokens = calibration.input_ids[train_rows + validation_rows, :plan["sequence_length"]].contiguous()
    identity = {"plan": plan, "arm": args.arm, "source_sha256": hash_file(source),
                "manifest_sha256": hash_file(run / "manifest.json"), "audit_sha256": hash_file(Path(plan["audit"])),
                "token_sha256": token_digest(tokens), "code_sha256": hash_file(Path(__file__))}
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "refit.json"
    report = {"identity": identity, "blocks": {}, "complete": False}
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report["identity"] != identity:
            raise ValueError("refit identity changed; preserve old output and choose a new path")
        if report["complete"]:
            if hash_file(root / "refitted.gguf") != report["export"]["sha256"]:
                raise ValueError("completed export changed")
            return 0
    atomic_write_json(report_path, report)
    store = LocalArtifactStore(run / "artifacts", use_persistent_validation_cache=False)
    calibration_ref = json.loads((run / "state/preprocessing.json").read_text())["calibration"]["artifact_id"]
    stats = json.loads((store.path_for(calibration_ref) / "stats.json").read_text())
    importances = {row["layer"]["block"]["index"]: from_dict(TensorRef, row["output_importance"])
                   for row in stats["layers"] if row["layer"]["path"] == "mlp.down_proj"}
    reader = gguf.GGUFReader(source)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    with acquire_device_lease("cuda"), _legacy_cuda_numerics():
        model = load_causal_language_model(Path(plan["snapshot"]), local_files_only=True,
                                           torch_dtype=torch.bfloat16, attention_implementation="eager").eval()
        model.requires_grad_(False)
        adapter = adapter_for_config(model.config.to_dict())
        embedding_before = model.get_input_embeddings().weight.detach().clone()
        metadata = capture_prefix_invocations(model.model.layers[0],
                     (lambda: adapter.run_decoder_forward(model, tokens[:1]),))[0].keyword
        metadata = _forward_metadata_to_device(metadata, "cuda")
        teacher_inputs = _run_prefix_batched(adapter, model, tokens, 4, "cpu").detach()
        student_inputs = teacher_inputs.clone()
        all_scales = {}
        for index, block in enumerate(model.model.layers):
            if args.stop_after_blocks is not None and index >= args.stop_after_blocks:
                break
            started = time.perf_counter()
            block.cuda()
            with torch.no_grad():
                targets = _run_block_batched(adapter, block, teacher_inputs, metadata, 4, "cpu").detach()
            # The comparison protocols use clean norms. Verify the source does too.
            for kind, path in NORMS.items():
                name = f"blk.{index}.{kind}.weight"
                if name in tensors:
                    reference = block.get_submodule(path).weight.detach().float().cpu().numpy() + 1
                    if not np.allclose(_float_tensor(tensors[name], gguf), reference, rtol=0, atol=1e-5):
                        raise ValueError(f"source norm differs from fixed clean norm: {name}")
            originals = {path: block.get_submodule(path) for path in PROJECTIONS.values()}
            owners = block_owners(reader, tensors, index)
            install_block(block, reader, tensors, index, owners, originals)
            checkpoint = root / f"block-{index:02d}.safetensors"
            entry = report["blocks"].get(str(index))
            if entry is not None:
                if hash_file(checkpoint) != entry["checkpoint_sha256"]:
                    raise ValueError("completed block checkpoint changed")
                scales = load_file(checkpoint)
                if set(scales) != set(scale_state(owners)):
                    raise ValueError("completed scale inventory differs")
                print(f"Reusing block {index}", flush=True)
            else:
                initial_scales = scale_state(owners)
                restore_dense(block, reader, tensors, index, owners, originals, initial_scales)
                with torch.no_grad():
                    before_dense = _run_block_batched(adapter, block, student_inputs, metadata, 4, "cpu").detach()
                before_dense_metrics = loss_metrics(before_dense, targets, len(train_rows))
                del before_dense, initial_scales
                install_block(block, reader, tensors, index, owners, originals)
                with torch.no_grad():
                    before = _run_block_batched(adapter, block, student_inputs, metadata, 4, "cpu").detach()
                before_metrics = loss_metrics(before, targets, len(train_rows))
                del before
                epoch_losses = []
                tuning = None
                if owners:
                    with LocalTensorStore(store).read(importances[index], "cpu") as value:
                        importance = value.clone()
                    request = TuningRequest(
                        student_inputs[:len(train_rows)], targets[:len(train_rows)],
                        plan["epochs"], plan["batch_size"], plan["learning_rate"],
                        output_importance=importance, microbatch_size=plan["microbatch_size"],
                        seed=plan["seed"], restore_best_state=False, epoch_loss_mode="legacy_training",
                        optimizer_backend="fused_adamw_scales_fp32",
                        epoch_observer=lambda epoch, loss, rows=epoch_losses: rows.append(
                            {"epoch": epoch, "loss": loss}),
                    )
                    with fp32_scale_masters(block):
                        tuning = post_block_refit(
                            block, request, lambda module, x: adapter.run_block(module, x, **metadata),
                        )
                with torch.no_grad():
                    after = _run_block_batched(adapter, block, student_inputs, metadata, 4, "cpu").detach()
                after_metrics = loss_metrics(after, targets, len(train_rows))
                del after
                scales = scale_state(owners)
                temporary = checkpoint.with_suffix(".tmp")
                save_file(scales, temporary)
                temporary.replace(checkpoint)
                entry = {"binary_owners": len(owners), "checkpoint_sha256": hash_file(checkpoint),
                         "before_factorized": before_metrics, "after_factorized": after_metrics,
                         "before_dense": before_dense_metrics, "epoch_losses": epoch_losses,
                         "tuning": None if tuning is None else to_dict(tuning)}
            restore_dense(block, reader, tensors, index, owners, originals, scales)
            with torch.no_grad():
                student_outputs = _run_block_batched(adapter, block, student_inputs, metadata, 4, "cpu").detach()
            if str(index) not in report["blocks"]:
                entry.update({"after_persisted_dense": loss_metrics(student_outputs, targets, len(train_rows)),
                              "seconds": time.perf_counter() - started})
                report["blocks"][str(index)] = entry
                atomic_write_json(report_path, report)
            all_scales.update(scales)
            teacher_inputs, student_inputs = targets, student_outputs
            block.cpu()
            del owners, originals, scales
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Block {index} done: {entry['after_persisted_dense']['validation']['normalized_mse']:.6f}, "
                  f"{time.perf_counter()-started:.1f}s", flush=True)
        if not torch.equal(model.get_input_embeddings().weight.detach(), embedding_before):
            raise RuntimeError("embedding changed")
        if len(report["blocks"]) == len(model.model.layers):
            report["export"] = export_scales(source, root / "refitted.gguf", all_scales)
            report["embedding_verified_unchanged"] = True
            report["complete"] = True
            atomic_write_json(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
