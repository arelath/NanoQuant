"""Profile block-zero tuning using the saved recipe and calibration.

Load the source model on CPU and move one decoder block to CUDA. Initialize the
gate from frozen factors or saved continuous ADMM latents and fitted scales.
Remaining projections use source BF16 weights, so this is an isolated replay.
Compiled mode resets weights after a cold invocation before timing a warm run.
Micro mode includes synchronized per-phase profiling overhead.
"""

import argparse
import copy
import gc
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import _paths  # noqa: F401
import torch
from safetensors.torch import save_file

import nanoquant.application.parity_adamw as adamw_module
import nanoquant.application.tuning as tuning_module
from nanoquant.application.layers import LayerFreezer, TrainableFactorizedLinear, thaw_residual_stages
from nanoquant.application.prefix_capture import capture_prefix_invocations
from nanoquant.application.tuning import (
    FactorizedTuningLearningRates,
    TuningRequest,
    tune_factorized,
    tune_non_factorized,
)
from nanoquant.config.codec import from_dict
from nanoquant.config.schema import ProfilingConfig, ProfilingLevel
from nanoquant.domain.models import ArtifactRef, FrozenNanoQuantState, TensorRef
from nanoquant.domain.profiling import NULL_RECORDER
from nanoquant.infrastructure.artifacts import LocalArtifactStore
from nanoquant.infrastructure.device_lease import acquire_device_lease
from nanoquant.infrastructure.hf_calibration_dataset import load_pinned_calibration
from nanoquant.infrastructure.hf_language_model import load_causal_language_model
from nanoquant.infrastructure.model_adapters import adapter_for_config
from nanoquant.infrastructure.profiling import Profiler
from nanoquant.infrastructure.safetensors_io import SAFETENSORS
from nanoquant.infrastructure.tensor_store import LocalTensorStore
from nanoquant.resident_quantization import (
    _forward_metadata_to_device,
    _legacy_cuda_numerics,
    _run_block_batched,
    _run_prefix_batched,
)


@contextmanager
def optimizer_backend(name):
    """Isolate experimental optimizer substitution from production and resume."""
    original_type = tuning_module.ParityAdamW
    original_update = adamw_module._foreach_update
    try:
        if name in {"torch-fused", "torch-fused-fp32", "torch-fused-scales-fp32"}:
            tuning_module.ParityAdamW = partial(torch.optim.AdamW, betas=(0.9, 0.99), eps=1e-6, fused=True)
        elif name == "compiled-parity":
            compiled_update = torch.compile(
                original_update, fullgraph=True,
                options={"emulate_precision_casts": True, "triton.cudagraphs": False},
            )

            def dynamic_update(*args, **kwargs):
                for key in ("beta1", "beta2", "learning_rate"):
                    kwargs[key] = torch.tensor(kwargs[key], dtype=torch.float64)
                return compiled_update(*args, **kwargs)

            adamw_module._foreach_update = dynamic_update
        yield
    finally:
        tuning_module.ParityAdamW = original_type
        adamw_module._foreach_update = original_update


def load_admm_gate(store, layer, records):
    """Recover the uniquely ordered stage/scale pairs before the layer commit."""
    primary_id = layer["factorization"]["artifact_id"]
    committed = next(row for row in records if row.get("kind") == "layer"
                     and row.get("block") == 0 and row.get("layer") == "mlp.gate_proj")
    start = store.validate(primary_id).committed_at
    end = store.validate(committed["artifact_id"]).committed_at
    candidates = []
    for path in store.root.glob("*/*/descriptor.json"):
        descriptor = json.loads(path.read_text())
        if start <= descriptor["committed_at"] < end and descriptor["artifact_type"] in {
            "factorization-attempt", "scale-fit",
        }:
            candidates.append(descriptor)
    candidates.sort(key=lambda item: item["committed_at"])
    stages = 1 + len(layer["frozen_state"]["residual_stages"])
    if ([item["artifact_type"] for item in candidates] != ["factorization-attempt", "scale-fit"] * stages
            or candidates[0]["artifact_id"] != primary_id):
        raise ValueError("cannot uniquely recover this layer's ADMM/scale-fit stage pairs")
    modules = []
    for index in range(stages):
        factor_id = candidates[2 * index]["artifact_id"]
        scale_id = candidates[2 * index + 1]["artifact_id"]
        store.validate(factor_id)
        store.validate(scale_id)
        factors = SAFETENSORS.load(store.path_for(factor_id) / "tensors.safetensors")
        scales = SAFETENSORS.load(store.path_for(scale_id) / "tensors.safetensors")
        modules.append(TrainableFactorizedLinear(
            factors["left_latent"], factors["right_latent"],
            scales["scale_pre"], scales["scale_mid"], scales["scale_post"],
        ).to(dtype=torch.bfloat16))
    modules[0].residual_stages = torch.nn.ModuleList(modules[1:])
    return modules[0], [item["artifact_id"] for item in candidates]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("eager", "compiled", "profile", "micro"), default="eager")
    parser.add_argument("--stage", choices=("factorized", "nonfactorized", "both"), default="both")
    parser.add_argument(
        "--optimizer",
        choices=("parity", "compiled-parity", "torch-fused", "torch-fused-fp32", "torch-fused-scales-fp32"),
        default="parity",
    )
    parser.add_argument("--initialization", choices=("frozen", "admm"), default="frozen")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.optimizer in {"torch-fused-fp32", "torch-fused-scales-fp32"} and args.stage != "factorized":
        parser.error("FP32 master factors currently require --stage factorized")
    run, snapshot, out = args.run_output, args.snapshot, args.output
    out.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((run / "manifest.json").read_text())["resolved_config"]["canonical_run_config"]
    tuning = recipe["block_tuning"]
    if tuning["epoch_loss_mode"] != "legacy_training" or tuning["restore_best_state"]:
        raise ValueError("This diagnostic requires legacy training loss without best-state restoration")

    with acquire_device_lease("cuda"), _legacy_cuda_numerics(), optimizer_backend(args.optimizer):
        tokens = load_pinned_calibration(
            run,
            ArtifactRef(
                "calibration-dataset-manifest",
                json.loads((run / "calibration-input.json").read_text())["artifact_id"],
                1,
            ),
        ).input_ids
        print("tokens loaded", flush=True)
        model = load_causal_language_model(
            snapshot, local_files_only=True, torch_dtype=torch.bfloat16, attention_implementation="eager",
        ).eval()
        adapter = adapter_for_config(model.config.to_dict())
        block = model.model.layers[0]
        metadata = capture_prefix_invocations(
            block, (lambda model=model: adapter.run_decoder_forward(model, tokens[:1]),)
        )[0].keyword
        inputs = _run_prefix_batched(adapter, model, tokens, 1, "cpu").detach()
        block = copy.deepcopy(block).cuda()
        metadata = _forward_metadata_to_device(metadata, "cuda")
        del model
        gc.collect()
        targets = _run_block_batched(adapter, block, inputs, metadata, 1, "cpu").detach()
        block = block.cpu()
        torch.cuda.empty_cache()
        store = LocalArtifactStore(run / "artifacts", use_persistent_validation_cache=False)
        records = [json.loads(line) for line in (run / "state/journal.jsonl").read_text().splitlines()]
        record = next(row for row in reversed(records) if row.get("kind") == "block" and row.get("block") == 0)
        store.validate(record["artifact_id"])
        result = json.loads((store.path_for(record["artifact_id"]) / "block-result.json").read_text())
        layer = next(x for x in result["layers"] if x["layer"]["path"] == "mlp.gate_proj")

        state = from_dict(FrozenNanoQuantState, layer["frozen_state"])
        frozen = LayerFreezer().load(state, LocalTensorStore(store)).module
        source_gate_weight = block.mlp.gate_proj.weight.detach().clone()
        block.mlp.gate_proj = TrainableFactorizedLinear(
            frozen.left_binary,
            frozen.right_binary,
            frozen.scale_pre,
            frozen.scale_mid,
            frozen.scale_post,
            residual_stages=thaw_residual_stages(frozen),
        ).to(dtype=torch.bfloat16)
        del frozen
        initialization_artifacts = []
        if args.initialization == "admm":
            block.mlp.gate_proj, initialization_artifacts = load_admm_gate(store, layer, records)
        calibration_ref = json.loads((run / "state/preprocessing.json").read_text())["calibration"]["artifact_id"]
        store.validate(calibration_ref)
        stats = json.loads((store.path_for(calibration_ref) / "stats.json").read_text())
        importance_ref = next(
            x["output_importance"]
            for x in stats["layers"]
            if x["layer"] == {"block": {"index": 0}, "path": "mlp.down_proj"}
        )
        with LocalTensorStore(store).read(from_dict(TensorRef, importance_ref), "cpu") as value:
            importance = value.clone()

        def forward(module, values):
            return adapter.run_block(module, values, **metadata)

        (out / "protocol.json").write_text(
            json.dumps(
                {
                    "run_output": str(run.resolve()),
                    "snapshot": str(snapshot.resolve()),
                    "block_artifact": record["artifact_id"],
                    "calibration_artifact": calibration_ref,
                    "input_shape": list(inputs.shape),
                    "tuning_config": tuning,
                    "seed": args.seed,
                    "initialization": args.initialization,
                    "initialization_artifacts": initialization_artifacts,
                    "remaining_projections": "source BF16",
                    "optimizer": args.optimizer,
                    "torch": str(torch.__version__),
                    "gpu": torch.cuda.get_device_name(),
                },
                indent=2,
            )
        )
        if tuning["compile_forward"] and args.mode != "eager":
            forward = torch.compile(forward, options={"emulate_precision_casts": True, "triton.cudagraphs": False})
        results = []
        for kind in (args.stage,) if args.stage != "both" else ("factorized", "nonfactorized"):
            working = copy.deepcopy(block).cuda()
            if args.optimizer == "torch-fused-fp32":
                working.mlp.gate_proj.float()
            elif args.optimizer == "torch-fused-scales-fp32":
                for module in working.mlp.gate_proj.modules():
                    for name, parameter in tuple(module.named_parameters(recurse=False)):
                        if name.startswith("scale_"):
                            setattr(module, name, torch.nn.Parameter(parameter.detach().float()))
            recorder = (
                Profiler(
                    ProfilingConfig(
                        level=ProfilingLevel.MICRO,
                        cuda_timing=True,
                        cuda_sample_every=1,
                        raw_samples_per_phase=1024,
                        emit_span_events=False,
                    ),
                    run_id=kind,
                )
                if args.mode == "micro"
                else NULL_RECORDER
            )
            cold_seconds = None
            stage_config = tuning["factorized" if kind == "factorized" else "non_factorized"]
            epochs = (
                stage_config["loop"]["epochs"] if kind == "factorized"
                else stage_config["epochs_by_layer_position"][0]
            )
            learning_rate = (
                stage_config["learning_rates"]["scale"] if kind == "factorized"
                else stage_config["optimizer"]["learning_rate"]
            )
            epoch_losses = []
            request = TuningRequest(
                inputs,
                targets,
                epochs,
                stage_config["loop"]["batch_size"],
                learning_rate,
                seed=args.seed,
                epoch_observer=lambda epoch, loss, values=epoch_losses: values.append((epoch, loss)),
                output_importance=importance,
                microbatch_size=tuning["microbatch_size"],
                restore_best_state=False,
                epoch_loss_mode="legacy_training",
            )

            def execute(kind=kind, working=working, request=request, recorder=recorder):
                if kind == "factorized":
                    return tune_factorized(
                        working,
                        "mlp.gate_proj",
                        request,
                        forward,
                        recorder,
                        learning_rates=FactorizedTuningLearningRates(
                            tuning["factorized"]["learning_rates"]["binary"],
                            tuning["factorized"]["learning_rates"]["scale"],
                            tuning["factorized"]["learning_rates"]["outlier"] or request.learning_rate,
                            tuning["factorized"]["learning_rates"]["bias"],
                        ),
                    )
                return tune_non_factorized(working, request, forward, recorder)

            initial_evaluated_loss = tuning_module._evaluate_loss(working, request, forward)
            if args.mode == "compiled":
                cold_start = time.perf_counter()
                execute()
                torch.cuda.synchronize()
                cold_seconds = time.perf_counter() - cold_start
                print("cold", kind, cold_seconds, flush=True)
                working.load_state_dict(block.state_dict())
            epoch_losses.clear()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if args.mode == "profile":
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as profiler:
                    metrics = execute()
                profiler.export_chrome_trace(str(out / f"{kind}-trace.json"))
                (out / f"{kind}-operators.txt").write_text(
                    profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=45)
                )
            else:
                metrics = execute()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if isinstance(recorder, Profiler):
                (out / f"{kind}-phases.json").write_text(json.dumps(recorder.snapshot(), indent=2))
            print(kind, seconds, metrics, flush=True)
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in working.named_parameters()},
                out / f"{args.mode}-{kind}.safetensors",
            )
            evaluated_loss = tuning_module._evaluate_loss(working, request, forward)
            unweighted_loss = tuning_module._evaluate_loss(
                working, replace(request, output_importance=None), forward,
            )
            with torch.no_grad():
                weight = copy.deepcopy(working.mlp.gate_proj).to(dtype=torch.bfloat16).dense_weight().float()
                reference_weight = source_gate_weight.to(device=weight.device, dtype=torch.float32)
                relative_error = float(torch.linalg.vector_norm(weight - reference_weight)
                                       / torch.linalg.vector_norm(reference_weight))
                del weight, reference_weight
            results.append(dict(
                kind=kind, optimizer=args.optimizer, seconds=seconds, cold_seconds=cold_seconds,
                metrics=asdict(metrics), epoch_losses=epoch_losses, initial_evaluated_loss=initial_evaluated_loss,
                evaluated_loss=evaluated_loss, unweighted_evaluated_loss=unweighted_loss,
                gate_relative_frobenius_error=relative_error,
            ))
            print("final evaluation", results[-1], flush=True)
            (out / f"{args.mode}.json").write_text(json.dumps(results, indent=2))
            del working
            del execute
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
