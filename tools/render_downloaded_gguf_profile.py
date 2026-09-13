"""Compare downloaded Gemma IQ2_M with retained best-methods and GGUF results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
import numpy as np

from nanoquant.infrastructure.io_utils import atomic_write_json, hash_file

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Results/060/060-best-methods-2bpw-compress-and-benchmark-gemma-3-1b-it-quality.json"
GGUF_REFERENCE = ROOT / "Results/048/048-adaptive-capability-correction-d2-gemma-3-1b-it-gguf-quality.json"


def paired_task_comparison(left, right):
    """Pair primary-metric correctness by sample ID; left is the external GGUF."""
    if left["primary_metric"] != right["primary_metric"]:
        raise ValueError("task primary metrics differ")
    field = "normalized_correct" if left["primary_metric"] == "acc_norm" else "raw_correct"
    differences = []
    for a, b in zip(left["examples"], right["examples"], strict=True):
        if a["sample_id"] != b["sample_id"] or a["correct_choice"] != b["correct_choice"]:
            raise ValueError("task examples are not aligned")
        differences.append(int(a[field])-int(b[field]))
    values = np.asarray(differences)
    draws = np.random.default_rng(6900).integers(0, len(values), size=(10000, len(values)))
    low, high = np.quantile(values[draws].mean(axis=1), [0.025, 0.975])
    return {"samples": len(values), "external_only_correct": int((values > 0).sum()),
            "nanoquant_only_correct": int((values < 0).sum()), "accuracy_delta": float(values.mean()),
            "paired_bootstrap_95pct_interval": [float(low), float(high)], "bootstrap_resamples": 10000}


def verify_deployment_protocol(external, retained):
    """Ignore cache fingerprint only when complete serialized scorer inputs match."""
    for key in ("input_sha256", "runner_sha256", "runtime_sha256", "scoring", "llama_cpp_commit",
                "gpu_layers", "parallel", "threads", "batch_threads"):
        if external["identity"][key] != retained["identity"][key]:
            raise ValueError(f"deployment comparison identity differs: {key}")
    protocols = [{k: v for k, v in x["protocol"].items() if k != "wikitext_fingerprint"}
                 for x in (external, retained)]
    if protocols[0] != protocols[1]:
        raise ValueError("deployment comparison semantic protocol differs")


def render(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frozen = json.loads((output / "frozen-protocol.json").read_text())
    inventory = json.loads((output / "gguf-inventory.json").read_text())
    external_path = output / "external_iq2_m-quality.json"
    external = json.loads(external_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    deployment = json.loads(GGUF_REFERENCE.read_text())
    stop = json.loads((output / "baseline-rescore-stop.json").read_text())
    if not all(x["passed"] for x in (external, reference, deployment)):
        raise ValueError("all included evaluations must be complete and successful")
    if hash_file(REFERENCE) != frozen["reference_quality_sha256"]:
        raise ValueError("retained best-methods reference changed")
    if external["protocol"] != reference["protocol"] or external["protocol"] != frozen["protocol"]:
        raise ValueError("best-methods reference protocol differs")
    if external["results"]["base"] != reference["results"]["base"]:
        raise ValueError("retained BF16 reference differs")
    if external["identity"]["input_sha256"] != frozen["prepared_input_sha256"]:
        raise ValueError("external scorer inputs differ from frozen inputs")
    verify_deployment_protocol(external, deployment)
    for item in (external["gguf"], deployment["gguf"]):
        if hash_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("GGUF no longer matches its completed evaluation")
    states = {"bf16_retained": reference["results"]["base"],
              "nanoquant060_factorized_retained": reference["results"]["frozen"],
              "external_iq2_m_gguf": external["results"]["gguf"],
              "nanoquant048_gguf_retained": deployment["results"]["gguf"]}
    main_labels = list(states)[:3]
    tasks = {label: {x["result"]["task_name"]: x["result"] for x in value["tasks"]}
             for label, value in states.items()}
    names = frozen["protocol"]["task_names"]
    pairs = {label: {name: paired_task_comparison(tasks["external_iq2_m_gguf"][name], tasks[label][name])
                    for name in names} for label in ("nanoquant060_factorized_retained", "nanoquant048_gguf_retained")}
    metrics = {label: {"perplexity": value["wikitext"]["perplexity"],
                       "task_macro_accuracy": float(np.mean([tasks[label][x]["primary_value"] for x in names])),
                       "tasks": {name: tasks[label][name]["primary_value"] for name in names}}
               for label, value in states.items()}
    count = inventory["external_iq2_m"]["original_model_parameters"]
    bp, nppl, ep = [metrics[label]["perplexity"] for label in main_labels]
    ba, na, ea = [100*metrics[label]["task_macro_accuracy"] for label in main_labels]
    size_ratio = inventory["external_iq2_m"]["file_bytes"] / inventory["nanoquant060"]["file_bytes"]
    result = {
        "status": "completed_with_retained_references", "download": frozen["download"], "metrics": metrics,
        "paired_task_comparisons": pairs, "inventory": inventory, "protocol": frozen["protocol"],
        "scorer_identity": external["identity"], "baseline_rescore_stop": stop,
        "primary_comparison_backend_limit": "IQ2_M llama.cpp versus retained 060 PyTorch factorized",
        "same_runtime_comparison": "IQ2_M versus retained 048 GGUF; identical runtime and input bytes",
        "perplexity_ratio_external_over_nanoquant060": ep/nppl,
        "macro_accuracy_delta_external_minus_nanoquant060": (ea-na)/100,
        "file_size_ratio_external_over_nanoquant060": size_ratio,
        "evaluation_artifact_hashes": {str(p): hash_file(p) for p in
                                      (external_path, REFERENCE, GGUF_REFERENCE, output / "frozen-protocol.json",
                                       output / "gguf-inventory.json", output / "baseline-rescore-stop.json")},
    }
    atomic_write_json(output / "summary.json", result)
    rows = []
    for name in names:
        a, b, c = [tasks[label][name]["primary_value"] for label in main_labels]
        rows.append(f"| {name} | {tasks[main_labels[1]][name]['primary_metric']} | "
                    f"{100*a:.1f}% | {100*b:.1f}% | {100*c:.1f}% | {100*(c-b):+.1f} pp |")
    storage = [f"| {label} | {inventory[label]['file_bytes']:,} | {inventory[label]['whole_file_bpw']:.4f} |"
               for label in ("nanoquant060", "external_iq2_m")]
    quant_rows = [f"| {kind} | {item['parameters']:,} | {100*item['parameters']/count:.2f}% | "
                  f"{8*item['bytes']/item['parameters']:.4f} |"
                  for kind, item in inventory["external_iq2_m"]["tensor_types"].items()]
    paired_rows = []
    for name, pair in pairs["nanoquant060_factorized_retained"].items():
        low, high = pair["paired_bootstrap_95pct_interval"]
        paired_rows.append(f"| {name} | {pair['external_only_correct']} | {pair['nanoquant_only_correct']} | "
                           f"{100*pair['accuracy_delta']:+.1f} pp | [{100*low:+.1f}, {100*high:+.1f}] pp |")
    gguf_rows = []
    for label in ("nanoquant048_gguf_retained", "external_iq2_m_gguf"):
        item = metrics[label]
        gguf_rows.append(f"| {label} | {item['perplexity']:.6f} | {100*item['task_macro_accuracy']:.2f}% | "
                         + " | ".join(f"{100*item['tasks'][name]:.1f}%" for name in names) + " |")
    value = states["external_iq2_m_gguf"]
    resources = ["unavailable" if value[key] is None else f"{value[key]/2**20:.1f} MiB"
                 for key in ("peak_device_bytes", "peak_device_shared_bytes", "peak_host_bytes")]
    (output / "report.md").write_text(
        "# Published Gemma 3 1B IQ2_M versus NanoQuant\n\n"
        f"The downloaded [Bartowski IQ2_M GGUF]({frozen['download']['source_url']}) was evaluated as published. "
        f"Against Experiment 060's retained best-methods results, it improves six-task average accuracy by "
        f"{ea-na:.2f} percentage points and lowers perplexity by {100*(1-ep/nppl):.2f}%, while its complete "
        f"GGUF is {100*(size_ratio-1):.2f}% larger. No compression, fitting, or tuning was run.\n\n"
        "**Backend qualification:** IQ2_M was freshly scored in llama.cpp; the primary baseline is "
        "060's retained PyTorch factorized quality reference. Inputs and evaluation protocol match, but "
        "runtime implementations differ. A separate completed GGUF comparison below uses 048 with the "
        "same llama.cpp binaries and byte-identical token inputs. BF16 is retained from 060.\n\n"
        "## Quality against the best-methods reference\n\n"
        "| Benchmark | Metric | BF16 retained | NQ 060 PyTorch retained | Published IQ2_M GGUF | IQ2_M minus NQ 060 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |\n"
        f"| WikiText-2 | Perplexity, lower is better | {bp:.6f} | {nppl:.6f} | {ep:.6f} | {ep-nppl:+.6f} |\n"
        + "\n".join(rows) + f"\n| Six-task mean | Accuracy | {ba:.2f}% | {na:.2f}% | {ea:.2f}% | {ea-na:+.2f} pp |\n\n"
        "The mean weights the six primary metrics equally. PIQA, ARC, and HellaSwag use normalized "
        "accuracy; Winogrande and BoolQ use raw accuracy. The largest gains are on the two ARC tasks.\n\n"
        "## What the 2-bit label contains\n\n"
        "IQ2_M is a mixed tensor-format recipe. Its name does not mean every weight or the whole model "
        "uses two bits. Many weights in this file use higher-bit formats, measured directly below.\n\n"
        "| GGUF tensor type | Original parameters | Share of model | Tensor payload bits/weight |\n"
        "| --- | ---: | ---: | ---: |\n" + "\n".join(quant_rows)
        + "\n\n| Complete deployment file | Bytes | Whole-file bits/original weight |\n"
        "| --- | ---: | ---: |\n" + "\n".join(storage)
        + f"\n\nThe denominator is {count:,} original parameters, checked against pinned source checkpoint "
        "headers. Whole-file BPW includes embeddings, sidecars, norms, metadata, and alignment. NanoQuant's "
        "historical 1.9634 charged factor BPW has a different scope. Both files have byte-identical Q8_0 "
        "embeddings. This does not establish a quality advantage at equal storage.\n\n"
        "## Completed comparison using the same GGUF runtime\n\n"
        "048 is the strongest completed retained NanoQuant GGUF benchmark found, but is weaker than the "
        "060 best-methods reference. It is included as a separate runtime-matched check.\n\n"
        "| GGUF | Perplexity | Task mean | PIQA | ARC Easy | ARC Challenge | HellaSwag | Winogrande | BoolQ |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n" + "\n".join(gguf_rows)
        + f"\n\nThe 048 file has {deployment['gguf']['bytes']:,} bytes "
        f"({8*deployment['gguf']['bytes']/count:.4f} whole-file BPW). Runtime, scorer, input bytes, scoring "
        "algorithm, and execution settings match IQ2_M. Its cache fingerprint differs; the serialized "
        "scorer input and all semantic protocol fields match exactly.\n\n"
        "## Protocol and provenance\n\n"
        "The reference is google/gemma-3-1b-it at dcc83ea841ab6100d6b47a070329e1ba4cf78752. The publisher "
        "names the same model without publishing its upstream weight revision. The revision in the raw "
        "evaluator identifies the tokenizer/BF16 reference, not proof of the publisher's weight revision. "
        "Architecture, vocabulary/type mapping, special IDs, parameter count, and embedding payload agree. "
        "The external GGUF's omitted SWA rope key defaults to 10000 in this runtime, matching NanoQuant.\n\n"
        "- WikiText-2: 64 independent 128-token windows, 8,128 targets; dataset revision "
        "b08601e04326c79dfdd32d625aee71d232d685c3. Perplexity is specific to these short windows.\n"
        "- Six tasks: first 1,000 examples each; matching prompts, token IDs, truncation, and normalization.\n"
        f"- Combined input SHA-256: `{frozen['prepared_input_sha256']}`.\n"
        f"- llama.cpp commit: `{external['identity']['llama_cpp_commit']}`; all GPU layers, parallel 4.\n"
        "- F32 logits with F64 host log-sum-exp; finite and complete target-token scores required.\n"
        "- Retained diagnostic protocol, not a fresh held-out model-selection gate.\n\n"
        "## Paired differences against retained 060\n\n"
        "| Task | Only IQ2_M correct | Only NQ 060 correct | Accuracy difference | Paired bootstrap 95% interval |\n"
        "| --- | ---: | ---: | ---: | --- |\n" + "\n".join(paired_rows)
        + "\n\nIntervals resample the same 1,000 paired examples 10,000 times with seed 6900. They describe "
        "uncertainty within this slice, excluding earlier model selection, runtime differences, and "
        "generalization to other tasks. No multiple-comparison adjustment is applied.\n\n"
        "## Execution and stopped extra baseline rescore\n\n"
        f"IQ2_M completed in {value['elapsed_seconds']:.2f} seconds. Peaks: {resources[0]} dedicated GPU, "
        f"{resources[1]} shared GPU, {resources[2]} host working set. These independently measured child "
        "resources describe this scoring workload, not standalone decode throughput.\n\n"
        f"A fresh 060 GGUF rescore was deliberately stopped after {stop['completed_batches']}/"
        f"{stop['total_batches']} batches in {stop['elapsed_seconds']/60:.1f} minutes because the pace "
        "implied several hours. Larger I8 salient-column sidecars take a slower custom kernel path. "
        "No incomplete scores enter this report. The existing 060 reference and completed 048 runtime "
        "check are used explicitly instead. Original frozen plan, stop receipt, events, and failure exit "
        "are retained; all scorer descendants were confirmed stopped.\n\n"
        "An earlier preflight stopped before GPU use on the 048/060 cache-fingerprint mismatch. The guard "
        "was corrected to require the current fingerprint to match 060, all semantic fields to match "
        "048, and complete input bytes to match 048. Both attempt logs are retained.\n\n"
        "![Quality comparison](quality-comparison.png)\n\n"
        "Raw external per-example scores, retained reference paths/hashes, inventory, frozen protocol, "
        "paired comparisons, and summary accompany this report.\n", encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [3, 1]}, layout="constrained")
    labels = ("BF16 retained", "NQ 060 PyTorch retained", "Published IQ2_M GGUF")
    x = np.arange(len(names))
    for index, (label, display) in enumerate(zip(main_labels, labels, strict=True)):
        axes[0].bar(x+(index-1)*0.25, [100*tasks[label][name]["primary_value"] for name in names],
                    width=0.25, label=display)
    axes[0].set(xticks=x, xticklabels=["PIQA", "ARC Easy", "ARC Hard", "HellaSwag", "Winogrande", "BoolQ"],
                ylabel="Primary accuracy (%)", ylim=(0, 100), title="Same 1,000 examples per task")
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend(fontsize=8)
    axes[1].bar(np.arange(3), [bp, nppl, ep], color=["C0", "C1", "C2"])
    axes[1].set(xticks=np.arange(3), xticklabels=["BF16", "NQ 060", "IQ2_M"], ylabel="Perplexity (lower is better)",
                title="WikiText-2\n64 × 128-token windows")
    fig.suptitle("Gemma 3 1B: published IQ2_M versus retained best-methods reference\n"
                 "Same evaluation inputs; different execution backends", fontsize=12)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"quality-comparison.{suffix}", dpi=160)
    plt.close(fig)
    print(json.dumps({"metrics": metrics, "perplexity_ratio": ep/nppl, "file_size_ratio": size_ratio}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "Results/069")
    render(parser.parse_args().output.resolve())
