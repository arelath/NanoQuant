"""Run discovery and canonical event-log CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import cast

from nanoquant.config.codec import to_dict
from nanoquant.domain.runs import RunStatus
from nanoquant.infrastructure.events import (
    EventStreamError,
    event_stream_integrity,
    read_event_prefix,
    render_event_line,
    render_event_log,
)
from nanoquant.infrastructure.run_registry import (
    DiscoveredRun,
    discover_runs,
    inspect_run_path,
    rebuild_registry,
    select_run,
)
from nanoquant.ports.event_sink import Event, Severity


class CommandError(RuntimeError):
    def __init__(self, message: str, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__(message)


def _add_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, default=Path("runs"))


def _add_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("selector", nargs="?")
    parser.add_argument("--path", dest="direct_path", type=Path)
    parser.add_argument("--status", choices=tuple(status.value for status in RunStatus))


def add_run_commands(subcommands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    runs = subcommands.add_parser("runs", help="list, inspect, and resolve run directories")
    run_commands = runs.add_subparsers(dest="runs_command", required=True)

    list_runs = run_commands.add_parser("list", help="list discoverable runs")
    _add_root(list_runs)
    list_runs.add_argument("--status", choices=tuple(status.value for status in RunStatus))
    list_runs.add_argument("--experiment", type=int)
    list_runs.add_argument("--limit", type=int, default=10)
    list_runs.add_argument("--json", action="store_true")

    show = run_commands.add_parser("show", help="show normalized run metadata")
    _add_root(show)
    _add_selector(show)
    show.add_argument("--json", action="store_true")

    path = run_commands.add_parser("path", help="print one absolute run evidence path")
    _add_root(path)
    _add_selector(path)
    path.add_argument("--kind", choices=("run", "manifest", "events", "journal", "report"), required=True)

    rebuild = run_commands.add_parser("rebuild-registry", help="rebuild immutable external-run pointers")
    _add_root(rebuild)
    rebuild.add_argument("--include-root", type=Path, action="append", default=[])

    vram = run_commands.add_parser("vram", help="summarize VRAM samples and checkpoint peaks")
    _add_root(vram)
    _add_selector(vram)
    vram.add_argument("--json", action="store_true")

    logs = subcommands.add_parser("logs", help="render or follow canonical run events")
    _add_root(logs)
    _add_selector(logs)
    logs.add_argument("--follow", action="store_true")
    logs.add_argument("--level", choices=tuple(level.value for level in Severity), default=Severity.DEBUG.value)
    logs.add_argument("--json", action="store_true")
    logs.add_argument("--save", action="store_true")
    logs.add_argument("--poll-seconds", type=float, default=1.0)


def _resolve(args: argparse.Namespace) -> DiscoveredRun:
    direct = getattr(args, "direct_path", None)
    if direct is not None:
        target = inspect_run_path(direct)
        if not target.path.is_dir():
            raise CommandError(f"run path does not exist: {target.path}", 3)
        return target
    selector = getattr(args, "selector", None)
    if not selector:
        raise CommandError("a run selector or --path is required", 2)
    status_value = getattr(args, "status", None)
    status = None if status_value is None else RunStatus(status_value)
    try:
        return select_run(args.run_root, selector, status=status)
    except FileNotFoundError as exc:
        raise CommandError(str(exc), 3) from exc
    except ValueError as exc:
        raise CommandError(str(exc), 2) from exc


def _view(item: DiscoveredRun) -> dict[str, object]:
    return {
        "run_id": item.run_id,
        "status": item.status,
        "created_at": item.created_at,
        "experiment_number": item.experiment_number,
        "component": item.component,
        "path": str(item.path),
        "source": item.source,
        "integrity": item.integrity,
    }


def _list_runs(args: argparse.Namespace) -> int:
    if args.limit <= 0:
        raise CommandError("--limit must be positive", 2)
    items = list(discover_runs(args.run_root))
    if args.status is not None:
        items = [item for item in items if item.status == args.status]
    if args.experiment is not None:
        items = [item for item in items if item.experiment_number == args.experiment]
    items.sort(key=lambda item: (item.created_at, item.run_id), reverse=True)
    items = items[: args.limit]
    views = [_view(item) for item in items]
    if args.json:
        print(json.dumps(views, sort_keys=True, indent=2))
        return 0
    print(f"{'RUN ID':36} {'STATUS':12} {'CREATED':27} {'EXP':5} {'COMPONENT':30} PATH")
    for view in views:
        experiment = "-" if view["experiment_number"] is None else str(view["experiment_number"])
        status = str(view["status"])
        if view["integrity"] != "ok":
            status = str(view["integrity"])
        print(
            f"{str(view['run_id'])[:36]:36} {status[:12]:12} {str(view['created_at'])[:27]:27} "
            f"{experiment[:5]:5} {str(view['component'])[:30]:30} {view['path']}"
        )
    return 0


def _show(args: argparse.Namespace) -> int:
    item = _resolve(args)
    payload = _view(item)
    payload["events_integrity"] = event_stream_integrity(item.path / "events.jsonl")
    payload["manifest"] = None if item.manifest is None else to_dict(item.manifest)
    if args.json:
        print(json.dumps(payload, sort_keys=True, indent=2))
    else:
        for key, value in payload.items():
            rendered = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
            print(f"{key}: {rendered}")
    return 0 if item.integrity in {"ok", "unmanaged"} else 4


def _path(args: argparse.Namespace) -> int:
    item = _resolve(args)
    paths = {
        "run": item.path,
        "manifest": item.path / "manifest.json",
        "events": item.path / "events.jsonl",
        "journal": item.path / "state" / "journal.jsonl",
        "report": (
            item.path / "reports" / "summary.md"
            if (item.path / "reports" / "summary.md").exists()
            else item.path / "report.md"
        ),
    }
    print(paths[args.kind].resolve())
    return 0


def _integer_field(event: Event, name: str) -> int | None:
    value = event.fields.get(name)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize_vram_events(events: Iterable[Event]) -> dict[str, object]:
    """Fold the canonical event stream into a read-only VRAM comparison view."""

    baseline: dict[str, int] | None = None
    latest: dict[str, int] = {}
    block_peaks: list[dict[str, int | float | None]] = []
    peak_allocated = 0
    peak_reserved = 0
    device_total = 0
    sample_count = 0
    sawtooth_count = 0
    previous_sample_reserved: int | None = None
    meter_names = (
        "cuda.allocated_bytes",
        "cuda.reserved_bytes",
        "cuda.device_free_bytes",
        "cuda.device_used_bytes",
        "cuda.device_total_bytes",
    )
    for event in events:
        current = {name: value for name in meter_names if (value := _integer_field(event, name)) is not None}
        if current:
            latest.update(current)
        if baseline is None and event.name in {"run.started", "run.resumed"} and current:
            baseline = current
        allocated_candidates = (
            _integer_field(event, "cuda.allocated_bytes"),
            _integer_field(event, "cuda.peak_allocated_bytes"),
            _integer_field(event, "cuda.window_peak_allocated_bytes"),
        )
        reserved_candidates = (
            _integer_field(event, "cuda.reserved_bytes"),
            _integer_field(event, "cuda.peak_reserved_bytes"),
            _integer_field(event, "cuda.window_peak_reserved_bytes"),
            _integer_field(event, "gpu_peak_bytes"),
        )
        peak_allocated = max(peak_allocated, *(value for value in allocated_candidates if value is not None))
        peak_reserved = max(peak_reserved, *(value for value in reserved_candidates if value is not None))
        device_total = max(device_total, _integer_field(event, "cuda.device_total_bytes") or 0)
        if event.name == "resource.sample":
            sample_count += 1
            reserved = _integer_field(event, "cuda.reserved_bytes")
            if reserved is not None and previous_sample_reserved is not None and reserved < previous_sample_reserved:
                sawtooth_count += 1
            if reserved is not None:
                previous_sample_reserved = reserved
        if event.name == "block.completed":
            window_allocated = _integer_field(event, "cuda.window_peak_allocated_bytes")
            window_reserved = _integer_field(event, "cuda.window_peak_reserved_bytes")
            planned = _integer_field(event, "planned_device_bytes")
            measured = max(window_allocated or 0, window_reserved or 0)
            block_peaks.append(
                {
                    "block": _integer_field(event, "block"),
                    "window_peak_allocated_bytes": window_allocated,
                    "window_peak_reserved_bytes": window_reserved,
                    "planned_device_bytes": planned,
                    "budget_utilization": measured / planned if planned and measured else None,
                }
            )
    return {
        "sample_count": sample_count,
        "baseline": baseline,
        "latest": latest or None,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "device_total_bytes": device_total,
        "empty_cache_sawtooth_count": sawtooth_count,
        "block_peaks": block_peaks,
    }


def _vram(args: argparse.Namespace) -> int:
    item = _resolve(args)
    events_path = item.path / "events.jsonl"
    if not events_path.exists():
        raise CommandError(f"event stream does not exist: {events_path}", 3)
    summary = summarize_vram_events(read_event_prefix(events_path))
    if args.json:
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    print(f"run: {item.run_id}")
    print(f"samples: {summary['sample_count']}")
    print(f"peak allocated bytes: {summary['peak_allocated_bytes']}")
    print(f"peak reserved bytes: {summary['peak_reserved_bytes']}")
    print(f"device total bytes: {summary['device_total_bytes']}")
    print(f"empty-cache sawtooths: {summary['empty_cache_sawtooth_count']}")
    blocks = cast(list[dict[str, object]], summary["block_peaks"])
    if blocks:
        print("block  allocated peak  reserved peak  planned  utilization")
        for block in blocks:
            utilization = block["budget_utilization"]
            rendered_utilization = (
                f"{utilization:.3f}" if isinstance(utilization, (int, float)) else "-"
            )
            print(
                f"{str(block['block']):5} {str(block['window_peak_allocated_bytes']):14} "
                f"{str(block['window_peak_reserved_bytes']):13} {str(block['planned_device_bytes']):8} "
                f"{rendered_utilization}"
            )
    return 0


def _print_event(event: Event, *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(asdict(event), sort_keys=True, separators=(",", ":")), flush=True)
    else:
        print(render_event_line(event), flush=True)


def _manifest_status(path: Path) -> str | None:
    manifest = path / "manifest.json"
    if not manifest.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        status = payload.get("status")
    except (OSError, json.JSONDecodeError):
        return None
    return status if isinstance(status, str) else None


def _logs(args: argparse.Namespace) -> int:
    item = _resolve(args)
    events_path = item.path / "events.jsonl"
    if not events_path.exists():
        raise CommandError(f"event stream does not exist: {events_path}", 3)
    level = Severity.parse(args.level)
    if args.poll_seconds <= 0:
        raise CommandError("--poll-seconds must be positive", 2)
    if args.save:
        render_event_log(events_path, item.path / "run.log")

    last_sequence = 0
    previous_identity: tuple[int, int] | None = None
    previous_size: int | None = None
    try:
        while True:
            stat = events_path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if (previous_identity is not None and identity != previous_identity) or (
                previous_size is not None and stat.st_size < previous_size
            ):
                last_sequence = 0
            previous_identity = identity
            previous_size = stat.st_size
            for event in read_event_prefix(events_path):
                if event.sequence <= last_sequence:
                    continue
                last_sequence = event.sequence
                if Severity.parse(event.severity).rank >= level.rank:
                    _print_event(event, json_output=args.json)
            integrity = event_stream_integrity(events_path)
            if not args.follow:
                return 0 if integrity == "ok" else 4
            status = _manifest_status(item.path)
            if status in {RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.INTERRUPTED.value}:
                print(f"[run] status={status}", file=sys.stderr)
                return 0
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        return 0


def execute_run_command(args: argparse.Namespace) -> int:
    try:
        if args.command == "logs":
            return _logs(args)
        if args.runs_command == "list":
            return _list_runs(args)
        if args.runs_command == "show":
            return _show(args)
        if args.runs_command == "path":
            return _path(args)
        if args.runs_command == "vram":
            return _vram(args)
        if args.runs_command == "rebuild-registry":
            count = rebuild_registry(args.run_root, tuple(args.include_root))
            print(count)
            return 0
        raise CommandError("unknown run command", 2)
    except CommandError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except EventStreamError as exc:
        print(str(exc), file=sys.stderr)
        return 4
    except PermissionError as exc:
        print(str(exc), file=sys.stderr)
        return 5
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 5
