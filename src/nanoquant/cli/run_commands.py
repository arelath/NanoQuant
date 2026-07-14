"""Run discovery and canonical event-log CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

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
