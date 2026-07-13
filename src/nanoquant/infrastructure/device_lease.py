"""Cross-process leases preventing accidental concurrent resident GPU runs."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


class DeviceLeaseError(RuntimeError):
    """Raised when another resident process already owns the requested device."""


class DeviceLease:
    def __init__(self, device: str, path: Path, token: str) -> None:
        self.device = device
        self.path = path
        self._token = token
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        owner = self.path / "owner.json"
        try:
            payload = json.loads(owner.read_text(encoding="utf-8"))
            if payload.get("token") != self._token:
                return
            owner.unlink()
            self.path.rmdir()
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            return

    def __enter__(self) -> DeviceLease:
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


def canonical_device_name(device: str) -> str:
    return "cuda:0" if device.lower() == "cuda" else device.lower()


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    import ctypes

    process = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if process:
        ctypes.windll.kernel32.CloseHandle(process)
        return True
    return ctypes.get_last_error() == 5


def _remove_stale_lease(path: Path) -> bool:
    owner = path / "owner.json"
    try:
        payload = json.loads(owner.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0))
    except (FileNotFoundError, OSError, TypeError, ValueError, json.JSONDecodeError):
        pid = 0
    if _process_exists(pid):
        return False
    stale = path.with_name(f"{path.name}.stale-{uuid.uuid4().hex}")
    try:
        path.rename(stale)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    try:
        stale_owner = stale / "owner.json"
        if stale_owner.exists():
            stale_owner.unlink()
        stale.rmdir()
    except OSError:
        pass
    return True


def _lease_root(device: str) -> Path:
    """Return a per-user root that is stable across process temp environments."""

    explicit_root = os.environ.get("NANOQUANT_DEVICE_LEASE_ROOT")
    if explicit_root and not device.startswith("cuda"):
        root = Path(explicit_root)
        if not root.is_absolute():
            raise ValueError("NANOQUANT_DEVICE_LEASE_ROOT must be an absolute path")
        return root
    if os.name == "nt":
        import ctypes

        # CSIDL_LOCAL_APPDATA resolves through the Windows shell/registry rather
        # than the caller's environment. Diagnostic launchers commonly override
        # LOCALAPPDATA for isolation; using that variable would silently split
        # the device-lock namespace and permit concurrent CUDA owners.
        buffer = ctypes.create_unicode_buffer(32768)
        result = ctypes.windll.shell32.SHGetFolderPathW(None, 0x001C, None, 0, buffer)
        if result == 0 and buffer.value:
            return Path(buffer.value) / "NanoQuant" / "leases"
        return Path.home() / ".nanoquant" / "leases"
    getuid = getattr(os, "getuid", None)
    user = str(getuid()) if callable(getuid) else str(Path.home()).replace(os.sep, "-")
    return Path("/tmp") / f"nanoquant-{user}" / "leases"


def acquire_device_lease(device: str) -> DeviceLease:
    canonical = canonical_device_name(device)
    safe_name = "".join(character if character.isalnum() else "-" for character in canonical)
    root = _lease_root(canonical)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / f"nanoquant-resident-{safe_name}.lease"
    for _attempt in range(2):
        try:
            path.mkdir()
        except FileExistsError as exc:
            if _remove_stale_lease(path):
                continue
            raise DeviceLeaseError(f"resident quantization device is already leased: {canonical}") from exc
        token = uuid.uuid4().hex
        try:
            (path / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "token": token}, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            path.rmdir()
            raise
        return DeviceLease(canonical, path, token)
    raise DeviceLeaseError(f"resident quantization device lease could not be acquired: {canonical}")
