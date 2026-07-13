import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from nanoquant.infrastructure.device_lease import (
    DeviceLeaseError,
    _lease_root,
    acquire_device_lease,
    canonical_device_name,
)


def test_device_lease_rejects_concurrent_owner_and_releases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NANOQUANT_DEVICE_LEASE_ROOT", str(tmp_path / "leases"))
    with acquire_device_lease("fixture:0"):
        with pytest.raises(DeviceLeaseError, match="already leased"):
            acquire_device_lease("fixture:0")
    with acquire_device_lease("fixture:0") as lease:
        assert lease.device == "fixture:0"


def test_default_cuda_alias_uses_cuda_zero_lease() -> None:
    assert canonical_device_name("cuda") == "cuda:0"
    assert canonical_device_name("CUDA:1") == "cuda:1"


def test_explicit_fixture_root_cannot_redirect_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = tmp_path / "isolated-leases"
    monkeypatch.setenv("NANOQUANT_DEVICE_LEASE_ROOT", str(explicit))

    assert _lease_root("fixture:0") == explicit
    assert _lease_root("cuda:0") != explicit


def test_device_lease_rejects_owner_with_different_environment_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease_root = tmp_path / "shared-leases"
    monkeypatch.setenv("NANOQUANT_DEVICE_LEASE_ROOT", str(lease_root))
    code = (
        "import time; "
        "from nanoquant.infrastructure.device_lease import acquire_device_lease; "
        "lease = acquire_device_lease('fixture:cross-process'); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    child_local_app_data = tmp_path / "child-local-app-data"
    child_local_app_data.mkdir()
    child_environment = os.environ.copy()
    child_environment.update(
        {
            "TEMP": str(child_temp),
            "TMP": str(child_temp),
            "TMPDIR": str(child_temp),
            "LOCALAPPDATA": str(child_local_app_data),
            "NANOQUANT_DEVICE_LEASE_ROOT": str(lease_root),
        }
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(DeviceLeaseError, match="already leased"):
            acquire_device_lease("fixture:cross-process")
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(os.name != "nt", reason="Windows named-mutex lease")
def test_windows_cuda_lease_rejects_cross_process_owner() -> None:
    device = f"cuda:test-{uuid.uuid4().hex}"
    code = (
        "import time; "
        "from nanoquant.infrastructure.device_lease import acquire_device_lease; "
        f"lease = acquire_device_lease({device!r}); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        with pytest.raises(DeviceLeaseError, match="already leased"):
            acquire_device_lease(device)
    finally:
        child.terminate()
        child.wait(timeout=10)
    with acquire_device_lease(device) as lease:
        assert lease.device == device
