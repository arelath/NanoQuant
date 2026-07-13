import os
import subprocess
import sys
from pathlib import Path

import pytest

from nanoquant.infrastructure.device_lease import DeviceLeaseError, acquire_device_lease, canonical_device_name


def test_device_lease_rejects_concurrent_owner_and_releases() -> None:
    with acquire_device_lease("fixture:0"):
        with pytest.raises(DeviceLeaseError, match="already leased"):
            acquire_device_lease("fixture:0")
    with acquire_device_lease("fixture:0") as lease:
        assert lease.device == "fixture:0"


def test_default_cuda_alias_uses_cuda_zero_lease() -> None:
    assert canonical_device_name("cuda") == "cuda:0"
    assert canonical_device_name("CUDA:1") == "cuda:1"


def test_device_lease_rejects_owner_in_another_process_with_different_temp_root(tmp_path: Path) -> None:
    code = (
        "import time; "
        "from nanoquant.infrastructure.device_lease import acquire_device_lease; "
        "lease = acquire_device_lease('fixture:cross-process'); "
        "print('ready', flush=True); "
        "time.sleep(30)"
    )
    child_temp = tmp_path / "child-temp"
    child_temp.mkdir()
    child_environment = os.environ.copy()
    child_environment.update({"TEMP": str(child_temp), "TMP": str(child_temp), "TMPDIR": str(child_temp)})
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
