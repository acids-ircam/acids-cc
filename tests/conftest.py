import os
import sys
from pathlib import Path

import pytest
import torch

sys_path = str(Path(__file__).parent.parent)
# prepend, so the local sources win over any installed cached_conv
if sys_path not in sys.path: sys.path.insert(0, sys_path)


def available_devices():
    """Devices to test on. Override with e.g. CC_TEST_DEVICES=cpu,cuda"""
    if os.environ.get("CC_TEST_DEVICES"):
        return os.environ["CC_TEST_DEVICES"].split(",")
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=available_devices())
def device(request):
    """Models are built on CPU then moved with .to(device), as users do."""
    return torch.device(request.param)
