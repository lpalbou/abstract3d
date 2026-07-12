"""Opt-in wrapper for the texture-path parity canaries.

The canaries are full CPU rebakes (~15 min total) — far too heavy for the
unit suite, but mandatory before landing texture-path changes. Enable
with ABSTRACT3D_PARITY_CANARY=1:

    ABSTRACT3D_PARITY_CANARY=1 .venv/bin/python -m pytest \
        tests/test_parity_canary.py -q

See scripts/parity_canary.py for what each canary pins and why.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(
    os.environ.get("ABSTRACT3D_PARITY_CANARY") != "1",
    reason="parity canaries are opt-in (set ABSTRACT3D_PARITY_CANARY=1); "
           "~15 min of CPU rebakes",
)


@pytest.mark.parametrize("canary", ["p1", "p2", "face"])
def test_parity_canary(canary: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/parity_canary.py"),
         "--only", canary],
        capture_output=True, text=True, timeout=3600,
    )
    assert proc.returncode == 0, (
        f"parity canary {canary} failed:\n{proc.stdout}\n{proc.stderr}")
