from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def _source_env() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(src) if not existing else os.pathsep.join([str(src), existing])
    return env


def test_top_level_imports_do_not_eagerly_load_backend_runtime_modules() -> None:
    root = Path(__file__).resolve().parents[1]
    script = textwrap.dedent(
        """
        import abstract3d
        import abstract3d.backends
        import sys

        assert "abstract3d.backends.triposr_runtime" not in sys.modules
        assert "abstract3d.backends.step1x_runtime" not in sys.modules
        assert "abstract3d.backends.trellis2_runtime" not in sys.modules
        """
    )
    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        cwd=root,
        env=_source_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
