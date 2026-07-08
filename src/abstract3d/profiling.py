"""Lightweight stage/memory profiling for generation and bake pipelines.

Design constraints:

- Zero impact on numerical output: profiling only reads clocks and process
  memory counters. It never touches tensors or arrays, so profiled runs stay
  bit-identical to unprofiled runs (verified by the golden-bake harness).
- Zero hard dependencies beyond `psutil` (already required by the runtime
  extras); `torch.mps` counters are sampled only when torch is importable
  and the accelerator is present.

Two cooperating pieces:

- `MemorySampler`: a daemon thread sampling process RSS (and MPS allocated
  bytes when available) on a fixed interval, keeping the full timeline so
  peaks BETWEEN stage boundaries are attributable.
- `StageProfiler`: records named stage spans and, for each span, the RSS
  peak the sampler observed inside it. `wrap_module_functions` instruments
  module-level functions externally (used by the harness so the library
  code path stays untouched during baseline measurements).
"""

from __future__ import annotations

import functools
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _mps_allocated_bytes() -> Optional[int]:
    try:
        import torch

        if torch.backends.mps.is_available():
            return int(torch.mps.current_allocated_memory())
    except Exception:
        return None
    return None


class MemorySampler:
    """Samples process RSS (and MPS allocation) on a background thread."""

    def __init__(self, interval_s: float = 0.05, sample_mps: bool = True) -> None:
        self.interval_s = float(interval_s)
        self.sample_mps = bool(sample_mps)
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        import psutil

        self._process = psutil.Process()

    def _sample_once(self) -> Dict[str, float]:
        record: Dict[str, float] = {
            "t": time.perf_counter(),
            "rss": float(self._process.memory_info().rss),
        }
        if self.sample_mps:
            mps = _mps_allocated_bytes()
            if mps is not None:
                record["mps"] = float(mps)
        return record

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._sample_once())
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def start(self) -> "MemorySampler":
        self.samples.append(self._sample_once())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.samples.append(self._sample_once())
        except Exception:
            pass

    def peak_between(self, t0: float, t1: float, key: str = "rss") -> Optional[float]:
        values = [s[key] for s in self.samples if t0 <= s["t"] <= t1 and key in s]
        return max(values) if values else None

    def peak(self, key: str = "rss") -> Optional[float]:
        values = [s[key] for s in self.samples if key in s]
        return max(values) if values else None


@dataclass
class StageRecord:
    name: str
    t_start: float
    t_end: float = 0.0
    rss_peak: Optional[float] = None
    mps_peak: Optional[float] = None
    calls: int = 1

    @property
    def seconds(self) -> float:
        return self.t_end - self.t_start


@dataclass
class StageProfiler:
    sampler: MemorySampler
    stages: List[StageRecord] = field(default_factory=list)
    _originals: List[Any] = field(default_factory=list)

    def record(self, name: str, t_start: float, t_end: float) -> StageRecord:
        record = StageRecord(
            name=name,
            t_start=t_start,
            t_end=t_end,
            rss_peak=self.sampler.peak_between(t_start, t_end, "rss"),
            mps_peak=self.sampler.peak_between(t_start, t_end, "mps"),
        )
        self.stages.append(record)
        return record

    def wrap_module_functions(self, module: Any, names: Sequence[str]) -> None:
        """Externally instrument module-level callables (harness-side only)."""

        for name in names:
            original = getattr(module, name, None)
            if original is None or not callable(original):
                continue

            def make_wrapper(fn: Any, fn_name: str) -> Any:
                @functools.wraps(fn)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    t0 = time.perf_counter()
                    try:
                        return fn(*args, **kwargs)
                    finally:
                        self.record(fn_name, t0, time.perf_counter())

                return wrapper

            self._originals.append((module, name, original))
            setattr(module, name, make_wrapper(original, name))

    def unwrap(self) -> None:
        for module, name, original in self._originals:
            setattr(module, name, original)
        self._originals.clear()

    def report(self) -> Dict[str, Any]:
        overall_peak = self.sampler.peak("rss")
        report: Dict[str, Any] = {
            "overall_rss_peak_bytes": overall_peak,
            "overall_mps_peak_bytes": self.sampler.peak("mps"),
            "stages": [
                {
                    "name": s.name,
                    "seconds": round(s.seconds, 3),
                    "t_start": s.t_start,
                    "t_end": s.t_end,
                    "rss_peak_bytes": s.rss_peak,
                    "mps_peak_bytes": s.mps_peak,
                }
                for s in self.stages
            ],
        }
        return report

    def save(self, path: Any) -> Dict[str, Any]:
        report = self.report()
        report["timeline"] = self.sampler.samples
        Path(path).write_text(json.dumps(report, indent=1))
        return report


def plot_timeline(
    report: Dict[str, Any],
    out_path: Any,
    *,
    title: str = "Memory timeline",
    min_stage_seconds: float = 0.5,
) -> None:
    """Render the RSS/MPS timeline with stage spans for the proof pack."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timeline = report.get("timeline") or []
    if not timeline:
        return
    t0 = timeline[0]["t"]
    ts = [s["t"] - t0 for s in timeline]
    rss = [s["rss"] / 1e9 for s in timeline]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, rss, lw=1.2, color="#4433aa", label="RSS (GB)")
    mps = [(s["t"] - t0, s["mps"] / 1e9) for s in timeline if "mps" in s]
    if mps:
        ax.plot([m[0] for m in mps], [m[1] for m in mps], lw=1.0,
                color="#22aa66", label="MPS allocated (GB)")
    # Alternate translucent spans for the slow stages so the plot stays legible.
    palette = ["#ffdd55", "#ff8855", "#55ccff", "#cc88ff", "#88dd88", "#ff88bb"]
    slow = [s for s in report.get("stages", []) if s["seconds"] >= min_stage_seconds]
    y_top = max(rss) if rss else 1.0
    for idx, stage in enumerate(slow):
        if "t_start" not in stage:
            continue
        x0, x1 = stage["t_start"] - t0, stage["t_end"] - t0
        color = palette[idx % len(palette)]
        ax.axvspan(x0, x1, alpha=0.18, color=color)
        ax.text((x0 + x1) / 2.0, y_top * 1.01, stage["name"], rotation=90,
                fontsize=6, ha="center", va="bottom", color="#333333")
    ax.set_xlabel("seconds")
    ax.set_ylabel("GB")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
