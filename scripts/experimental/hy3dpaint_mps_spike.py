"""Spike: can the Hunyuan3D-2.1 paint stack LOAD on this Mac (MPS)?

Scope (deliberately bounded — the GPU is contended):
  1. clone the agenticvibes hy3dpaint fork at a pinned commit (MPS/MLX port of the
     official Tencent hy3dpaint; port code MIT, Tencent code under the Hunyuan3D-2.1
     community license the repo already gates),
  2. build the two compiled extensions IN PLACE (no venv installs):
     custom_rasterizer (CppExtension on macOS) + DifferentiableRenderer mesh
     inpaint processor,
  3. smoke-test the CPU rasterizer on one real triangle (shape + coverage checks),
  4. download the paint weights (hunyuan3d-paintpbr-v2-1, ~3.7 GB) and
     facebook/dinov2-giant (~4.5 GB) into the shared HF cache,
  5. load the multiview paint pipeline to the target device in fp16 and dry
     shape-check it (module tree, param counts, device/dtype, RSS).

It NEVER runs a denoise step or a bake. Exit code 0 = every critical stage passed.
Every stage records worked/refused with the exact error; the final table is the
spike report.

Run:
    <workspace>/.venv/bin/python \
        scripts/experimental/hy3dpaint_mps_spike.py [--device mps|cpu] [--skip-download]
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

FORK_URL = "https://github.com/agenticvibes/ComfyUI-Hunyuan3d-Paint.git"
FORK_COMMIT = "3d8e93106f4b91cfb3b24ffc311a3b432b091a86"
PAINT_REPO = "tencent/Hunyuan3D-2.1"
PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"
DINO_REPO = "facebook/dinov2-giant"
WORK_ROOT = Path.home() / ".cache" / "abstract3d" / "experimental" / "hy3dpaint-spike"

RESULTS: list[dict] = []


def record(stage: str, ok: bool, detail: str) -> None:
    RESULTS.append({"stage": stage, "ok": ok, "detail": detail})
    print(f"[{'OK' if ok else 'REFUSED'}] {stage}: {detail}", flush=True)


def rss_gb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 1024**3


def check_contention() -> None:
    # Repo discipline: never stack a heavy load onto a running generation.
    probe = subprocess.run(["pgrep", "-fl", "i23d|bust_|harmonize|regen_views"], capture_output=True, text=True)
    lines = [ln for ln in probe.stdout.strip().splitlines() if ln and "hy3dpaint_mps_spike" not in ln]
    if lines:
        record("gpu-contention-preflight", False, f"competing jobs running: {lines[:3]} — aborting")
        finish(2)
    record("gpu-contention-preflight", True, "no competing generation processes")


def clone_fork() -> Path | None:
    dest = WORK_ROOT / "fork" / FORK_COMMIT
    if (dest / "hy3dpaint" / "textureGenPipeline.py").exists():
        record("clone-fork", True, f"already present: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".tmp-{os.getpid()}"
    try:
        subprocess.run(["git", "clone", "--filter=blob:none", FORK_URL, str(tmp)], check=True, capture_output=True, text=True)
        subprocess.run(["git", "-C", str(tmp), "checkout", "--detach", FORK_COMMIT], check=True, capture_output=True, text=True)
        shutil.rmtree(tmp / ".git", ignore_errors=True)
        tmp.rename(dest)
        record("clone-fork", True, f"pinned {FORK_COMMIT[:12]} -> {dest}")
        return dest
    except Exception as exc:  # noqa: BLE001 - spike reports every failure verbatim
        record("clone-fork", False, f"{type(exc).__name__}: {getattr(exc, 'stderr', exc)}")
        shutil.rmtree(tmp, ignore_errors=True)
        return None


def build_ext_inplace(setup_dir: Path, stage: str) -> bool:
    """Build a torch cpp extension in place; nothing is installed into the venv."""
    try:
        proc = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=str(setup_dir),
            capture_output=True,
            text=True,
            timeout=1200,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-12:]
            record(stage, False, "build failed: " + " | ".join(tail))
            return False
        built = sorted(str(p.name) for p in setup_dir.glob("*.so")) + sorted(
            str(p.relative_to(setup_dir)) for p in setup_dir.glob("**/*.so") if p.parent != setup_dir
        )
        record(stage, True, f"built artifacts: {built or 'none found (check layout)'}")
        return bool(built)
    except Exception as exc:  # noqa: BLE001
        record(stage, False, f"{type(exc).__name__}: {exc}")
        return False


def smoke_rasterizer(fork: Path) -> bool:
    """One real triangle through the CPU rasterizer path (device-dispatch: cpu tensor -> cpu kernel)."""
    raster_dir = fork / "hy3dpaint" / "custom_rasterizer"
    sys.path.insert(0, str(raster_dir))
    try:
        import torch

        kernel = importlib.import_module("custom_rasterizer_kernel")
        # Clip-space homogeneous vertices of one triangle covering image center.
        V = torch.tensor(
            [[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]],
            dtype=torch.float32,
        )
        F = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        D = torch.zeros(0, dtype=torch.float32)
        findices, barycentric = kernel.rasterize_image(V, F, D, 64, 64, 0.1, 0)
        covered = int((findices > 0).sum())
        ok = tuple(findices.shape) == (64, 64) and tuple(barycentric.shape) == (64, 64, 3) and covered > 200
        record(
            "rasterizer-cpu-smoke",
            ok,
            f"findices {tuple(findices.shape)}, barycentric {tuple(barycentric.shape)}, covered texels {covered}",
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        record("rasterizer-cpu-smoke", False, f"{type(exc).__name__}: {exc}")
        return False


def download_weights(skip: bool) -> tuple[str | None, str | None]:
    if skip:
        record("download-weights", True, "skipped by flag (--skip-download)")
        return None, None
    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        paint_dir = snapshot_download(repo_id=PAINT_REPO, allow_patterns=[f"{PAINT_SUBFOLDER}/*"])
        record("download-paint-weights", True, f"{PAINT_REPO}/{PAINT_SUBFOLDER} in {time.time()-t0:.0f}s -> {paint_dir}")
    except Exception as exc:  # noqa: BLE001
        record("download-paint-weights", False, f"{type(exc).__name__}: {exc}")
        return None, None
    try:
        t0 = time.time()
        dino_dir = snapshot_download(repo_id=DINO_REPO, allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"])
        record("download-dino-weights", True, f"{DINO_REPO} in {time.time()-t0:.0f}s -> {dino_dir}")
    except Exception as exc:  # noqa: BLE001
        record("download-dino-weights", False, f"{type(exc).__name__}: {exc}")
        return paint_dir, None
    return paint_dir, dino_dir


def load_pipeline(fork: Path, device: str, paint_dir: str, dino_dir: str | None) -> bool:
    """Load the multiview paint UNet stack to the target device in fp16. NO generation."""
    hy = fork / "hy3dpaint"
    # The fork uses package-relative imports (`from ..hunyuanpaintpbr...`); import it as
    # a namespace package from the FORK ROOT (hy3dpaint/ has no __init__.py — PEP 420).
    sys.path.insert(0, str(fork))
    base_rss = rss_gb()
    try:
        import torch

        if device == "mps" and not torch.backends.mps.is_available():
            record("load-pipeline", False, "MPS not available in this torch build")
            return False

        # multiviewDiffusionNet reads a Hunyuan3DPaintConfig-shaped object; we build a
        # minimal stand-in so the spike does not depend on the fork's ctor signature
        # (and so imageSuperNet / RealESRGAN stays entirely out of the load).
        # Fork contract (verified at 3d8e931): device, multiview_cfg_path,
        # paint_model_path (LOCAL dir, required), dino_ckpt_path, diffusion_backend.
        class _SpikeCfg:
            pass

        cfg = _SpikeCfg()
        cfg.device = device
        cfg.multiview_cfg_path = str(hy / "cfgs" / "hunyuan-paint-pbr.yaml")
        cfg.paint_model_path = str(Path(paint_dir) / PAINT_SUBFOLDER)
        cfg.dino_ckpt_path = dino_dir or DINO_REPO
        cfg.diffusion_backend = "pytorch"  # MLX arm needs converted weights; out of spike scope

        from hy3dpaint.utils.multiview_utils import multiviewDiffusionNet  # fork pkg, path-injected

        # diffusers 0.38 refuses model-dir custom code (the fork's own pinned
        # unet/modules.py, which it copies into the model dir) without
        # trust_remote_code=True. The code is pinned + inspectable (commit above), so
        # the SPIKE shims the flag; the real integration must vendor the UNet class
        # instead (abstractmusic precedent: no trust_remote_code at runtime).
        from hy3dpaint.hunyuanpaintpbr.pipeline import HunyuanPaintPipeline

        _orig_fp = HunyuanPaintPipeline.from_pretrained.__func__

        def _trusted_fp(cls, *a, **k):
            k.setdefault("trust_remote_code", True)
            return _orig_fp(cls, *a, **k)

        HunyuanPaintPipeline.from_pretrained = classmethod(_trusted_fp)
        record("diffusers-drift-note", True, "diffusers 0.38 requires trust_remote_code for unet custom code; shimmed for spike only")

        t0 = time.time()
        net = multiviewDiffusionNet(cfg)
        load_s = time.time() - t0

        unet = net.pipeline.unet
        n_params = sum(p.numel() for p in unet.parameters()) / 1e9
        devices = {str(p.device) for p in unet.parameters()}
        dtypes = {str(p.dtype) for p in unet.parameters()}
        has_dino = hasattr(net, "dino_v2")
        record(
            "load-pipeline",
            True,
            f"UNet {n_params:.2f}B params on {sorted(devices)} {sorted(dtypes)}; "
            f"dino_loaded={has_dino}; view_size={getattr(net.pipeline, 'view_size', '?')}; "
            f"load {load_s:.0f}s; RSS {base_rss:.1f} -> {rss_gb():.1f} GB",
        )

        # Dry shape-check: the conditioning encoders exist and accept a dummy 512x512
        # image on the CPU side (no denoise, no VAE decode, no scheduler stepping).
        try:
            from PIL import Image
            import numpy as np

            probe = Image.fromarray(np.full((512, 512, 3), 128, dtype=np.uint8))
            if has_dino:
                with torch.no_grad():
                    feats = net.dino_v2([probe])
                record("dino-dry-forward", True, f"dino feature shape {tuple(feats.shape)}")
            else:
                record("dino-dry-forward", True, "pipeline reports use_dino=False; nothing to probe")
        except Exception as exc:  # noqa: BLE001
            record("dino-dry-forward", False, f"{type(exc).__name__}: {exc}")
        return True
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc().strip().splitlines()[-6:]
        record("load-pipeline", False, f"{type(exc).__name__}: {exc} | " + " | ".join(tb))
        return False


def finish(code: int) -> None:
    report_path = WORK_ROOT / "spike_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"commit": FORK_COMMIT, "results": RESULTS}, indent=2), encoding="utf-8")
    print(f"\n== spike report -> {report_path}")
    for row in RESULTS:
        print(f"  {'OK     ' if row['ok'] else 'REFUSED'} {row['stage']}")
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--skip-download", action="store_true", help="stop after the build+rasterizer stages")
    args = parser.parse_args()

    check_contention()
    fork = clone_fork()
    if fork is None:
        finish(1)

    raster_ok = build_ext_inplace(fork / "hy3dpaint" / "custom_rasterizer", "build-custom-rasterizer")
    build_ext_inplace(fork / "hy3dpaint" / "DifferentiableRenderer", "build-mesh-inpaint")  # non-fatal: CPU fallback exists

    if raster_ok:
        smoke_rasterizer(fork)

    paint_dir, dino_dir = download_weights(args.skip_download)
    if not args.skip_download and paint_dir is not None:
        load_pipeline(fork, args.device, paint_dir, dino_dir)

    critical = [r for r in RESULTS if r["stage"] in {"build-custom-rasterizer", "rasterizer-cpu-smoke", "download-paint-weights", "load-pipeline"}]
    finish(0 if critical and all(r["ok"] for r in critical) else 1)


if __name__ == "__main__":
    main()
