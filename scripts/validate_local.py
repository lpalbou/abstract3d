#!/usr/bin/env python3
"""Run a reproducible local validation suite for Abstract3D."""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import platform
import signal
import subprocess
import shutil
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image

from abstract3d import Scene3DManager
from abstract3d.image_composition import describe_image_binding
from abstract3d.rendering import build_case_contact_sheet


DEFAULT_T23D_PROMPTS = [
    "a ceramic teapot with a curved spout and matte glaze",
    "a mid-century lounge chair with walnut legs and woven fabric",
]

DEFAULT_I23D_IMAGE_PROMPTS = [
    "a studio product photo of a red espresso machine with rounded corners",
    "a studio product photo of a carved wooden owl figurine",
]

try:
    import psutil
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None


def _default_validation_output_dir(*, root: Path, backend: str, model_subfolder: str | None) -> Path:
    normalized_backend = str(backend or "triposr").strip().lower()
    if "step1x" in normalized_backend:
        selected_subfolder = str(model_subfolder or "").strip().lower()
        suffix = "label" if not selected_subfolder or "label" in selected_subfolder else "base"
        return root / "artifacts" / "validation" / f"local-step1x-{suffix}"
    if "triposr" in normalized_backend:
        return root / "artifacts" / "validation" / "local-triposr"
    if "trellis2" in normalized_backend:
        return root / "artifacts" / "validation" / "local-trellis2"
    backend_slug = normalized_backend.replace("_", "-").replace(":", "-")
    return root / "artifacts" / "validation" / f"local-{backend_slug}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a local Abstract3D backend.")
    parser.add_argument("--backend", default="triposr")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--publish-doc-assets", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Compute device (auto resolves to mps, cuda, then cpu depending on the host).",
    )
    parser.add_argument(
        "--mc-resolution",
        type=int,
        default=None,
        help=(
            "Grid resolution override. When omitted, each backend applies its own default "
            "(TripoSR 256, Step1X per-device, Hunyuan3D octree 384)."
        ),
    )
    parser.add_argument("--texture-mode", default=None, choices=["vertex_color", "baked_basecolor"])
    parser.add_argument("--texture-resolution", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--max-facenum", type=int, default=None)
    parser.add_argument("--foreground-ratio", type=float, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-subfolder", default=None)
    parser.add_argument("--image-provider", default=None)
    parser.add_argument("--image-model", default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image-prompt", action="append", default=[])
    parser.add_argument("--case-timeout-s", type=int, default=900)
    parser.add_argument("--case-rss-limit-gb", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-case-manifest", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--write-result", default=None, help=argparse.SUPPRESS)
    return parser


def _save_generated_input(path: Path, payload: bytes) -> Dict[str, Any]:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return {
        "path": str(path),
        "width": image.width,
        "height": image.height,
        "bytes": path.stat().st_size,
    }


def _safe_mean(values: Iterable[float | int | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return round(mean(filtered), 4)


def _release_mlx_cache() -> None:
    gc.collect()
    try:
        mx = __import__("mlx.core", fromlist=["core"])
    except Exception:
        return
    try:
        clear_cache = getattr(mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    except Exception:
        pass
    try:
        reset_peak = getattr(mx, "reset_peak_memory", None)
        if callable(reset_peak):
            reset_peak()
    except Exception:
        pass


def _gib(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / (1024 ** 3), 4)
    except Exception:
        return None


def _build_summary(
    rows: List[Dict[str, Any]],
    *,
    backend: str,
    device: str,
    mc_resolution: int,
    image_provider: str | None,
    image_model: str | None,
    texture_mode: str | None,
    texture_resolution: int | None,
) -> Dict[str, Any]:
    t23d_rows = [row for row in rows if row.get("mode") == "t23d"]
    i23d_rows = [row for row in rows if row.get("mode") == "i23d"]
    return {
        "backend": backend,
        "cases": len(rows),
        "t23d_cases": len(t23d_rows),
        "i23d_cases": len(i23d_rows),
        "device": device,
        "mc_resolution": mc_resolution,
        "image_provider": describe_image_binding(image_provider),
        "image_model": image_model,
        "texture_mode": texture_mode,
        "texture_resolution": texture_resolution,
        "avg_total_s": _safe_mean(row.get("timings_s", {}).get("total") for row in rows),
        "avg_inference_s": _safe_mean(row.get("timings_s", {}).get("inference") for row in rows),
        "avg_mesh_s": _safe_mean(row.get("timings_s", {}).get("mesh") for row in rows),
        "avg_preprocess_s": _safe_mean(row.get("timings_s", {}).get("preprocess") for row in rows),
        "avg_t23d_image_s": _safe_mean(row.get("timings_s", {}).get("source_image_generation") for row in t23d_rows),
        "avg_rss_gib": _safe_mean(_gib(row.get("memory", {}).get("rss_bytes")) for row in rows),
        "avg_mps_allocated_gib": _safe_mean(_gib(row.get("memory", {}).get("mps_allocated_bytes")) for row in rows),
        "avg_vertices": _safe_mean(row.get("vertex_count") for row in rows),
        "avg_faces": _safe_mean(row.get("face_count") for row in rows),
    }


def _summary_markdown(*, rows: List[Dict[str, Any]], summary: Dict[str, Any], inputs: List[Dict[str, Any]]) -> str:
    succeeded = [row for row in rows if row.get("status") == "succeeded"]
    failed = [row for row in rows if row.get("status") != "succeeded"]
    lines = [
        "# Abstract3D Local Validation",
        "",
        f"- Backend: `{summary['backend']}`",
        f"- Platform: `{platform.platform()}`",
        f"- Python: `{platform.python_version()}`",
        f"- Device: `{summary['device']}`",
        f"- Marching cubes resolution: `{summary['mc_resolution']}`",
        f"- Texture mode: `{summary.get('texture_mode')}`",
        f"- Texture resolution: `{summary.get('texture_resolution')}`",
        f"- Image provider for composed `t23d`: `{summary['image_provider']}`",
        f"- Image model for composed `t23d`: `{describe_image_binding(summary['image_model'])}`",
        "",
        "## Aggregate",
        "",
        f"- Cases: `{summary['cases']}` (`t23d={summary['t23d_cases']}`, `i23d={summary['i23d_cases']}`)",
        f"- Successful cases: `{len(succeeded)}`",
        f"- Failed cases: `{len(failed)}`",
        f"- Average total time: `{summary['avg_total_s']}` s",
        f"- Average inference time: `{summary['avg_inference_s']}` s",
        f"- Average mesh extraction time: `{summary['avg_mesh_s']}` s",
        f"- Average preprocessing time: `{summary['avg_preprocess_s']}` s",
        f"- Average text-to-image composition time: `{summary['avg_t23d_image_s']}` s",
        f"- Average RSS: `{summary['avg_rss_gib']}` GiB",
        f"- Average MPS allocated: `{summary['avg_mps_allocated_gib']}` GiB",
        f"- Average vertices: `{summary['avg_vertices']}`",
        f"- Average faces: `{summary['avg_faces']}`",
        "",
        "## Generated Inputs",
        "",
    ]
    for item in inputs:
        lines.append(
            f"- `{Path(item['path']).name}`: `{item['width']}x{item['height']}`, `{item['bytes']}` bytes"
        )
    lines.extend(
        [
            "",
            "## Per Case",
            "",
            "| Case | Mode | Status | Total s | Image s | Prep s | Infer s | Mesh s | Vertices | Faces | RSS GiB | MPS GiB |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        timings = row.get("timings_s", {})
        memory = row.get("memory", {})
        lines.append(
            "| {case} | {mode} | {status} | {total} | {img} | {prep} | {infer} | {mesh} | {verts} | {faces} | {rss} | {mps} |".format(
                case=row.get("case_id", "?"),
                mode=row.get("mode", "?"),
                status=row.get("status", "?"),
                total=timings.get("total"),
                img=timings.get("source_image_generation"),
                prep=timings.get("preprocess"),
                infer=timings.get("inference"),
                mesh=timings.get("mesh"),
                verts=row.get("vertex_count"),
                faces=row.get("face_count"),
                rss=_gib(memory.get("rss_bytes")),
                mps=_gib(memory.get("mps_allocated_bytes")),
            )
        )
        if row.get("status") != "succeeded":
            lines.append("")
            lines.append(f"- `{row.get('case_id', '?')}` failure: `{row.get('error') or row.get('status')}`")
    return "\n".join(lines) + "\n"


def _build_case_specs(
    *,
    prompts: Sequence[str],
    images: Sequence[str],
    image_prompts: Sequence[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        case_dir = output_dir / f"{index:02d}_t23d"
        cases.append(
            {
                "case_id": case_dir.name,
                "mode": "t23d",
                "prompt": prompt,
                "output_dir": str(case_dir),
                "backend": args.backend,
                "image_provider": args.image_provider,
                "image_model": args.image_model,
                "mc_resolution": args.mc_resolution,
                "device": args.device,
                "model": args.model,
                "model_subfolder": args.model_subfolder,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "max_facenum": args.max_facenum,
                "foreground_ratio": args.foreground_ratio,
                "texture_mode": args.texture_mode,
                "texture_resolution": args.texture_resolution,
                "remove_background": None,
            }
        )
    for offset, image_path in enumerate(images, start=1):
        case_dir = output_dir / f"{len(prompts) + offset:02d}_i23d"
        cases.append(
            {
                "case_id": case_dir.name,
                "mode": "i23d",
                "prompt": image_prompts[offset - 1] if len(image_prompts) >= offset else None,
                "image": str(image_path),
                "output_dir": str(case_dir),
                "backend": args.backend,
                "mc_resolution": args.mc_resolution,
                "device": args.device,
                "model": args.model,
                "model_subfolder": args.model_subfolder,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "max_facenum": args.max_facenum,
                "foreground_ratio": args.foreground_ratio,
                "texture_mode": args.texture_mode,
                "texture_resolution": args.texture_resolution,
                "remove_background": None,
            }
        )
    return cases


def _case_paths(case_dir: Path) -> tuple[Path, Path]:
    return case_dir / "case_manifest.json", case_dir / "result.json"


def _default_case_rss_limit_gb(*, backend: str, device: str) -> float | None:
    """Default per-case memory guard by backend.

    The guard exists to keep one pathological case from taking down the
    host, so it must apply whenever a heavy local backend actually runs —
    including through the `auto` device default. Requesting `cpu` on the
    heavy backends still deserves the guard: those runs use the same or
    more system memory than accelerator runs.
    """
    normalized_backend = str(backend or "").strip().lower()
    del device
    if "step1x" in normalized_backend:
        return 64.0
    if "hunyuan" in normalized_backend:
        return 64.0
    return None


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _coerce_serializable_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    payload["status"] = str(payload.get("status") or "succeeded")
    return payload


def _make_failure_contact_sheet(*, case: Mapping[str, Any], error: str, output_path: Path) -> str:
    title = str(case.get("prompt") or case.get("mode") or case.get("case_id") or "validation failure")
    source = Image.new("RGB", (320, 320), "#ece9e2")
    views = [Image.new("RGB", (320, 320), "#f3f2ee") for _ in range(4)]
    stats = [
        "status failed",
        f"mode {case.get('mode')}",
        f"backend {case.get('backend')}",
        (error or "unknown error")[:120],
    ]
    sheet = build_case_contact_sheet(title=title, source_image=source, views=views, stats_lines=stats)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _failure_row(*, case: Mapping[str, Any], status: str, error: str, timeout_s: int | None = None) -> Dict[str, Any]:
    case_dir = Path(str(case["output_dir"])).expanduser().resolve()
    contact_sheet_path = _make_failure_contact_sheet(
        case=case,
        error=error,
        output_path=case_dir / "contact_sheet.png",
    )
    row: Dict[str, Any] = {
        "case_id": case.get("case_id"),
        "mode": case.get("mode"),
        "status": status,
        "error": error,
        "contact_sheet_path": contact_sheet_path,
        "timings_s": {
            "source_image_generation": None,
            "preprocess": None,
            "inference": None,
            "mesh": None,
            "total": None,
        },
        "memory": {
            "rss_bytes": None,
            "mps_allocated_bytes": None,
        },
        "vertex_count": None,
        "face_count": None,
        "prompt": case.get("prompt"),
        "input_image": case.get("image"),
    }
    if timeout_s is not None:
        row["case_timeout_s"] = int(timeout_s)
    return row


def _process_tree_rss_bytes(pid: int) -> int | None:
    if psutil is None:
        return None
    try:
        process = psutil.Process(int(pid))
        total = int(process.memory_info().rss)
        for child in process.children(recursive=True):
            try:
                total += int(child.memory_info().rss)
            except Exception:
                continue
        return total
    except Exception:
        return None


def _write_case_manifest(path: Path, case: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(case), indent=2, sort_keys=True), encoding="utf-8")


def _terminate_process_tree(process: subprocess.Popen, *, force: bool) -> None:
    """Stop a case subprocess and all of its children on any host OS.

    POSIX hosts can address the whole process group directly. Windows has no
    process groups in that sense, so we walk the child tree through psutil
    when it is available and fall back to terminating the direct child.
    """
    sig_name = "SIGKILL" if force else "SIGTERM"
    if hasattr(os, "killpg") and hasattr(signal, sig_name):
        try:
            os.killpg(process.pid, getattr(signal, sig_name))
            return
        except Exception:
            pass
    if psutil is not None:
        try:
            root = psutil.Process(process.pid)
            children = root.children(recursive=True)
            for child in children:
                try:
                    child.kill() if force else child.terminate()
                except Exception:
                    continue
        except Exception:
            pass
    try:
        process.kill() if force else process.terminate()
    except Exception:
        pass


def _run_case_subprocess(
    *,
    case: Mapping[str, Any],
    timeout_s: int,
    rss_limit_gb: float | None,
    resume: bool,
) -> Dict[str, Any]:
    case_dir = Path(str(case["output_dir"])).expanduser().resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, result_path = _case_paths(case_dir)
    if resume and result_path.exists():
        return _coerce_serializable_case_row(_read_json(result_path))
    _write_case_manifest(manifest_path, case)
    env = dict(os.environ)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC) if not pythonpath else f"{SRC}{os.pathsep}{pythonpath}"
    # Isolate each case in its own process group/session where the platform
    # supports it, so guard kills can address the whole child tree. Windows
    # does not support start_new_session; the psutil tree fallback in
    # _terminate_process_tree covers that case.
    popen_kwargs: Dict[str, Any] = {}
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run-case-manifest",
            str(manifest_path),
            "--write-result",
            str(result_path),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **popen_kwargs,
    )
    started = time.monotonic()
    kill_status: str | None = None
    kill_error: str | None = None
    observed_rss_bytes: int | None = None
    rss_limit_bytes = None if rss_limit_gb is None else int(float(rss_limit_gb) * (1024 ** 3))
    while process.poll() is None:
        elapsed = time.monotonic() - started
        if elapsed > max(1, int(timeout_s)):
            kill_status = "timed_out"
            kill_error = f"Case exceeded {int(timeout_s)}s timeout."
            break
        if rss_limit_bytes is not None:
            observed_rss_bytes = _process_tree_rss_bytes(process.pid)
            if observed_rss_bytes is not None and observed_rss_bytes > int(rss_limit_bytes):
                kill_status = "memory_guard_killed"
                kill_error = (
                    f"Case exceeded RSS limit of {float(rss_limit_gb):.1f} GiB "
                    f"(observed {_gib(observed_rss_bytes)} GiB)."
                )
                break
        time.sleep(1.0)
    if kill_status is not None:
        _terminate_process_tree(process, force=False)
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process, force=True)
            stdout, stderr = process.communicate()
        failure = _failure_row(
            case=case,
            status=kill_status,
            error=str(kill_error),
            timeout_s=int(timeout_s),
        )
        failure["exit_code"] = process.returncode
        if observed_rss_bytes is not None:
            failure["observed_rss_gib"] = _gib(observed_rss_bytes)
        failure["stdout"] = stdout.strip() or None
        failure["stderr"] = stderr.strip() or None
        result_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
        return failure
    stdout, stderr = process.communicate()
    if result_path.exists():
        row = _coerce_serializable_case_row(_read_json(result_path))
        row.setdefault("exit_code", process.returncode)
        if row.get("status") != "succeeded" and stdout.strip():
            row.setdefault("stdout", stdout.strip())
        if row.get("status") != "succeeded" and stderr.strip():
            row.setdefault("stderr", stderr.strip())
        return row
    error = stderr.strip() or stdout.strip() or f"Case process exited with code {process.returncode}."
    failure = _failure_row(case=case, status="failed", error=error)
    failure["exit_code"] = process.returncode
    failure["stdout"] = stdout.strip() or None
    failure["stderr"] = stderr.strip() or None
    result_path.write_text(json.dumps(failure, indent=2, sort_keys=True), encoding="utf-8")
    return failure


def _run_single_case(manifest_path: Path, result_path: Path) -> int:
    case = _read_json(manifest_path)
    manager = Scene3DManager(backend_id=case["backend"])
    output_dir = str(case["output_dir"])
    result: Dict[str, Any]
    started = time.perf_counter()
    try:
        common_kwargs = {
            "device": case.get("device"),
            "model": case.get("model"),
            "model_subfolder": case.get("model_subfolder"),
            "num_inference_steps": case.get("num_inference_steps"),
            "guidance_scale": case.get("guidance_scale"),
            "texture_mode": case.get("texture_mode"),
            "texture_resolution": case.get("texture_resolution"),
        }
        if case.get("mc_resolution") is not None:
            common_kwargs["mc_resolution"] = case.get("mc_resolution")
        if case.get("foreground_ratio") is not None:
            common_kwargs["foreground_ratio"] = case.get("foreground_ratio")
        if case.get("max_facenum") is not None:
            common_kwargs["max_facenum"] = case.get("max_facenum")
        if case.get("label") is not None:
            common_kwargs["label"] = case.get("label")
        if case["mode"] == "t23d":
            out = manager.t23d(
                str(case["prompt"]),
                output_dir=output_dir,
                image_provider=case.get("image_provider"),
                image_model=case.get("image_model"),
                **common_kwargs,
            )
        else:
            out = manager.i23d(
                str(case["image"]),
                prompt=case.get("prompt"),
                output_dir=output_dir,
                remove_background=case.get("remove_background"),
                **common_kwargs,
            )
        meta = dict(out.get("metadata") or {}) if isinstance(out, dict) else {}
        meta["case_id"] = case.get("case_id")
        meta["mode"] = case.get("mode")
        meta["status"] = "succeeded"
        meta["wall_clock_s"] = round(time.perf_counter() - started, 4)
        result = meta
    except Exception as exc:
        result = _failure_row(case=case, status="failed", error=f"{type(exc).__name__}: {exc}")
        result["wall_clock_s"] = round(time.perf_counter() - started, 4)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "succeeded" else 1


def main() -> int:
    args = _parser().parse_args()
    if args.run_case_manifest:
        manifest_path = Path(args.run_case_manifest).expanduser().resolve()
        result_path = Path(args.write_result).expanduser().resolve() if args.write_result else manifest_path.with_name("result.json")
        return _run_single_case(manifest_path, result_path)
    backend_slug = str(args.backend or "triposr").strip().lower().replace("_", "-").replace(":", "-")
    output_dir = Path(
        args.output_dir
        or _default_validation_output_dir(
            root=ROOT,
            backend=args.backend,
            model_subfolder=args.model_subfolder,
        )
    ).expanduser().resolve()
    doc_assets_dir = (
        Path(args.publish_doc_assets).expanduser().resolve()
        if args.publish_doc_assets
        else None
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    manager = Scene3DManager(backend_id=args.backend)
    backend = manager.backend
    prompts = list(args.prompt or []) or list(DEFAULT_T23D_PROMPTS)
    image_prompts = list(args.image_prompt or []) or list(DEFAULT_I23D_IMAGE_PROMPTS)
    images = [str(Path(path).expanduser().resolve()) for path in list(args.image or [])]
    generated_inputs: List[Dict[str, Any]] = []

    if not images:
        for index, prompt in enumerate(image_prompts, start=1):
            started = time.perf_counter()
            payload = backend._make_source_image(
                prompt,
                image_provider=args.image_provider,
                image_model=args.image_model,
                image_width=768,
                image_height=768,
                image_seed=1000 + index,
            )
            meta = _save_generated_input(inputs_dir / f"{index:02d}_input.png", payload)
            meta["prompt"] = prompt
            meta["generation_s"] = round(time.perf_counter() - started, 4)
            generated_inputs.append(meta)
            images.append(meta["path"])
            _release_mlx_cache()

    cases = _build_case_specs(
        prompts=prompts,
        images=images,
        image_prompts=image_prompts,
        output_dir=output_dir,
        args=args,
    )
    rows = [
        _run_case_subprocess(
            case=case,
            timeout_s=int(args.case_timeout_s),
            rss_limit_gb=(
                float(args.case_rss_limit_gb)
                if args.case_rss_limit_gb is not None
                else _default_case_rss_limit_gb(backend=args.backend, device=args.device)
            ),
            resume=bool(args.resume),
        )
        for case in cases
    ]
    sheets = []
    for row in rows:
        sheet_path = row.get("contact_sheet_path")
        if isinstance(sheet_path, str) and Path(sheet_path).exists():
            sheets.append(Image.open(sheet_path).convert("RGB"))
    if not sheets:
        placeholder = _make_failure_contact_sheet(
            case={"mode": "validation", "backend": args.backend, "prompt": "validation failed"},
            error="No validation cases completed successfully.",
            output_path=output_dir / "summary" / "empty.png",
        )
        sheets.append(Image.open(placeholder).convert("RGB"))
    from abstract3d.rendering import stack_contact_sheets

    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_sheet = stack_contact_sheets(sheets, columns=2 if len(sheets) > 1 else 1)
    summary_sheet_path = summary_dir / "contact_sheet.png"
    summary_sheet.save(summary_sheet_path)
    stats_path = summary_dir / "stats.json"
    stats_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    aggregate = _build_summary(
        rows,
        backend=backend_slug,
        device=args.device,
        mc_resolution=args.mc_resolution,
        image_provider=args.image_provider,
        image_model=args.image_model,
        texture_mode=args.texture_mode,
        texture_resolution=args.texture_resolution,
    )
    aggregate["generated_inputs"] = generated_inputs

    summary_json_path = output_dir / "summary" / "summary.json"
    summary_json_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")

    summary_md_path = output_dir / "summary" / "summary.md"
    summary_md_path.write_text(
        _summary_markdown(rows=rows, summary=aggregate, inputs=generated_inputs),
        encoding="utf-8",
    )

    if doc_assets_dir is not None:
        doc_assets_dir.mkdir(parents=True, exist_ok=True)
        for name in ("contact_sheet.png", "stats.json", "summary.json", "summary.md"):
            shutil.copy2(output_dir / "summary" / name, doc_assets_dir / name)

    payload = {
        "output_dir": str(output_dir),
        "doc_assets_dir": str(doc_assets_dir) if doc_assets_dir is not None else None,
        "contact_sheet": str(summary_sheet_path),
        "stats": str(stats_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
