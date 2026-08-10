"""End-to-end proof that AbstractCore can generate, manipulate, modify and
analyze 3D objects through the Abstract3D integration.

Stages (each writes real artifacts under --output-dir):
  A. GENERATE through AbstractCore's Python capability lane:
     core.generate(text=..., output={"modality": "scene3d", ...}) -> GLB.
  B. ANALYZE / MANIPULATE / MODIFY through AbstractCore's ToolRegistry,
     executing abstract3d tools exactly the way an agent does
     (analyze -> transform -> re-analyze -> compose scene -> convert -> previews).
  C. GENERATE over HTTP through the AbstractCore server's OpenAI-compatible
     surface: POST /v1/scene3d/generations (i23d from the stage-A source image).
  D. Optional live-LLM tool call: an actual model (LMStudio) receives the
     abstract3d tools and must call analyze_3d_object on the generated file.

Run inside the framework venv:
  python scripts/abstractcore_integration_proof.py --output-dir ./out/proof \
      --device mps --image-provider mlx-gen \
      --image-model AbstractFramework/flux.2-klein-4b-8bit

Every stage records its result in PROOF.md; failures are recorded honestly
and the script exits non-zero if any REQUIRED stage fails (stage D is
best-effort and reported as such).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import urllib.request


@dataclass
class StageResult:
    name: str
    required: bool
    ok: bool
    seconds: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="a small ceramic owl statue, studio photo, neutral background")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--image-provider", default=os.environ.get("ABSTRACT3D_IMAGE_PROVIDER") or "mlx-gen")
    parser.add_argument(
        "--image-model",
        default=os.environ.get("ABSTRACT3D_IMAGE_MODEL") or "AbstractFramework/flux.2-klein-4b-8bit",
    )
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--server-port", type=int, default=8321)
    parser.add_argument("--skip-http", action="store_true", help="Skip the HTTP server stage.")
    parser.add_argument("--skip-llm", action="store_true", help="Skip the live-LLM tool-call stage.")
    parser.add_argument("--llm-provider", default="lmstudio")
    parser.add_argument("--llm-model", default="qwen/qwen3-vl-4b")
    parser.add_argument(
        "--reuse-generated",
        action="store_true",
        help="Reuse an existing stage-A bundle in --output-dir (iteration aid; the final proof should run clean).",
    )
    return parser.parse_args()


def _bundle_dir(root: Path, name: str) -> Path:
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _stage_a_generate(args: argparse.Namespace, root: Path) -> StageResult:
    """Text -> 3D through core.generate output routing (the capability lane)."""
    started = time.perf_counter()
    try:
        from abstractcore.server.capability_generation import create_capability_generation_core

        core = create_capability_generation_core()
        bundle_dir = _bundle_dir(root, "a_generated_t23d")
        result = core.generate(
            text=args.prompt,
            output={
                "modality": "scene3d",
                "task": "text_to_scene3d",
                "provider": "triposr",
                "format": "glb",
                "device": args.device,
                # TripoSR itself is feed-forward (no sampler seed); determinism
                # rides the composed image stage via image_seed.
                "image_seed": args.seed,
                "mc_resolution": args.mc_resolution,
                "image_provider": args.image_provider,
                "image_model": args.image_model,
                "output_dir": str(bundle_dir),
            },
        )
        items = getattr(result, "outputs", {}).get("scene3d", [])
        if not items:
            raise RuntimeError("core.generate returned no scene3d outputs")
        item = items[0]
        glb_path = root / "generated.glb"
        glb_path.write_bytes(bytes(item.data))
        details = {
            "glb_path": str(glb_path),
            "glb_bytes": len(item.data),
            "backend_id": item.backend_id,
            "model": item.model,
            "content_type": item.content_type,
            "bundle_dir": str(bundle_dir),
            "bundle_files": sorted(p.name for p in bundle_dir.rglob("*") if p.is_file()),
        }
        return StageResult("A: generate via core.generate(output=scene3d)", True, True, time.perf_counter() - started, details)
    except Exception as e:
        return StageResult("A: generate via core.generate(output=scene3d)", True, False, time.perf_counter() - started, error=f"{type(e).__name__}: {e}")


def _execute_tool(registry: Any, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    from abstractcore.tools.core import ToolCall

    result = registry.execute_tool(ToolCall(name=name, arguments=arguments, call_id=f"proof-{name}"))
    if not result.success:
        raise RuntimeError(f"{name} failed: {result.error}")
    payload = json.loads(str(result.output))
    if not payload.get("success"):
        raise RuntimeError(f"{name} reported failure: {payload.get('error')}")
    return payload


def _stage_b_tools(root: Path) -> StageResult:
    """Analyze/manipulate/modify via abstract3d tools executed through
    AbstractCore's ToolRegistry — the exact path an agent's tool calls take."""
    started = time.perf_counter()
    try:
        from abstract3d.tools import register_tools
        from abstractcore.tools.registry import ToolRegistry

        registry = ToolRegistry()
        register_tools(registry)

        generated = root / "generated.glb"
        modified = root / "modified.glb"
        scene = root / "scene.glb"
        converted = root / "generated.stl"

        before = _execute_tool(registry, "analyze_3d_object", {"path": str(generated)})

        transform = _execute_tool(
            registry,
            "transform_3d_object",
            {
                "input_path": str(generated),
                "output_path": str(modified),
                "scale": 1.5,
                "rotate_deg": [0.0, 45.0, 0.0],
                "center": True,
            },
        )

        after = _execute_tool(registry, "analyze_3d_object", {"path": str(modified)})

        # Modification must be measurable: linear extents scale by 1.5.
        ratio = after["report"]["surface_area"] / max(before["report"]["surface_area"], 1e-12)
        if not (2.0 < ratio < 2.6):  # 1.5^2 = 2.25 +/- tolerance
            raise RuntimeError(f"transform did not scale the mesh as requested (area ratio {ratio:.3f}, expected ~2.25)")

        composed = _execute_tool(
            registry,
            "compose_3d_scene",
            {
                "parts": [
                    {"path": str(generated), "name": "original", "translate": [-1.0, 0.0, 0.0]},
                    {"path": str(modified), "name": "modified", "translate": [1.0, 0.0, 0.0]},
                ],
                "output_path": str(scene),
            },
        )

        converted_out = _execute_tool(
            registry, "convert_3d_object", {"input_path": str(generated), "output_path": str(converted)}
        )

        previews = {}
        for label, path in (("generated", generated), ("modified", modified), ("scene", scene)):
            preview = _execute_tool(
                registry,
                "render_3d_preview",
                {"input_path": str(path), "output_path": str(root / f"preview_{label}.png")},
            )
            previews[label] = preview["output_path"]

        details = {
            "analysis_before": before["summary"],
            "analysis_after": after["summary"],
            "surface_area_ratio": round(ratio, 4),
            "transform": {k: transform[k] for k in ("output_path", "bounds_before", "bounds_after")},
            "scene": {k: composed[k] for k in ("output_path", "part_count", "vertex_count")},
            "converted": {k: converted_out[k] for k in ("output_path", "output_format", "output_bytes")},
            "previews": previews,
        }
        return StageResult("B: analyze/manipulate/modify via ToolRegistry", True, True, time.perf_counter() - started, details)
    except Exception as e:
        return StageResult("B: analyze/manipulate/modify via ToolRegistry", True, False, time.perf_counter() - started, error=f"{type(e).__name__}: {e}")


def _wait_for_server(port: int, *, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1.0):
                return True
        except Exception:
            time.sleep(1.0)
    return False


def _stage_c_http(args: argparse.Namespace, root: Path) -> StageResult:
    """Image -> 3D over the AbstractCore server's /v1/scene3d/generations."""
    started = time.perf_counter()
    server: Optional[subprocess.Popen] = None
    log_path = root / "server.log"
    try:
        source_candidates = sorted((root / "a_generated_t23d").rglob("input.png"))
        if not source_candidates:
            raise RuntimeError("stage A source image (input.png) not found; cannot run the HTTP i23d proof")
        image_b64 = base64.b64encode(source_candidates[0].read_bytes()).decode()

        env = dict(os.environ)
        env.setdefault("ABSTRACTCORE_SERVER_LOG_LEVEL", "info")
        # Intentional local/dev proof server: no auth token, loopback only.
        env.setdefault("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
        with open(log_path, "wb") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-P",
                    "-m",
                    "uvicorn",
                    "abstractcore.server.app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.server_port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(Path.home()),
            )
            if not _wait_for_server(args.server_port):
                raise RuntimeError(f"AbstractCore server did not open port {args.server_port} (see {log_path})")

            base = f"http://127.0.0.1:{args.server_port}"

            with urllib.request.urlopen(f"{base}/v1/capabilities", timeout=30) as resp:
                capabilities = json.loads(resp.read().decode())

            body = json.dumps(
                {
                    "image_b64": image_b64,
                    "task": "image_to_scene3d",
                    "provider": "triposr",
                    "format": "glb",
                    "device": args.device,
                    "mc_resolution": args.mc_resolution,
                }
            ).encode()
            request = urllib.request.Request(
                f"{base}/v1/scene3d/generations",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=1800) as resp:
                glb_bytes = resp.read()
                headers = {
                    "content-type": resp.headers.get("Content-Type"),
                    "x-abstractcore-backend-id": resp.headers.get("X-AbstractCore-Backend-Id"),
                    "x-abstractcore-model": resp.headers.get("X-AbstractCore-Model"),
                    "x-abstractcore-task": resp.headers.get("X-AbstractCore-Task"),
                }

        http_glb = root / "http_generated.glb"
        http_glb.write_bytes(glb_bytes)
        if not glb_bytes.startswith(b"glTF"):
            raise RuntimeError("HTTP response is not a valid GLB (missing glTF magic)")

        capability_status = capabilities.get("status") if isinstance(capabilities.get("status"), dict) else capabilities
        scene3d_capability = ((capability_status.get("capabilities") or {}).get("scene3d")) or {}
        details = {
            "glb_path": str(http_glb),
            "glb_bytes": len(glb_bytes),
            "headers": headers,
            "capabilities_scene3d": scene3d_capability,
            "server_log": str(log_path),
        }
        return StageResult("C: generate via HTTP /v1/scene3d/generations", True, True, time.perf_counter() - started, details)
    except Exception as e:
        return StageResult("C: generate via HTTP /v1/scene3d/generations", True, False, time.perf_counter() - started, error=f"{type(e).__name__}: {e}")
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except Exception:
                server.kill()


def _stage_d_llm(args: argparse.Namespace, root: Path) -> StageResult:
    """Best-effort: a live LLM receives the abstract3d tools and should call
    analyze_3d_object on the generated GLB (proves AI-usable tool schemas)."""
    started = time.perf_counter()
    try:
        from abstractcore import create_llm
        from abstract3d.tools import abstract3d_tools

        llm = create_llm(args.llm_provider, model=args.llm_model)
        generated = root / "generated.glb"
        response = llm.generate(
            f"Use the analyze_3d_object tool to inspect the 3D file at {generated} and report its face count.",
            tools=abstract3d_tools(),
            execute_tools=False,
        )

        def _call_name(call: Any) -> Optional[str]:
            # Tool calls arrive either as ToolCall-like objects or as
            # OpenAI-wire dicts ({"function": {"name": ..., "arguments": ...}}).
            name = getattr(call, "name", None)
            if name:
                return str(name)
            if isinstance(call, dict):
                if call.get("name"):
                    return str(call["name"])
                function = call.get("function")
                if isinstance(function, dict) and function.get("name"):
                    return str(function["name"])
            return None

        tool_calls = list(getattr(response, "tool_calls", None) or [])
        called = [{"name": _call_name(c)} for c in tool_calls]
        ok = any(c.get("name") == "analyze_3d_object" for c in called)
        details = {"tool_calls": called, "content_head": str(getattr(response, "content", ""))[:300]}
        if not ok:
            return StageResult(
                "D: live LLM emits abstract3d tool call (best-effort)",
                False,
                False,
                time.perf_counter() - started,
                details,
                error="model did not call analyze_3d_object",
            )
        return StageResult("D: live LLM emits abstract3d tool call (best-effort)", False, True, time.perf_counter() - started, details)
    except Exception as e:
        return StageResult(
            "D: live LLM emits abstract3d tool call (best-effort)",
            False,
            False,
            time.perf_counter() - started,
            error=f"{type(e).__name__}: {e}",
        )


def _write_proof(root: Path, args: argparse.Namespace, stages: List[StageResult]) -> Path:
    lines: List[str] = [
        "# AbstractCore x Abstract3D integration proof",
        "",
        f"- date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"- prompt: {args.prompt!r}",
        f"- device: {args.device}; image stage: {args.image_provider} / {args.image_model}",
        f"- seed: {args.seed}; mc_resolution: {args.mc_resolution}",
        "",
    ]
    for stage in stages:
        badge = "PASS" if stage.ok else ("FAIL" if stage.required else "SKIP/FAIL (best-effort)")
        lines.append(f"## {stage.name} — {badge} ({stage.seconds:.1f}s)")
        if stage.error:
            lines.append(f"- error: {stage.error}")
        for key, value in stage.details.items():
            lines.append(f"- {key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
        lines.append("")
    proof = root / "PROOF.md"
    proof.write_text("\n".join(lines), encoding="utf-8")
    return proof


def main() -> int:
    args = _parse_args()
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("ABSTRACT3D_IMAGE_PROVIDER", args.image_provider)
    os.environ.setdefault("ABSTRACT3D_IMAGE_MODEL", args.image_model)

    stages: List[StageResult] = []
    generated = root / "generated.glb"
    if args.reuse_generated and generated.exists():
        stages.append(
            StageResult(
                "A: generate via core.generate(output=scene3d)",
                True,
                True,
                0.0,
                {"glb_path": str(generated), "reused": True, "note": "reused prior stage-A artifacts (--reuse-generated)"},
            )
        )
    else:
        stages.append(_stage_a_generate(args, root))
    if stages[-1].ok:
        stages.append(_stage_b_tools(root))
        if not args.skip_http:
            stages.append(_stage_c_http(args, root))
        if not args.skip_llm:
            stages.append(_stage_d_llm(args, root))

    proof = _write_proof(root, args, stages)
    print(proof.read_text())
    required_failed = [s for s in stages if s.required and not s.ok]
    print(f"PROOF: {proof}")
    return 1 if required_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
