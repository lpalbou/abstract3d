#!/usr/bin/env python3
"""Audit exported GLB/MTL material factors against the textured-export contract.

Baked-texture exports must keep the texture bytes authoritative: any factor a
spec-compliant viewer multiplies on top of the base color texture has to be
identity. Concretely, for every material that carries a base color texture:

- GLB: ``baseColorFactor`` [1,1,1,1] (or absent, the glTF default),
  ``metallicFactor`` 0.0 (present! absence means the glTF default 1.0 =
  fully metallic), ``roughnessFactor`` in [0.85, 1.0], ``emissiveFactor``
  absent or [0,0,0].
- MTL: ``Ka``/``Kd`` 1.0 (identity over ``map_Kd``) and ``Ks`` 0.0 (no
  synthetic specular sheen over photo-derived albedo).

Untextured materials are reported but never fail the check, so geometry-only
exports remain unaffected.

Usage:
    python scripts/check_export_materials.py PATH [PATH ...] [--strict] [--json]

PATH may be a .glb file, a .mtl file, or a directory (scanned recursively).
``--strict`` exits 1 when any textured material violates the contract.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

GLB_MAGIC = 0x46546C67
GLB_CHUNK_JSON = 0x4E4F534A

ROUGHNESS_RANGE = (0.85, 1.0)
FACTOR_TOLERANCE = 1e-6


@dataclass
class MaterialReport:
    """Normalized material factors plus contract violations for one material."""

    source: str
    name: str
    textured: bool
    factors: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "textured": self.textured,
            "factors": self.factors,
            "violations": self.violations,
        }


def parse_glb_json(data: bytes) -> Dict[str, Any]:
    """Extract the JSON chunk from a GLB container (binary glTF 2.0)."""
    if len(data) < 20:
        raise ValueError("GLB too short")
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB container")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != GLB_CHUNK_JSON:
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(data[20 : 20 + chunk_length])


def _is_identity(values: Optional[Sequence[float]], expected: Sequence[float]) -> bool:
    if values is None:
        return False
    if len(values) != len(expected):
        return False
    return all(abs(float(v) - float(e)) <= FACTOR_TOLERANCE for v, e in zip(values, expected))


def audit_glb_materials(path: Path) -> List[MaterialReport]:
    gltf = parse_glb_json(path.read_bytes())
    reports: List[MaterialReport] = []
    for index, material in enumerate(gltf.get("materials", [])):
        pbr = material.get("pbrMetallicRoughness", {})
        textured = "baseColorTexture" in pbr
        base_color = pbr.get("baseColorFactor")
        metallic = pbr.get("metallicFactor")
        roughness = pbr.get("roughnessFactor")
        emissive = material.get("emissiveFactor")
        report = MaterialReport(
            source=str(path),
            name=str(material.get("name") or f"material_{index}"),
            textured=textured,
            factors={
                "baseColorFactor": base_color,
                "metallicFactor": metallic,
                "roughnessFactor": roughness,
                "emissiveFactor": emissive,
                "doubleSided": material.get("doubleSided"),
            },
        )
        if textured:
            # Absent baseColorFactor defaults to [1,1,1,1] per glTF spec.
            if base_color is not None and not _is_identity(base_color, (1.0, 1.0, 1.0, 1.0)):
                report.violations.append(f"baseColorFactor {base_color} != [1,1,1,1]")
            # Absent metallicFactor defaults to 1.0 (fully metallic) per spec.
            effective_metallic = 1.0 if metallic is None else float(metallic)
            if abs(effective_metallic) > FACTOR_TOLERANCE:
                report.violations.append(
                    f"metallicFactor {metallic if metallic is not None else 'ABSENT (spec default 1.0)'} != 0.0"
                )
            effective_roughness = 1.0 if roughness is None else float(roughness)
            if not (ROUGHNESS_RANGE[0] <= effective_roughness <= ROUGHNESS_RANGE[1]):
                report.violations.append(f"roughnessFactor {effective_roughness} outside {ROUGHNESS_RANGE}")
            if emissive is not None and not _is_identity(emissive, (0.0, 0.0, 0.0)):
                report.violations.append(f"emissiveFactor {emissive} != [0,0,0]")
        reports.append(report)
    return reports


def parse_mtl_materials(text: str) -> List[Dict[str, Any]]:
    """Parse the factor lines of a Wavefront MTL file."""
    materials: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        key = parts[0]
        if key == "newmtl":
            current = {"name": " ".join(parts[1:]) or "unnamed"}
            materials.append(current)
        elif current is not None and key in {"Ka", "Kd", "Ks", "Ke"}:
            current[key] = [float(value) for value in parts[1:4]]
        elif current is not None and key == "Ns":
            current[key] = float(parts[1])
        elif current is not None and key.startswith("map_"):
            current[key] = " ".join(parts[1:])
    return materials


def audit_mtl_materials(path: Path) -> List[MaterialReport]:
    reports: List[MaterialReport] = []
    for material in parse_mtl_materials(path.read_text(encoding="utf-8", errors="replace")):
        textured = "map_Kd" in material
        report = MaterialReport(
            source=str(path),
            name=str(material.get("name", "unnamed")),
            textured=textured,
            factors={key: material.get(key) for key in ("Ka", "Kd", "Ks", "Ke", "Ns", "map_Kd")},
        )
        if textured:
            for key in ("Ka", "Kd"):
                if not _is_identity(material.get(key), (1.0, 1.0, 1.0)):
                    report.violations.append(f"{key} {material.get(key)} != [1,1,1]")
            if material.get("Ks") is not None and not _is_identity(material.get("Ks"), (0.0, 0.0, 0.0)):
                report.violations.append(f"Ks {material.get('Ks')} != [0,0,0]")
            if material.get("Ke") is not None and not _is_identity(material.get("Ke"), (0.0, 0.0, 0.0)):
                report.violations.append(f"Ke {material.get('Ke')} != [0,0,0]")
        reports.append(report)
    return reports


def collect_targets(paths: Iterable[Path]) -> List[Path]:
    targets: List[Path] = []
    for path in paths:
        if path.is_dir():
            targets.extend(sorted(path.rglob("*.glb")))
            targets.extend(sorted(path.rglob("*.mtl")))
        else:
            targets.append(path)
    return targets


def audit_paths(paths: Iterable[Path]) -> List[MaterialReport]:
    reports: List[MaterialReport] = []
    for target in collect_targets(paths):
        if target.suffix.lower() == ".glb":
            reports.extend(audit_glb_materials(target))
        elif target.suffix.lower() == ".mtl":
            reports.extend(audit_mtl_materials(target))
        else:
            raise ValueError(f"Unsupported file type: {target}")
    return reports


def format_report(report: MaterialReport) -> str:
    status = "OK" if not report.violations else "VIOLATION"
    lines = [f"[{status}] {report.source} :: {report.name} (textured={report.textured})"]
    for key, value in report.factors.items():
        lines.append(f"    {key}: {value}")
    for violation in report.violations:
        lines.append(f"    !! {violation}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help=".glb/.mtl files or directories")
    parser.add_argument("--strict", action="store_true", help="exit 1 on any textured-material violation")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    reports = audit_paths(args.paths)
    if args.json:
        print(json.dumps([report.to_dict() for report in reports], indent=2))
    else:
        for report in reports:
            print(format_report(report))
    violation_count = sum(1 for report in reports if report.violations)
    if not args.json:
        print(f"\n{len(reports)} material(s) checked, {violation_count} with violations.")
    if args.strict and violation_count:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
