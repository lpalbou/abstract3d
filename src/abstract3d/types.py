"""Lightweight data types used by Abstract3D."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


ArtifactRef = Dict[str, Any]


@dataclass(frozen=True)
class GeneratedSceneAsset:
    data: bytes
    mime_type: str
    format: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene3DBundle:
    root_dir: Path
    primary_path: Path
    metadata_path: Path
    source_image_path: Optional[Path] = None
    preview_path: Optional[Path] = None
    contact_sheet_path: Optional[Path] = None
    secondary_paths: Dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class Scene3DModelSpec:
    model_id: str
    provider_id: str
    backend_kind: str
    tasks: tuple[str, ...]
    license: str
    status: str
    source_url: str
    local: bool = True
    remote: bool = False
    validated: bool = False
    apple_silicon: str = "unknown"
    footprint_gb: Optional[float] = None
    model_memory_gb: Optional[float] = None
    notes: str = ""

    def to_capability_model(self) -> Dict[str, Any]:
        raw_metadata: Dict[str, Any] = {
            "backend_kind": self.backend_kind,
            "status": self.status,
            "source_url": self.source_url,
            "apple_silicon": self.apple_silicon,
        }
        if self.notes:
            raw_metadata["notes"] = self.notes
        if self.footprint_gb is not None:
            raw_metadata["footprint_gb"] = float(self.footprint_gb)
        if self.model_memory_gb is not None:
            raw_metadata["model_memory_gb"] = float(self.model_memory_gb)
        return {
            "model_id": self.model_id,
            "provider_id": self.provider_id,
            "tasks": list(self.tasks),
            "modalities": ["scene3d"],
            "local": self.local,
            "remote": self.remote,
            "status": "available" if self.validated else self.status,
            "formats": ["glb", "obj", "zip"],
            "license": self.license,
            "recommended": bool(self.validated),
            "raw_metadata": raw_metadata,
        }
