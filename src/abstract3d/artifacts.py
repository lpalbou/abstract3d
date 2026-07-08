"""Artifact helpers for Abstract3D."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Union

from .types import ArtifactRef


def is_artifact_ref(value: Any) -> bool:
    return isinstance(value, dict) and isinstance(value.get("$artifact"), str) and bool(value.get("$artifact"))


def make_artifact_ref(
    artifact_id: str,
    *,
    content_type: Optional[str] = None,
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ArtifactRef:
    ref: ArtifactRef = {"$artifact": str(artifact_id)}
    if content_type:
        ref["content_type"] = str(content_type)
    if filename:
        ref["filename"] = str(filename)
    if isinstance(metadata, dict) and metadata:
        ref["metadata"] = dict(metadata)
    return ref


def artifact_ref_from_store_result(value: Any) -> Optional[ArtifactRef]:
    if is_artifact_ref(value):
        return dict(value)
    raw = getattr(value, "artifact_id", None)
    if isinstance(raw, str) and raw.strip():
        return {"$artifact": raw.strip()}
    if isinstance(value, dict):
        raw = value.get("artifact_id")
        if isinstance(raw, str) and raw.strip():
            return {"$artifact": raw.strip()}
    if isinstance(value, str) and value.strip():
        return {"$artifact": value.strip()}
    return None


def store_bytes(
    artifact_store: Any,
    content: bytes,
    *,
    content_type: str,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    artifact_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Union[bytes, ArtifactRef]:
    store = getattr(artifact_store, "store", None)
    if not callable(store):
        return content
    stored = store(
        bytes(content),
        content_type=str(content_type),
        run_id=run_id,
        tags=dict(tags) if isinstance(tags, dict) else None,
        artifact_id=artifact_id,
    )
    ref = artifact_ref_from_store_result(stored)
    if ref is None:
        return content
    if isinstance(metadata, dict) and metadata:
        ref = dict(ref)
        ref["metadata"] = dict(metadata)
    return ref


def stable_artifact_id(content: bytes) -> str:
    return hashlib.sha256(bytes(content)).hexdigest()[:32]
