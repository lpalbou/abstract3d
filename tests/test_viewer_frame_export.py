"""EXP-01 regression guards: exports present glTF viewer orientation.

The pipeline's canonical object frame is Z-up / front +X; glTF mandates
Y-up / front +Z. Exports bake the exact axis permutation (x,y,z)->(y,z,x)
into vertices so both GLB and OBJ load upright in standards-compliant
viewers, while `viewer_frame=False` preserves the canonical frame for
internal working files (geometry.glb consumed by rebakes).
"""
from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

from abstract3d.backends.triposr_runtime import (
    _mesh_export_bytes,
    _tripo_export_obj_with_textures,
)


def _textured_canonical_mesh() -> trimesh.Trimesh:
    """Tall-in-Z textured box in the pipeline's canonical frame."""
    mesh = trimesh.creation.box(extents=(1.0, 0.6, 2.0))
    rng = np.random.default_rng(7)
    texture = Image.fromarray(rng.uniform(0, 255, size=(32, 32, 3)).astype(np.uint8), "RGB")
    uv = (mesh.vertices[:, :2] - mesh.vertices[:, :2].min(0)) / np.ptp(
        mesh.vertices[:, :2], axis=0
    )
    mesh.visual = TextureVisuals(
        uv=uv,
        material=PBRMaterial(
            baseColorTexture=texture,
            baseColorFactor=(255, 255, 255, 255),
            metallicFactor=0.0,
            roughnessFactor=1.0,
        ),
    )
    return mesh


def test_glb_export_is_yup_and_float_exact() -> None:
    mesh = _textured_canonical_mesh()
    glb = _mesh_export_bytes(mesh, file_type="glb")
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    extents = loaded.bounds[1] - loaded.bounds[0]
    # Tall axis must now be Y (index 1), not Z.
    assert int(np.argmax(extents)) == 1
    # The permutation itself is float-exact; the only representation change
    # is glTF's mandated float32 storage. Inverting the permutation must
    # restore the canonical vertices exactly at float32 precision.
    restored = np.asarray(loaded.vertices, dtype=np.float32)[:, [2, 0, 1]]
    canonical32 = np.asarray(mesh.vertices, dtype=np.float32)
    assert np.array_equal(np.sort(restored, axis=0), np.sort(canonical32, axis=0))


def test_glb_internal_frame_is_preserved_on_request() -> None:
    mesh = _textured_canonical_mesh()
    glb = _mesh_export_bytes(mesh, file_type="glb", viewer_frame=False)
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh")
    extents = loaded.bounds[1] - loaded.bounds[0]
    assert int(np.argmax(extents)) == 2  # still Z-tall (canonical)


def test_glb_texture_bytes_untouched_by_orientation() -> None:
    mesh = _textured_canonical_mesh()

    def embedded_png(data: bytes) -> bytes:
        import json
        import struct

        offset = 12
        gltf_json, binary = None, b""
        while offset + 8 <= len(data):
            length, kind = struct.unpack_from("<II", data, offset)
            chunk = data[offset + 8 : offset + 8 + length]
            if kind == 0x4E4F534A:
                gltf_json = json.loads(chunk)
            elif kind == 0x004E4942:
                binary = chunk
            offset += 8 + length + ((4 - length % 4) % 4 if length % 4 else 0)
        view = gltf_json["bufferViews"][gltf_json["images"][0]["bufferView"]]
        start = view.get("byteOffset", 0)
        return binary[start : start + view["byteLength"]]

    yup = _mesh_export_bytes(mesh, file_type="glb")
    canonical = _mesh_export_bytes(mesh, file_type="glb", viewer_frame=False)
    assert embedded_png(yup) == embedded_png(canonical)


def test_obj_export_matches_glb_orientation() -> None:
    mesh = _textured_canonical_mesh()
    obj_bytes, _ = _tripo_export_obj_with_textures(mesh)
    vertices = np.array(
        [
            [float(part) for part in line.split()[1:4]]
            for line in obj_bytes.decode("utf-8").splitlines()
            if line.startswith("v ")
        ]
    )
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    assert int(np.argmax(extents)) == 1  # Y-tall, same as GLB
