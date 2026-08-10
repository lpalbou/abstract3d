from __future__ import annotations

import math
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

from abstract3d.errors import InvalidRequestError
from abstract3d.mesh_ops import (
    analyze_mesh,
    as_single_mesh,
    compose_scene,
    convert_mesh,
    load_mesh,
    repair_mesh,
    transform_mesh,
)


@pytest.fixture()
def box_glb(tmp_path: Path) -> Path:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.glb"
    mesh.export(str(path))
    return path


@pytest.fixture()
def sphere_glb(tmp_path: Path) -> Path:
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    path = tmp_path / "sphere.glb"
    mesh.export(str(path))
    return path


def test_load_and_as_single_mesh_roundtrip(box_glb: Path) -> None:
    loaded = load_mesh(box_glb)
    mesh = as_single_mesh(loaded)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_load_rejects_missing_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="not found"):
        load_mesh(tmp_path / "missing.glb")
    bad = tmp_path / "model.xyz"
    bad.write_text("nope")
    with pytest.raises(InvalidRequestError, match="Unsupported mesh format"):
        load_mesh(bad)


def test_analyze_reports_box_facts(box_glb: Path) -> None:
    report = analyze_mesh(box_glb)
    assert report.vertex_count > 0
    assert report.face_count == 12
    assert report.is_watertight is True
    assert report.volume == pytest.approx(6.0, rel=1e-6)
    assert report.surface_area == pytest.approx(22.0, rel=1e-6)
    assert tuple(round(v, 6) for v in report.extents) == (1.0, 2.0, 3.0)
    assert report.connected_components == 1
    assert report.duplicate_faces == 0
    assert report.degenerate_faces == 0
    # JSON-safety: to_dict must round-trip through json
    import json

    json.dumps(report.to_dict())
    assert "watertight=True" in report.summary()


def test_analyze_open_mesh_has_no_volume(tmp_path: Path) -> None:
    # A single triangle is not watertight; volume must be None, not 0/garbage.
    tri = trimesh.Trimesh(
        vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False
    )
    path = tmp_path / "tri.ply"
    tri.export(str(path))
    report = analyze_mesh(path)
    assert report.is_watertight is False
    assert report.volume is None


def test_transform_scale_changes_bounds(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "box_scaled.glb"
    result = transform_mesh(box_glb, out, scale=2.0)
    assert Path(result["output_path"]) == out
    report = analyze_mesh(out)
    assert tuple(round(v, 5) for v in report.extents) == (2.0, 4.0, 6.0)
    # Volume scales by 8x
    assert report.volume == pytest.approx(48.0, rel=1e-5)


def test_transform_rotation_swaps_extents(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "box_rot.glb"
    transform_mesh(box_glb, out, rotate_deg=[90, 0, 0])
    report = analyze_mesh(out)
    # 90 deg around X swaps Y and Z extents of the 1x2x3 box.
    assert tuple(round(abs(v), 5) for v in report.extents) == (1.0, 3.0, 2.0)


def test_transform_translate_moves_centroid(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "box_moved.glb"
    transform_mesh(box_glb, out, translate=[10.0, 0.0, 0.0])
    report = analyze_mesh(out)
    assert report.centroid[0] == pytest.approx(10.0, abs=1e-5)


def test_transform_center_recentres(box_glb: Path, tmp_path: Path) -> None:
    moved = tmp_path / "box_moved.glb"
    transform_mesh(box_glb, moved, translate=[5.0, 5.0, 5.0])
    centered = tmp_path / "box_centered.glb"
    transform_mesh(moved, centered, center=True)
    report = analyze_mesh(centered)
    assert all(abs(c) < 1e-5 for c in report.centroid)


def test_transform_mirror_preserves_volume_sign(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "box_mirror.glb"
    transform_mesh(box_glb, out, mirror_axis="x")
    report = analyze_mesh(out)
    # Winding must be fixed after the negative-determinant transform:
    # a correctly wound watertight mesh reports positive volume.
    assert report.is_watertight is True
    assert report.volume is not None and report.volume > 0
    assert report.volume == pytest.approx(6.0, rel=1e-5)


def test_transform_requires_an_operation(box_glb: Path) -> None:
    with pytest.raises(InvalidRequestError, match="at least one"):
        transform_mesh(box_glb)


def test_transform_composition_order_scale_then_translate(box_glb: Path, tmp_path: Path) -> None:
    # If translate were applied before scale, the centroid would land at 20.
    out = tmp_path / "box_ts.glb"
    transform_mesh(box_glb, out, scale=2.0, translate=[10.0, 0.0, 0.0])
    report = analyze_mesh(out)
    assert report.centroid[0] == pytest.approx(10.0, abs=1e-5)


def test_compose_scene_two_parts(box_glb: Path, sphere_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "scene.glb"
    result = compose_scene(
        [
            {"path": str(box_glb), "name": "box", "translate": [-2.0, 0.0, 0.0]},
            {"path": str(sphere_glb), "name": "ball", "translate": [2.0, 0.0, 0.0], "scale": 2.0},
        ],
        out,
    )
    assert result["part_count"] == 2
    assert {p["name"] for p in result["parts"]} == {"box", "ball"}
    report = analyze_mesh(out)
    assert report.connected_components == 2
    # Parts sit 4 units apart, so X extent must exceed either part alone.
    assert report.extents[0] > 4.0


def test_compose_scene_duplicate_names_are_disambiguated(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "scene_dup.glb"
    result = compose_scene(
        [{"path": str(box_glb)}, {"path": str(box_glb), "translate": [3.0, 0.0, 0.0]}],
        out,
    )
    names = [p["name"] for p in result["parts"]]
    assert len(set(names)) == 2


def test_compose_scene_adversarial_name_collision_drops_no_part(box_glb: Path, tmp_path: Path) -> None:
    # ["box", "box_2", "box"] used to assign the third part the name "box_2",
    # colliding with the explicit second part and silently orphaning its
    # geometry in the exported file (adversarial finding, executed repro).
    out = tmp_path / "scene_collision.glb"
    result = compose_scene(
        [
            {"path": str(box_glb), "name": "box"},
            {"path": str(box_glb), "name": "box_2", "translate": [5.0, 0.0, 0.0]},
            {"path": str(box_glb), "name": "box", "translate": [10.0, 0.0, 0.0]},
        ],
        out,
    )
    names = [p["name"] for p in result["parts"]]
    assert len(set(names)) == 3, names
    report = analyze_mesh(out)
    # All three boxes must be INSTANCED in the scene graph, not just present
    # as orphaned geometry: 3 boxes x 12 faces.
    assert report.face_count == 36
    assert report.connected_components == 3


def test_compose_scene_rejects_unknown_keys(box_glb: Path, tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="unsupported keys"):
        compose_scene([{"path": str(box_glb), "colour": "red"}], tmp_path / "s.glb")


def test_compose_scene_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="at least one part"):
        compose_scene([], tmp_path / "s.glb")


def test_convert_glb_to_obj_and_stl(box_glb: Path, tmp_path: Path) -> None:
    obj_out = tmp_path / "box.obj"
    result = convert_mesh(box_glb, obj_out)
    assert obj_out.exists() and result["output_format"] == "obj"
    stl_out = tmp_path / "box.stl"
    result = convert_mesh(box_glb, stl_out)
    assert stl_out.exists() and result["output_format"] == "stl"
    # Geometry preserved through conversion
    report = analyze_mesh(stl_out)
    assert report.volume == pytest.approx(6.0, rel=1e-5)


def test_convert_rejects_unsupported_target(box_glb: Path, tmp_path: Path) -> None:
    with pytest.raises(InvalidRequestError, match="Unsupported export format"):
        convert_mesh(box_glb, tmp_path / "box.usdz")


def test_repair_welds_duplicate_vertices(tmp_path: Path) -> None:
    # Build a box with deliberately un-welded (duplicated) vertices.
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    exploded = trimesh.Trimesh(
        vertices=box.triangles.reshape(-1, 3),
        faces=[[i * 3, i * 3 + 1, i * 3 + 2] for i in range(len(box.faces))],
        process=False,
    )
    src = tmp_path / "exploded.ply"
    exploded.export(str(src))
    out = tmp_path / "welded.ply"
    result = repair_mesh(src, out)
    assert result["before"]["vertex_count"] == 36
    assert result["after"]["vertex_count"] == 8
    assert result["after"]["is_watertight"] is True


def test_repair_preserves_scene_structure_and_materials(tmp_path: Path) -> None:
    # A two-part scene must stay a two-part scene through repair — flattening
    # (and dropping named materials) is not "conservative cleanup"
    # (adversarial finding).
    from PIL import Image

    def _textured_box(name: str, offset: float):
        mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        mesh.apply_translation([offset, 0, 0])
        uv = [[0.0, 0.0]] * len(mesh.vertices)
        material = trimesh.visual.material.SimpleMaterial(
            image=Image.new("RGB", (4, 4), "#ff0000"), name=name
        )
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
        return mesh

    scene = trimesh.Scene()
    scene.add_geometry(_textured_box("m1", 0.0), node_name="a", geom_name="a")
    scene.add_geometry(_textured_box("m2", 3.0), node_name="b", geom_name="b")
    src = tmp_path / "two_parts.glb"
    scene.export(str(src))

    out = tmp_path / "two_parts_repaired.glb"
    result = repair_mesh(src, out)
    assert result["scene"] is True
    assert result["part_count"] == 2

    reloaded = trimesh.load(str(out), process=False)
    assert isinstance(reloaded, trimesh.Scene)
    assert len(reloaded.geometry) == 2


def test_transform_refuses_point_cloud_only_scene(tmp_path: Path) -> None:
    # analyze refuses face-less content; write operations must match instead
    # of writing an empty mesh and reporting success (adversarial finding).
    cloud = trimesh.points.PointCloud([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    scene = trimesh.Scene()
    scene.add_geometry(cloud, node_name="pts", geom_name="pts")
    src = tmp_path / "points.glb"
    scene.export(str(src))

    with pytest.raises(InvalidRequestError, match="no triangle geometry"):
        transform_mesh(src, tmp_path / "points_out.glb", scale=2.0)


def test_analyze_components_none_when_graph_engine_missing(box_glb: Path, monkeypatch) -> None:
    # Without scipy/networkx trimesh's split() raises; the report must say
    # "unknown", never a fabricated count (adversarial finding: bare
    # `except: components = 1` put a wrong fact into LLM contexts).
    import trimesh as trimesh_module

    def _no_split(self, only_watertight=False):
        raise ImportError("no graph engines available!")

    monkeypatch.setattr(trimesh_module.Trimesh, "split", _no_split)
    report = analyze_mesh(box_glb)
    assert report.connected_components is None
    assert "unknown" in report.summary()


def test_transform_output_defaults_to_input_overwrite(box_glb: Path) -> None:
    before = analyze_mesh(box_glb)
    transform_mesh(box_glb, scale=3.0)
    after = analyze_mesh(box_glb)
    assert after.extents[0] == pytest.approx(before.extents[0] * 3.0, rel=1e-5)


def test_render_preview_stamps_gltf_frame_marker(box_glb: Path, tmp_path: Path, monkeypatch) -> None:
    # glTF inputs must reach the shared renderer with the export-frame marker
    # so its canonical-frame camera math un-rotates them (composed scenes lose
    # per-part metadata through Scene.to_mesh(), which rendered scenes lying
    # on their sides before this was stamped from the file suffix).
    from abstract3d import mesh_ops

    seen = {}

    def _fake_render(mesh, **kwargs):
        seen["marker"] = dict(getattr(mesh, "metadata", {}) or {}).get("abstract3d_export_frame")
        from PIL import Image

        return [Image.new("RGB", (8, 8), "#ffffff")]

    monkeypatch.setattr("abstract3d.rendering.render_mesh_views", _fake_render)
    mesh_ops.render_preview(box_glb, tmp_path / "p.png", size=8)
    assert seen["marker"] == "gltf_yup_front_pz"
