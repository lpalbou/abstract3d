from __future__ import annotations

import json
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")

from abstract3d import tools as a3d_tools


@pytest.fixture()
def box_glb(tmp_path: Path) -> Path:
    mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
    path = tmp_path / "box.glb"
    mesh.export(str(path))
    return path


def _parse(payload: str) -> dict:
    out = json.loads(payload)
    assert isinstance(out, dict)
    assert "success" in out
    return out


def test_tools_module_import_is_light() -> None:
    # Importing the tools module must not drag model runtimes or mesh deps in.
    # Run in a clean subprocess: this test process may already carry torch.
    import subprocess
    import sys

    code = (
        "import sys, json; import abstract3d.tools; "
        "heavy = [m for m in ('torch', 'trimesh', 'numpy', 'PIL', 'matplotlib', 'scipy') "
        "if m in sys.modules]; print(json.dumps(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip()) == []

    fns = a3d_tools.get_tool_functions()
    assert len(fns) == 8
    assert all(callable(f) for f in fns)


def test_tool_definitions_attached_when_abstractcore_present() -> None:
    pytest.importorskip("abstractcore")
    for fn in a3d_tools.get_tool_functions():
        tool_def = getattr(fn, "_tool_definition", None)
        assert tool_def is not None, f"{fn.__name__} lost its ToolDefinition"
        assert tool_def.name == fn.__name__
        assert tool_def.description
        # Tool schemas must be JSON-serializable for native tool declaration.
        json.dumps(tool_def.parameters)


def test_analyze_tool_success_and_failure(box_glb: Path, tmp_path: Path) -> None:
    ok = _parse(a3d_tools.analyze_3d_object(str(box_glb)))
    assert ok["success"] is True
    assert ok["report"]["face_count"] == 12
    assert "summary" in ok

    bad = _parse(a3d_tools.analyze_3d_object(str(tmp_path / "missing.glb")))
    assert bad["success"] is False
    assert "not found" in bad["error"]


def test_transform_tool_roundtrip(box_glb: Path, tmp_path: Path) -> None:
    out = tmp_path / "big.glb"
    result = _parse(
        a3d_tools.transform_3d_object(
            str(box_glb), str(out), scale=2.0, rotate_deg=[0, 0, 0]
        )
    )
    assert result["success"] is True
    assert Path(result["output_path"]) == out
    check = _parse(a3d_tools.analyze_3d_object(str(out)))
    assert check["report"]["extents"][0] == pytest.approx(2.0, rel=1e-5)


def test_transform_tool_rejects_double_scale(box_glb: Path) -> None:
    result = _parse(
        a3d_tools.transform_3d_object(str(box_glb), scale=2.0, scale_xyz=[1, 2, 3])
    )
    assert result["success"] is False
    assert "not both" in result["error"]


def test_compose_tool_accepts_list_and_json_string(box_glb: Path, tmp_path: Path) -> None:
    parts = [
        {"path": str(box_glb), "translate": [-2, 0, 0]},
        {"path": str(box_glb), "translate": [2, 0, 0]},
    ]
    out1 = tmp_path / "scene1.glb"
    result = _parse(a3d_tools.compose_3d_scene(parts, str(out1)))
    assert result["success"] is True and result["part_count"] == 2

    out2 = tmp_path / "scene2.glb"
    result = _parse(a3d_tools.compose_3d_scene(json.dumps(parts), str(out2)))
    assert result["success"] is True and Path(result["output_path"]) == out2


def test_convert_and_repair_tools(box_glb: Path, tmp_path: Path) -> None:
    result = _parse(a3d_tools.convert_3d_object(str(box_glb), str(tmp_path / "box.stl")))
    assert result["success"] is True and result["output_format"] == "stl"

    result = _parse(a3d_tools.repair_3d_object(str(box_glb), str(tmp_path / "fixed.glb")))
    assert result["success"] is True
    assert result["after"]["vertex_count"] > 0


def test_list_backends_tool() -> None:
    result = _parse(a3d_tools.list_3d_backends(validated_only=True))
    assert result["success"] is True
    assert result["count"] >= 1
    assert any("triposr" in str(row).lower() for row in result["backends"])


def test_generate_tool_validates_inputs(tmp_path: Path) -> None:
    result = _parse(a3d_tools.generate_3d_object(output_dir=str(tmp_path)))
    assert result["success"] is False
    assert "prompt" in result["error"] or "image_path" in result["error"]

    result = _parse(
        a3d_tools.generate_3d_object(
            output_dir=str(tmp_path), image_path=str(tmp_path / "nope.png")
        )
    )
    assert result["success"] is False
    assert "not found" in result["error"]


def test_register_tools_into_fresh_registry() -> None:
    pytest.importorskip("abstractcore")
    from abstractcore.tools.registry import ToolRegistry

    registry = ToolRegistry()
    defs = a3d_tools.register_tools(registry)
    assert len(defs) == 8
    names = {d.name for d in defs}
    assert "generate_3d_object" in names
    assert "analyze_3d_object" in names
    listed = {d.name for d in registry.list_tools()}
    assert names <= listed


def test_register_tools_refuses_missing_registry() -> None:
    from abstract3d.errors import Abstract3DError

    with pytest.raises(Abstract3DError, match="explicit ToolRegistry"):
        a3d_tools.register_tools(None)


def test_classification_is_exhaustive_both_ways() -> None:
    # Every tool must be classified; every classification must name a tool.
    tool_names = {fn.__name__ for fn in a3d_tools.TOOL_FUNCTIONS}
    classified = set(a3d_tools.SCENE3D_TOOL_CLASSIFICATION)
    assert tool_names == classified
    for name, row in a3d_tools.SCENE3D_TOOL_CLASSIFICATION.items():
        assert set(row) == {"mutating", "remote_write_capable", "downloads_model_weights"}, name
        assert all(isinstance(v, bool) for v in row.values()), name
    # Read-only tools must never be classified mutating.
    assert a3d_tools.SCENE3D_TOOL_CLASSIFICATION["analyze_3d_object"]["mutating"] is False
    assert a3d_tools.SCENE3D_TOOL_CLASSIFICATION["list_3d_backends"]["mutating"] is False


def test_tool_specs_are_json_safe_and_annotated() -> None:
    pytest.importorskip("abstractcore")
    specs = a3d_tools.abstract3d_tool_specs()
    assert len(specs) == 8
    json.dumps(specs)
    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["generate_3d_object"]["mutating"] is True
    assert by_name["analyze_3d_object"]["mutating"] is False
    assert all("description" in spec and spec["description"] for spec in specs)


def test_aligned_accessors_exist() -> None:
    # `abstract3d_*` is the ruled accessor naming (core, commons 2026-07-19).
    assert a3d_tools.abstract3d_tools() == list(a3d_tools.TOOL_FUNCTIONS)
    assert a3d_tools.get_tool_functions() == a3d_tools.abstract3d_tools()
    defs = a3d_tools.abstract3d_tool_definitions()
    assert [d.name for d in defs] == [fn.__name__ for fn in a3d_tools.TOOL_FUNCTIONS]


def test_string_booleans_never_invert_intent(box_glb: Path, tmp_path: Path) -> None:
    # Tool-call wire formats can deliver "false" as a string; Python
    # truthiness would invert it (framework coercion-risk class 2026-02-20).
    out = tmp_path / "b.glb"
    result = _parse(
        a3d_tools.transform_3d_object(str(box_glb), str(out), scale=2.0, center="false")
    )
    assert result["success"] is True
    assert result["applied"]["center"] is False

    result = _parse(a3d_tools.repair_3d_object(str(box_glb), str(tmp_path / "r.glb"), fill_holes="false"))
    assert result["success"] is True
    assert result["filled_holes"] is False

    # Unrecognized strings fail loudly instead of guessing.
    result = _parse(
        a3d_tools.transform_3d_object(str(box_glb), str(tmp_path / "c.glb"), scale=2.0, center="maybe")
    )
    assert result["success"] is False
    assert "center" in result["error"]


def test_vector_strings_are_rejected_loudly(box_glb: Path, tmp_path: Path) -> None:
    # "123" must not silently parse as [1.0, 2.0, 3.0] (string iteration).
    result = _parse(
        a3d_tools.transform_3d_object(str(box_glb), str(tmp_path / "v.glb"), translate="123")
    )
    assert result["success"] is False
    assert "3-vector" in result["error"]


def test_corrupt_file_error_is_actionable(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.glb"
    bad.write_bytes(b"glTF garbage that is not a real file")
    result = _parse(a3d_tools.analyze_3d_object(str(bad)))
    assert result["success"] is False
    assert "corrupt.glb" in result["error"]
    assert "could not be parsed" in result["error"]
