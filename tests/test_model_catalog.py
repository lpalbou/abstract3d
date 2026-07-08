from __future__ import annotations

from abstract3d.model_catalog import capability_model_records, catalog_rows, iter_model_specs


def test_catalog_filters_validated_records_only() -> None:
    rows = catalog_rows(validated_only=True)

    assert [row["model_id"] for row in rows] == ["stabilityai/TripoSR"]
    assert rows[0]["apple_silicon"] == "validated"
    assert rows[0]["license"] == "MIT"


def test_catalog_task_filter_matches_scene3d_tasks() -> None:
    specs = list(iter_model_specs(task="image_to_scene3d"))

    assert any(spec.model_id == "stabilityai/TripoSR" for spec in specs)
    assert any(spec.model_id == "stepfun-ai/Step1X-3D" for spec in specs)
    assert all("image_to_scene3d" in spec.tasks for spec in specs)


def test_step1x_catalog_entry_is_experimental_geometry_only_family() -> None:
    step1x = next(spec for spec in iter_model_specs(validated_only=False) if spec.model_id == "stepfun-ai/Step1X-3D")

    assert step1x.provider_id == "step1x"
    assert step1x.backend_kind == "step1x"
    assert step1x.status == "experimental"
    assert step1x.apple_silicon == "experimental"
    assert "text_to_scene3d" in step1x.tasks


def test_capability_model_records_default_to_validated_entries() -> None:
    rows = capability_model_records()

    assert [row["model_id"] for row in rows] == ["stabilityai/TripoSR"]
    assert rows[0]["formats"] == ["glb", "obj", "zip"]
    assert rows[0]["recommended"] is True
