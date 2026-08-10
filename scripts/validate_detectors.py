#!/usr/bin/env python3
"""Two-way validation harness for the defect detectors (red-team contract).

A detector that cannot reproduce the operator's verdict on the named case is
a failure; a detector validated only one way is worthless. This harness
re-runs every detector on its known-POSITIVE and known-NEGATIVE models from
the 2026-07-21 operator review and exits non-zero on any violation, so a
threshold/code change that breaks a validation shows up as a red exit code,
not as a silently shifted number.

Cases (operator verdicts, out/laurent-bust-redo):
  open_mouth        FAIL e18_windowed          PASS e20_fixed_views
  double_mouth      FAIL e2_2mv_explicit_refs  PASS e20_fixed_views + all 8
  mouth_striation   FAIL e15_cleanfront, e17_clean4   PASS e20_fixed_views
  face_inflation    FAIL e17_clean4            PASS e20_fixed_views
  profile_ripples   FAIL e17_clean4            PASS e20_fixed_views
  side_smear        FAIL e10 (left -60/-90)    PASS e20_rebake_fixed (right)
  skin_on_crown     FAIL e10 (az-30)           PASS e20_rebake_fixed
  ghost_glasses     FAIL e21_single_refs       PASS e20_rebake_fixed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from defect_detectors import geometry_report, texture_report  # noqa: E402

ROOT = Path("out/laurent-bust-redo")
TROOT = ROOT / "review" / "turntable"


def main() -> int:
    failures = []

    def check(name: str, condition: bool, evidence: str) -> None:
        status = "ok  " if condition else "FAIL"
        print(f"{status} {name}: {evidence}")
        if not condition:
            failures.append(name)

    geo = {}
    for model in ("e2_2mv_explicit_refs", "e10_2mv_registered_refs", "e15_cleanfront",
                  "e17_clean4", "e18_windowed", "e20_fixed_views"):
        geo[model] = geometry_report(ROOT / model / "scene.glb")

    om_pos = geo["e18_windowed"]["open_mouth"]
    om_neg = geo["e20_fixed_views"]["open_mouth"]
    check("open_mouth fires on e18", om_pos["fired"], f"value {om_pos['value']} >= {om_pos['threshold']}")
    check("open_mouth silent on e20", not om_neg["fired"], f"value {om_neg['value']} < {om_neg['threshold']}")

    dm_pos = geo["e2_2mv_explicit_refs"]["double_mouth"]
    dm_neg = geo["e20_fixed_views"]["double_mouth"]
    check("double_mouth fires on e2", dm_pos["fired"], f"seams {dm_pos['value']} >= {dm_pos['threshold']}")
    check("double_mouth silent on e20", not dm_neg["fired"], f"seams {dm_neg['value']}")

    for model in ("e15_cleanfront", "e17_clean4"):
        ms = geo[model]["mouth_striation"]
        check(f"mouth_striation fires on {model}", ms["fired"], f"ridges {ms['value']} >= {ms['threshold']}")
    ms_neg = geo["e20_fixed_views"]["mouth_striation"]
    check("mouth_striation silent on e20", not ms_neg["fired"], f"ridges {ms_neg['value']}")

    fi_pos = geo["e17_clean4"]["face_inflation"]
    fi_neg = geo["e20_fixed_views"]["face_inflation"]
    check("face_inflation fires on e17", fi_pos["fired"], f"value {fi_pos['value']} >= {fi_pos['threshold']}")
    check("face_inflation silent on e20", not fi_neg["fired"], f"value {fi_neg['value']}")

    pr_pos = geo["e17_clean4"]["profile_ripples"]
    pr_neg = geo["e20_fixed_views"]["profile_ripples"]
    check("profile_ripples fires on e17", pr_pos["fired"], f"count {pr_pos['value']} >= {pr_pos['threshold']}")
    check("profile_ripples silent on e20", not pr_neg["fired"], f"count {pr_neg['value']}")

    tex = {}
    for model in ("e10_2mv_registered_refs", "e20_rebake_fixed", "e21_single_refs"):
        tex[model] = texture_report(
            TROOT / model, geo.get(model, geometry_report(ROOT / model / "scene.glb"))["nose_row_frac"],
            ROOT / model / "scene.glb",
        )

    smear_pos = tex["e10_2mv_registered_refs"]["side_smear"]
    left_fired = smear_pos["az-60"]["fired"] or smear_pos["az-90"]["fired"]
    check("side_smear fires on e10 LEFT", left_fired,
          f"az-60 {smear_pos['az-60']['value']} / az-90 {smear_pos['az-90']['value']}")
    smear_neg = tex["e20_rebake_fixed"]["side_smear"]
    right_clean = not any(smear_neg[k]["fired"] for k in ("az+30", "az+60", "az+90"))
    check("side_smear silent on e20_rebake RIGHT", right_clean,
          f"az+30 {smear_neg['az+30']['value']} / az+60 {smear_neg['az+60']['value']} / az+90 {smear_neg['az+90']['value']}")

    crown_pos = tex["e10_2mv_registered_refs"]["skin_on_crown"]["az-30"]
    crown_neg = max(tex["e20_rebake_fixed"]["skin_on_crown"].values(), key=lambda d: d["value"])
    check("skin_on_crown fires on e10 az-30", crown_pos["fired"], f"value {crown_pos['value']}")
    check("skin_on_crown silent on e20_rebake", not crown_neg["fired"], f"worst value {crown_neg['value']}")

    gg_pos = tex["e21_single_refs"]["ghost_glasses"]
    gg_neg = tex["e20_rebake_fixed"]["ghost_glasses"]
    check("ghost_glasses fires on e21", gg_pos["fired"], f"score {gg_pos['value']} >= {gg_pos['threshold']}")
    check("ghost_glasses silent on e20_rebake", not gg_neg["fired"], f"score {gg_neg['value']}")

    print()
    if failures:
        print(f"VALIDATION FAILED ({len(failures)}): " + ", ".join(failures))
        return 1
    print("All detector validations hold (both ways).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
