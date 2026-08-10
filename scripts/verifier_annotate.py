#!/usr/bin/env python3
"""Compose annotated contact sheets from the adversarial verifier's defect log.

Reads the renders produced by scripts/verifier_orbit_render.py and writes
out/laurent-bust-redo/review/verifier/<model>/sheet_annotated.png — six
diagnostic views per model, each captioned with the defects visible in it,
plus a verdict title bar and a defect summary footer.

The DEFECTS dict below IS the machine-readable defect log backing
docs/research/independent_verification.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1] / "out" / "laurent-bust-redo" / "review" / "verifier"

# Per model: verdicts, six (filename, caption) evidence tiles, summary lines.
SHEETS: dict[str, dict] = {
    "e10_2mv_registered_refs": {
        "verdict": "MESH: MAJOR (deformed features)  |  TEXTURE: BLOCKER (chrome smear + ghost face)",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: pale smear at temple (image-left); red parted lips"),
            ("face_az0_clay.png", "az0 clay: flattened wide nose, statue parted lips, earpiece brick"),
            ("az270_el0_tex.png", "az270 tex: white/chrome smear over jaw+neck (BLOCKER)"),
            ("az300_el0_tex.png", "az300 tex: smear + ghost face fragment at sideburn (BLOCKER)"),
            ("face_az90_clay.png", "az90 clay: carved parted lips; kinked/floating cable"),
            ("az180_el0_tex.png", "az180 tex: tan 'cardboard' patch in hair; gray hair wash"),
        ],
        "summary": [
            "BLOCKER  white/chrome smear across jaw+neck with ghost second-face fragment (az240-330 tex)",
            "MAJOR    front: pale mottled smear at temple; flat 'pig' nose; statue parted lips (photo: closed)",
            "MAJOR    earpiece rendered as crude brick; tan cardboard patch in hair; red parted lip texture",
            "MINOR    lumpy cheeks; stepped hair shelf at crown; kinked cable; skin bleed on sleeve",
        ],
    },
    "e11_2mv_reg_hq": {
        "verdict": "MESH: BLOCKER (melted face, nose erased)  |  TEXTURE: MAJOR (best of 8, still dirty)",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: doubled translucent glasses; skin through lens bottoms"),
            ("face_az0_clay.png", "az0 clay: nose eroded to stub; V-notch glasses, nose pokes through"),
            ("face_az90_clay.png", "az90 clay: near-flat profile, lips protrude beyond nose (BLOCKER)"),
            ("face_az330_tex.png", "az330 tex: pale cheek smear, black jaw blotch, over-bright ear"),
            ("az90_el0_clay.png", "az90 clay: melted flat ear; shredded hair flakes at crown"),
            ("az180_el0_clay.png", "az180 clay: wedge hair fin at back-left crown"),
        ],
        "summary": [
            "BLOCKER  face reads melted/mannequin: nose a stub with notch, flat profile silhouette (az90 clay)",
            "MAJOR    wide slit 'grimace' mouth; melted flat left ear; shredded hair flakes; wedge fin at crown",
            "MAJOR    doubled glasses in texture (2nd translucent pair + skin through lens); cheek smear + jaw blotch",
            "MINOR    floating wire stub below jaw; pale beard streaks; dark chips along jawline",
        ],
    },
    "e15_cleanfront": {
        "verdict": "MESH: MAJOR (antenna, button nose, bumps)  |  TEXTURE: least-bad, front combo borderline BLOCKER",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: flesh brick in hair + white streak across lens (BLOCKER combo)"),
            ("face_az0_clay.png", "az0 clay: floating antenna rod at temple; square forehead patch"),
            ("az300_el0_tex.png", "az300 tex: orange/brown smudge in hair; gray-yellow shoulder blob"),
            ("face_az90_clay.png", "az90 clay: button nose, pursed duck lips; acne-like bump noise"),
            ("az0_el0_clay.png", "az0 clay: antenna rod breaks silhouette; big ears with slab"),
            ("face_az330_tex.png", "az330 tex: pale ear + beige 'mold' smudges in hair"),
        ],
        "summary": [
            "BLOCKER* front combo: flesh-colored brick floating in hair + white streak over lens (az0 tex)",
            "MAJOR    floating antenna rod above temple (in silhouette); square slab patch in forehead",
            "MAJOR    button/clown nose; pursed protruding lips; acne-like bump noise across cheeks/jaw",
            "MINOR    toga drape folds on shoulder; floating cable loop; tan shoulder smudge; neck streak",
        ],
    },
    "e17_clean4": {
        "verdict": "MESH: BLOCKER (fused visor, shredded hair)  |  TEXTURE: BLOCKER (glow smear, ghosts)",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: visor = smeared band, nostril dots painted on it"),
            ("face_az0_clay.png", "az0 clay: giant visor fused over half the face (BLOCKER)"),
            ("face_az270_tex.png", "az270 tex: white glow smear before ear + black ghost drip (BLOCKER)"),
            ("face_az270_clay.png", "az270 clay: bulbous nose fused to visor; doubled everted lips"),
            ("az90_el0_clay.png", "az90 clay: shredded tentacle hair spikes (BLOCKER silhouette)"),
            ("az240_el0_tex.png", "az240 tex: neck striping; shoulder tan patches; sash roll"),
        ],
        "summary": [
            "BLOCKER  glasses are a giant fused visor slab swallowing the upper face, merged into nose",
            "BLOCKER  hair shredded into tentacle spikes (silhouette wrong at arm's length)",
            "BLOCKER  bright white glow smear on cheek + black ghost drip; visor band with painted nostril dots",
            "MAJOR    doubled everted duck lips (parted at profile); neck striping; sash ridge across chest",
        ],
    },
    "e18_windowed": {
        "verdict": "MESH: BLOCKER (open mouth, wrong glasses)  |  TEXTURE: BLOCKER (worst of 8)",
        "tiles": [
            ("face_az0_clay.png", "az0 clay: OPEN mouth; nose pierces glasses notch; knob at hairline"),
            ("face_az0_tex.png", "az0 tex: ghost eyes+brow on forehead; glasses = butterfly blob"),
            ("az90_el0_tex.png", "az90 tex: whole-head double exposure, hair bleeding across face"),
            ("az180_el0_tex.png", "az180 tex: gray marbled shirt; smoke smear; bald crown patch"),
            ("face_az90_clay.png", "az90 clay: open lips in profile; long droopy nose"),
            ("az240_el0_tex.png", "az240 tex: chrome neck smear; white curtain patches on shoulders"),
        ],
        "summary": [
            "BLOCKER  mouth OPEN in clay + texture (photo: closed) — operator finding CONFIRMED",
            "BLOCKER  wrong glasses: thin swooping aviator visor, nose pierces the bridge gap",
            "BLOCKER  texture chaos: ghost eyes on forehead, second face on side, smoke shirt, double exposure",
            "MAJOR    knob/bun at hairline; droopy witch nose; strap fold across chest; ragged sleeve",
        ],
    },
    "e20_fixed_views": {
        "verdict": "MESH: BEST of 8 (minor crown dents)  |  TEXTURE: BLOCKER (globally displaced)",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: texture shifted up ~10-15%: beard on nose, lips at philtrum"),
            ("face_az0_clay.png", "az0 clay: correct flat-top glasses, straight nose, closed mouth"),
            ("az270_el0_tex.png", "az270 tex: cyan patch behind ear; displaced glasses band"),
            ("face_az90_tex.png", "az90 tex: features ride high; doubled glasses-arm band on temple"),
            ("az0_el20_clay.png", "az0 el20 clay: crown dents + spikes (worst mesh flaw)"),
            ("az180_el0_tex.png", "az180 tex: amber glow at nape/collar; pale crown streak"),
        ],
        "summary": [
            "BLOCKER  global upward texture shift: painted glasses in top half of lens, translucent below,",
            "         beard reaches glasses line, red lips at philtrum, brows above frame (every face view)",
            "MAJOR    cyan out-of-palette patch behind ear (az270); doubled glasses-arm band; temple streaks",
            "MINOR    crown dents/spikes; sash ridge on shoulder; weak chin transition; amber nape glow",
        ],
    },
    "e20_rebake_fixed": {
        "verdict": "MESH: = e20_fixed (byte-identical)  |  TEXTURE: BLOCKER — 'fixed' claim REFUTED",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: center registration improved BUT lens junk + cheek dark seam"),
            ("face_az330_tex.png", "az330 tex: jagged seam down cheek-jaw, red blotch, amber lens (BLOCKER)"),
            ("az270_el0_tex.png", "az270 tex: amber/teal misprojection filling lens (BLOCKER)"),
            ("az90_el0_tex.png", "az90 tex: cream wash down neck; dark crescent behind ear (BLOCKER)"),
            ("az180_el0_tex.png", "az180 tex: cream nape patch hairline-to-collar; sleeve skin blotches"),
            ("face_az0_clay.png", "az0 clay: same best-of-8 mesh as e20_fixed_views"),
        ],
        "summary": [
            "REFUTED  center face alignment IS better than e20_fixed, but new blockers were introduced:",
            "BLOCKER  jagged dark seam down right cheek/jaw (stronger than e20_fixed's); amber/teal lens garbage",
            "BLOCKER  cream/beige wash on neck below ear + large cream nape patch (out of palette)",
            "MAJOR    red cheek blotch; dark crescent behind ear; gray smoke crown; patchy beard chin gap",
        ],
    },
    "e21_single_refs": {
        "verdict": "MESH: MAJOR (wrong glasses, shelf, ear crater)  |  TEXTURE: BLOCKER (ghost eye/glasses)",
        "tiles": [
            ("face_az0_tex.png", "az0 tex: skewed trapezoid lens patches; black drip tendril at jaw"),
            ("face_az0_clay.png", "az0 clay: wrong scalloped wraparound glasses; scowl folds"),
            ("face_az300_tex.png", "az300 tex: ghost eye+brow painted above glasses arm (BLOCKER)"),
            ("face_az90_clay.png", "az90 clay: forehead shelf like cap brim; ear crater"),
            ("face_az330_tex.png", "az330 tex: pale cheek patch + jagged black seam"),
            ("az0_el0_clay.png", "az0 clay: antenna strands at crown; downturned mouth"),
        ],
        "summary": [
            "BLOCKER  ghost eye + eyebrow in hair above glasses arm; translucent lenses with eye fragments",
            "MAJOR    glasses shape wrong (thin scalloped wraparound vs flat-top slab in photo)",
            "MAJOR    forehead shelf + temple crease; left ear crater; black drip tendril down neck",
            "MINOR    skin bleed on sleeve; pale hairline streaks; beard smudge on cheek",
        ],
    },
}

TILE = 620
CAP_H = 66
COLS = 3


def _font(size: int):
    for cand in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    return ImageFont.load_default()


def compose(model: str, spec: dict) -> Path:
    mdir = ROOT / model
    tiles = spec["tiles"]
    rows = (len(tiles) + COLS - 1) // COLS
    header = 96
    footer = 44 + 34 * len(spec["summary"])
    w = COLS * TILE
    h = header + rows * (TILE + CAP_H) + footer
    canvas = Image.new("RGB", (w, h), "#14171c")
    draw = ImageDraw.Draw(canvas)

    draw.text((18, 12), f"{model} — adversarial verification", fill="#f7f6f2", font=_font(34))
    draw.text((18, 56), spec["verdict"], fill="#ffd166", font=_font(24))

    f_cap = _font(19)
    for i, (fname, caption) in enumerate(tiles):
        r, c = divmod(i, COLS)
        x0 = c * TILE
        y0 = header + r * (TILE + CAP_H)
        cell = Image.new("RGB", (TILE, TILE), "#e9e5dc")
        try:
            img = Image.open(mdir / fname).convert("RGB")
            img.thumbnail((TILE - 8, TILE - 8))
            cell.paste(img, ((TILE - img.width) // 2, (TILE - img.height) // 2))
        except Exception:
            pass
        canvas.paste(cell, (x0, y0))
        draw.rectangle((x0, y0 + TILE, x0 + TILE - 1, y0 + TILE + CAP_H - 1), fill="#22262e")
        # wrap caption to two lines if needed
        words = caption.split()
        line1, line2 = caption, ""
        if draw.textlength(caption, font=f_cap) > TILE - 20:
            for k in range(len(words) - 1, 0, -1):
                cand = " ".join(words[:k])
                if draw.textlength(cand, font=f_cap) <= TILE - 20:
                    line1, line2 = cand, " ".join(words[k:])
                    break
        draw.text((x0 + 10, y0 + TILE + 8), line1, fill="#ff8f8f" if "BLOCKER" in caption else "#dfe3ea", font=f_cap)
        if line2:
            draw.text((x0 + 10, y0 + TILE + 34), line2, fill="#ff8f8f" if "BLOCKER" in caption else "#dfe3ea", font=f_cap)

    y = header + rows * (TILE + CAP_H) + 12
    draw.text((18, y), "Defect summary (full table: docs/research/independent_verification.md)", fill="#9aa3b2", font=_font(20))
    y += 32
    f_sum = _font(21)
    for line in spec["summary"]:
        color = "#ff8f8f" if line.strip().startswith("BLOCKER") or line.strip().startswith("REFUTED") else "#dfe3ea"
        draw.text((18, y), line, fill=color, font=f_sum)
        y += 34

    out = mdir / "sheet_annotated.png"
    canvas.save(out)
    return out


def main() -> None:
    targets = sys.argv[1:] or list(SHEETS)
    for model in targets:
        out = compose(model, SHEETS[model])
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
