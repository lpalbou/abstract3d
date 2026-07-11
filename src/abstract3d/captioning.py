"""Automatic subject captioning for prompt-free reference generation.

The generated-reference pipeline needs a textual subject slot, and it must
come from the pipeline itself: a human-written description is unavailable
in production and, when wrong, actively harmful (a carved wooden owl was
generated as glazed ceramic because a hand-written hint said "ceramic").
A weak-but-honest local captioner (BLIP base, ~1 GB, already used by the
ecosystem) reads the description off the source photo instead.

The caption is a WEAK PRIOR by design: downstream prompts must treat the
left-panel photo as the material authority and the caption as naming the
subject only. See reference_generation for how it is composed.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

_LOCK = threading.Lock()
_CAPTIONER: Optional[Any] = None
_FAILED: bool = False

_DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"

# Material/finish/color vocabulary is BANNED from prompt text regardless of
# where it came from (captioner or user): a wrong material claim in text
# overrides the source photo's pixels (proven: a hand-written "ceramic with
# glaze" hint turned a carved wooden owl into glazed pottery), while a
# correct one adds nothing the photo doesn't already carry. Asymmetric cost
# justifies the blunt stoplist.
_MATERIAL_STOPLIST = frozenset("""
wood wooden ceramic porcelain glazed glaze glossy matte metal metallic steel
iron bronze brass copper gold golden silver marble stone granite plastic
rubber glass crystal clay terracotta painted paint enamel lacquered polished
varnished leather fabric cloth woven knitted fur furry velvet satin silk
shiny reflective smooth rough carved
black white red green blue yellow orange purple pink brown beige cream tan
grey gray ivory teal turquoise magenta maroon navy olive
dark light pale bright deep warm cool
""".split())


def extract_subject_noun(text: Optional[str], *, max_words: int = 4) -> str:
    """Reduce any description (caption or user prompt) to a material-free
    subject noun phrase for the generation prompt's only text slot.

    "a wooden owl figurine with warm brown glaze" -> "owl figurine".
    Empty results fall back to "object" — the composite's left panel is the
    material authority, text only names the subject class.
    """

    import re

    if not text:
        return "object"
    # Function/viewpoint/composition words never name a subject; a t23d
    # prompt like "a red sports car seen from a three-quarter front angle,
    # studio photo" must reduce to "sports car", not "sports car seen from"
    # (measured: the garbled noun degraded every generation of that run).
    stop = frozenset((
        "a", "an", "the", "of", "with", "and", "on", "in", "its",
        "his", "her", "very", "photo", "photograph", "image",
        "picture", "closeup", "close", "up", "background",
        "seen", "from", "view", "viewed", "angle", "front", "side",
        "back", "top", "bottom", "three", "quarter", "profile",
        "studio", "shot", "render", "rendering",
    ))
    words = re.findall(r"[a-zA-Z]+", str(text).lower())
    kept: list = []
    for word in words:
        if word in stop:
            continue
        if word in _MATERIAL_STOPLIST:
            continue
        kept.append(word)
        if len(kept) >= int(max_words):
            break
    return " ".join(kept) if kept else "object"


def caption_image(image: Any, *, max_words: int = 16) -> Optional[str]:
    """One-line subject caption for a PIL image, or None when unavailable.

    Loads BLIP lazily and caches it for the process; every failure path
    returns None (callers fall back to their no-hint behavior) — the
    captioner must never take down a generation run.
    """

    global _CAPTIONER, _FAILED
    if _FAILED:
        return None
    try:
        with _LOCK:
            if _CAPTIONER is None:
                from transformers import AutoProcessor, BlipForConditionalGeneration

                processor = AutoProcessor.from_pretrained(_DEFAULT_MODEL)
                model = BlipForConditionalGeneration.from_pretrained(_DEFAULT_MODEL)
                model.eval()
                _CAPTIONER = (processor, model)
        processor, model = _CAPTIONER
        rgb = image.convert("RGB")
        inputs = processor(images=rgb, return_tensors="pt")
        import torch

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=int(max_words) + 8,
                # BLIP-base degenerates into token repetition on some
                # subjects ("... bra bra bra"); the n-gram ban stops it.
                no_repeat_ngram_size=2,
            )
        text = processor.decode(output[0], skip_special_tokens=True).strip()
        if not text:
            return None
        words = text.split()
        # Belt and braces: collapse any residual immediate word repeats.
        deduped = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]
        if len(deduped) > int(max_words):
            deduped = deduped[: int(max_words)]
        return " ".join(deduped)
    except Exception:
        _FAILED = True
        return None
