from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

FFPP_FAKE = re.compile(r"^(deepfakes|face2face|faceswap|neuraltextures|faceshifter)_(\d{3})_(\d{3})$")
POLYGLOT_FAKE = re.compile(r"^fake_([a-z]{2})_(\d+)_to_")
CELEB_FAKE = re.compile(r"^id(\d+)_id(\d+)_\d+$")
HIDF_FAKE = re.compile(r"^(c\d+)_(c\d+)$")
FAKEAV_DONOR = re.compile(r"_(id\d+)_")


def identity_stems(fakeavceleb_raw_root: Path) -> dict[str, set[str]]:
    """FakeAVCeleb identity folder to the clip numbers of its real videos."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in (fakeavceleb_raw_root / "RealVideo-RealAudio").glob("*/*/*/*.mp4"):
        out[path.parts[-2]].add(path.stem)
    return out


def partner_keys(dataset: str, name: str, stems_of_identity: dict[str, set[str]] | None = None) -> list[tuple[str, str]]:
    """What a fake was built from or with, besides the clip its own subject key follows: (kind, subject key or sample name)
    of the real video or identity that supplied the source video or the swapped-in face."""
    if dataset == "polyglotfake" and (m := POLYGLOT_FAKE.match(name)):
        return [("source real", f"{m[1]}_{m[2]}")]
    if dataset == "ffpp" and (m := FFPP_FAKE.match(name)):
        return [("donor identity", m[3])]
    if dataset == "celebdfv2" and (m := CELEB_FAKE.match(name)):
        return [("donor identity", f"celeb_id{m[2]}")]
    if dataset == "hidf" and (m := HIDF_FAKE.match(name)):
        return [("second identity", m[2])]
    if dataset == "fakeavceleb" and stems_of_identity and (m := FAKEAV_DONOR.search(name)):
        return [("donor identity", f"fac_{stem}") for stem in sorted(stems_of_identity.get(m[1], ()))]
    return []
