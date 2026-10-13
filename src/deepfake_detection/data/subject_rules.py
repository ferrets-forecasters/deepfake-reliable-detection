from __future__ import annotations

from typing import Callable


def _base(sample_id: str) -> str:
    return sample_id.rsplit("_seg", 1)[0]


def _polyglotfake(sample_id: str) -> str:
    # real_ar_10 / fake_ar_10_to_en_MicroTts: the numeric speaker id is shared across languages
    parts = _base(sample_id).split("_")
    return parts[2] if len(parts) >= 3 else sample_id


def _hidf(sample_id: str) -> str:
    # c00003 (real) / c00003_c11507 (fake): the first component is the target identity
    return _base(sample_id).split("_")[0]


def _celebdfv2(sample_id: str) -> str:
    # id0_0000 (real), 00000 (YouTube real), id0_id16_0000 (fake: first id is the target face)
    base = _base(sample_id)
    first = base.split("_")[0]
    if first.startswith("id"):
        return f"celeb_{first}"
    return f"celeb_yt_{base}"


def _fakeavceleb(sample_id: str) -> str:
    # the source clip number is shared by a real clip and all its fake derivatives
    return f"fac_{_base(sample_id).split('_')[0]}"


def _ffpp(sample_id: str) -> str:
    # 000 (real), deepfakes_000_003 (fake, source is the first number), deepfakedetection_01_02__... (actor ids)
    base = _base(sample_id)
    parts = base.split("_")
    if parts[0].isdigit():
        return f"ffpp_{parts[0]}"
    numbers = [p for p in parts[1:] if p.isdigit()]
    if numbers:
        source = numbers[0]
        return f"ffpp_dfd_{source}" if len(source) <= 2 else f"ffpp_{source}"
    return f"ffpp_unk_{base[:10]}"


SUBJECT_RULES: dict[str, Callable[[str], str]] = {
    "polyglotfake": _polyglotfake,
    "hidf": _hidf,
    "celebdfv2": _celebdfv2,
    "fakeavceleb": _fakeavceleb,
    "ffpp": _ffpp,
}


def subject_id(rule: str, sample_id: str) -> str:
    if rule not in SUBJECT_RULES:
        raise KeyError(f"unknown subject rule '{rule}', known: {sorted(SUBJECT_RULES)}")
    return SUBJECT_RULES[rule](sample_id)
