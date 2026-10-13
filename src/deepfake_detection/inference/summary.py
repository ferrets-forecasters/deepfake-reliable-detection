from __future__ import annotations

from deepfake_detection.inference.multi_subject import VideoResult


def person_names(subjects) -> dict[int, str]:
    """Person A, Person B, ... by position in the frame, left to right."""
    ordered = sorted(subjects, key=lambda s: s.center_x)
    return {s.track_id: f"Person {chr(ord('A') + i)}" for i, s in enumerate(ordered)}


def format_result(result: VideoResult, source: str) -> str:
    """A text table with one row per person and the video verdict underneath."""
    if not result.subjects:
        return f"{source}: no face found in a full 2-second segment"
    names = person_names(result.subjects)
    rows = [("person", "speaker", "segments", "p_fake", "alpha", "verdict")]
    for subject in sorted(result.subjects, key=lambda s: s.center_x):
        rows.append((
            names[subject.track_id], subject.speaker or "-", str(len(subject.segments)), f"{subject.p_fake:.4f}",
            "-" if subject.alpha is None else f"{subject.alpha:.3f}", "FAKE" if subject.flagged else "real",
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)) for row in rows]
    verdict = "FAKE" if result.video_p_fake >= result.threshold else "real"
    return "\n".join([source, *lines, f"video: {verdict} (p_fake {result.video_p_fake:.4f}, threshold {result.threshold})"])


class RollingVerdict:
    """Mean p_fake over the last chunks of a live feed, per face position (faces ordered left to right)."""

    def __init__(self, window: int = 5):
        self.window = window
        self.history: list[list[float]] = []

    def update(self, result: VideoResult) -> list[float]:
        ordered = [s.p_fake for s in sorted(result.subjects, key=lambda s: s.center_x)]
        if self.history and len(self.history[-1]) != len(ordered):
            self.history.clear()
        self.history.append(ordered)
        self.history = self.history[-self.window:]
        return [sum(column) / len(column) for column in zip(*self.history)] if ordered else []
