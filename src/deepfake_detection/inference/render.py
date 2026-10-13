from __future__ import annotations

import cv2
import numpy as np

from deepfake_detection.inference.multi_subject import VideoResult
from deepfake_detection.inference.summary import person_names

FAKE_COLOR = (200, 70, 60)    # RGB
REAL_COLOR = (70, 110, 150)
IDLE_COLOR = (150, 150, 150)
QUIET_COLOR = (215, 215, 215)
FONT = cv2.FONT_HERSHEY_SIMPLEX



def _text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color=(30, 30, 30), thickness: int = 1) -> None:
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def waveform_columns(waveform: np.ndarray | None, columns: int) -> np.ndarray:
    """Peak amplitude per pixel column in [0, 1], scaled by the loud end of the clip."""
    if waveform is None or len(waveform) == 0:
        return np.full(columns, 0.5)
    edges = np.linspace(0, len(waveform), columns + 1).astype(int)
    peaks = np.array([np.abs(waveform[a:max(b, a + 1)]).max() for a, b in zip(edges[:-1], edges[1:])])
    return np.clip(peaks / max(float(np.percentile(np.abs(waveform), 99.5)), 1e-6), 0.0, 1.0)


def render_result(frame_rgb: np.ndarray, result: VideoResult, time_seconds: float) -> np.ndarray:
    """The frame with one box and verdict per person, over one waveform row per speaker.
    Returns an RGB image. A person's box and voice are red when flagged and blue otherwise."""
    height, width = frame_rgb.shape[:2]
    scale = max(width / 900, 0.5)
    names = person_names(result.subjects)
    image = frame_rgb.copy()
    line = max(int(round(2 * scale)), 1)
    for subject in result.subjects:
        if subject.box is None:
            continue
        color = FAKE_COLOR if subject.flagged else REAL_COLOR
        x0, y0, x1, y1 = (int(v) for v in subject.box)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, line)
        label = f"{names[subject.track_id]}  p_fake {subject.p_fake:.2f}  {'FAKE' if subject.flagged else 'real'}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6 * scale, line)
        top = max(y0 - th - 10, 0)
        cv2.rectangle(image, (x0, top), (x0 + tw + 8, top + th + 8), color, -1)
        _text(image, label, (x0 + 4, top + th + 2), 0.6 * scale, (255, 255, 255), line)

    speakers = sorted({t.speaker for t in result.turns}) or ["audio"]
    row = int(46 * scale)
    pad = int(14 * scale)
    panel_h = pad * 2 + row * len(speakers) + int(18 * scale)
    panel = np.full((panel_h, width, 3), 255, dtype=np.uint8)
    left, right = int(110 * scale), width - pad
    columns = right - left
    span = max(result.duration, 1e-6)
    amplitude = waveform_columns(result.waveform, columns)
    column_time = (np.arange(columns) + 0.5) / columns * span
    owner_of = {s.speaker: s for s in result.subjects if s.speaker}
    for index, speaker in enumerate(speakers):
        top = pad + index * row
        mid, half = top + row // 2, row // 2 - 3
        owner = owner_of.get(speaker)
        color = IDLE_COLOR if owner is None else FAKE_COLOR if owner.flagged else REAL_COLOR
        name = names[owner.track_id] if owner else "Audio" if len(speakers) == 1 else "Other voice"
        _text(panel, name, (pad, mid + int(5 * scale)), 0.5 * scale)
        active = np.zeros(columns, dtype=bool)
        for turn in result.turns:
            if turn.speaker == speaker:
                active |= (column_time >= turn.start) & (column_time < turn.end)
        if not result.turns:
            active[:] = True
        for x in range(columns):
            reach = max(int(amplitude[x] * half), 1)
            cv2.line(panel, (left + x, mid - reach), (left + x, mid + reach), (color if active[x] else QUIET_COLOR), 1)
    marker = left + int(columns * min(time_seconds / span, 1.0))
    cv2.line(panel, (marker, pad - 4), (marker, panel_h - int(18 * scale)), (30, 30, 30), line)
    for second in range(int(span) + 1):
        x = left + int(columns * second / span)
        _text(panel, f"{second}s", (x - 6, panel_h - 4), 0.4 * scale, (90, 90, 90))

    verdict = "FAKE" if result.video_p_fake >= result.threshold else "real"
    title = np.full((int(34 * scale), width, 3), 255, dtype=np.uint8)
    _text(title, f"video: {verdict}   max p_fake {result.video_p_fake:.2f}   threshold {result.threshold}", (pad, int(24 * scale)), 0.65 * scale, FAKE_COLOR if verdict == "FAKE" else REAL_COLOR, line)
    return np.vstack([title, image, panel])
