from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from demo import build_pipeline
from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.inference.render import render_result
from deepfake_detection.utils.tools import find_tool
from deepfake_detection.xai.overlay import overlay_cam_on_box

SEGMENT_SECONDS = 2.0
TUBELET = 2
SAMPLED_FRAMES = 16


def heat_for_frame(cam: np.ndarray, frame: int, segment_start: int, segment_end: int) -> np.ndarray:
    """The CAM step that covers a frame: the 16 sampled frames of a segment span it evenly and
    each time step covers two of them."""
    position = (frame - segment_start) / max(segment_end - segment_start, 1) * (SAMPLED_FRAMES - 1)
    step = min(int(position) // TUBELET, len(cam) - 1)
    peak = float(cam.max())
    return cam[step] / peak if peak > 0 else cam[step]


def main() -> None:
    parser = argparse.ArgumentParser(description="Video of the per-face Grad-CAM: the heat map of each face's VideoMAE tokens drawn on the face box, with the verdict boxes and the speaker timeline.")
    parser.add_argument("video")
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--name", default="demo")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--detector-fps", type=float, default=8.0)
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_pipeline(args)
    path = Path(args.video)
    result = pipeline.run(path, explain=True)
    if not result.subjects:
        raise SystemExit("no face found in a full 2-second segment")

    out_dir = config.paths.xai_dir(f"gradcam_{args.model}_on_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    silent = _project_root() / "temp" / f"{path.stem}_silent.mp4"
    final = out_dir / f"{path.stem}.mp4"

    cap = cv2.VideoCapture(str(path))
    fps = result.fps
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None
    for frame_index in tqdm(range(total), desc="frames"):
        ok, bgr = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        segment = int((frame_index / fps) // SEGMENT_SECONDS)
        start, end = int(round(segment * SEGMENT_SECONDS * fps)), int(round((segment + 1) * SEGMENT_SECONDS * fps)) - 1
        for track_id, per_segment in result.cams.items():
            if segment in per_segment:
                box = result.tracks[track_id].boxes_for_indices([frame_index])[frame_index]
                frame = overlay_cam_on_box(frame, box, heat_for_frame(per_segment[segment], frame_index, start, end))
        image = cv2.cvtColor(render_result(frame, result, frame_index / fps), cv2.COLOR_RGB2BGR)
        if writer is None:
            writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image.shape[1], image.shape[0]))
        writer.write(image)
    writer.release()
    cap.release()

    subprocess.run([
        str(find_tool("ffmpeg")), "-y", "-loglevel", "error", "-i", str(silent), "-i", str(path), "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(final),
    ], check=True)
    silent.unlink()
    print(final)


if __name__ == "__main__":
    main()
