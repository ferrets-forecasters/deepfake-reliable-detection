from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from deepfake_detection.data.face_crop import FaceLandmarkDetector


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"frame {index} of {path} could not be read")
    return frame


def square_crop(frame: np.ndarray, box: tuple[float, float, float, float], scale: float) -> np.ndarray:
    """A square around the box centre, scale times the longer side, kept inside the frame."""
    height, width = frame.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(min(scale * max(box[2] - box[0], box[3] - box[1]), width, height))
    x0 = int(np.clip(cx - side / 2, 0, width - side))
    y0 = int(np.clip(cy - side / 2, 0, height - side))
    return frame[y0:y0 + side, x0:x0 + side]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write one matched frame of a real video and of its fake, cropped to the same square around the face found in the real frame, one folder per dataset.")
    parser.add_argument("--out", type=Path, required=True, help="dataset folder, for example datasets/samples/celebdfv2; real/<name>/ and fake/<name>/ are created inside")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("REAL", "FAKE"), required=True, help="paths of a real video and its fake")
    parser.add_argument("--frame", type=int, default=40)
    parser.add_argument("--crop-scale", type=float, default=2.4, help="square side as a multiple of the detected face box")
    args = parser.parse_args()

    detector = FaceLandmarkDetector(max_faces=1, margin=0.25)
    for real, fake in tqdm(args.pair, desc="pairs"):
        real_frame, fake_frame = read_frame(Path(real), args.frame), read_frame(Path(fake), args.frame)
        faces = detector.detect(cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB))
        if not faces:
            print(f"no face in {real}")
            continue
        box = faces[0].box
        for role, path, frame in (("real", Path(real), real_frame), ("fake", Path(fake), fake_frame)):
            target = args.out / role / path.stem / f"frame_{args.frame:04d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(target), square_crop(frame, box, args.crop_scale), [cv2.IMWRITE_JPEG_QUALITY, 95])


if __name__ == "__main__":
    main()
