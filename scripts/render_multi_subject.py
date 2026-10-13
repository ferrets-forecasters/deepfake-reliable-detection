from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from demo import build_pipeline
from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT
from deepfake_detection.inference.render import render_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw the per-face verdicts and the speaker timeline of a video as one image.")
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--time", type=float, default=1.0, help="second of the frame to draw")
    parser.add_argument("--name", default="demo", help="dataset name used in the output folder")
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--detector-fps", type=float, default=8.0)
    parser.add_argument("--no-diarization", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    pipeline = build_pipeline(args)
    out_dir = load_config(args.config).paths.predictions_dir(f"{args.model}_on_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in map(Path, args.videos):
        result = pipeline.run(path)
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
        ok, frame = cap.read()
        cap.release()
        image = render_result(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), result, args.time)
        out = out_dir / f"{path.stem}.png"
        cv2.imwrite(str(out), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(out)


if __name__ == "__main__":
    main()
