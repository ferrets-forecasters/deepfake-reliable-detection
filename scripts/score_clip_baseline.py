from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from remote_zip import DRIVE_URL, open_remote  # noqa: E402

from deepfake_detection.config import load_config  # noqa: E402
from deepfake_detection.data.face_crop import FaceLandmarkDetector  # noqa: E402

CELEBDF_ZIP_ID = "1ycE03-UpDrrTKfem5FSewMNf5mtHL5Cw"
CELEBDF_TEST_LIST = "List_of_testing_videos.txt"
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
COLUMNS = ["video_id", "dataset", "label", "manip_type", "p_fake", "frames_scored", "frames_requested", "path"]
FAKE_FOLDERS = {"Celeb-synthesis"}


def preprocess():
    """The CLIP image processor of openai/clip-vit-large-patch14 written out: bicubic resize of the short side to 224, centre crop, CLIP mean and std."""
    return transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(224),
                               transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])


def celebdf_official_test(config, fetch: bool) -> list[dict]:
    """The 518 videos of the Celeb-DF v2 official test list. With --fetch the missing ones are read from the shared-Drive zip by ranged requests."""
    root = config.paths.datasets_root / "celebdfv2" / "raw" / "test_official"
    listing = root / CELEBDF_TEST_LIST
    archive = None
    if not listing.exists():
        if not fetch:
            raise SystemExit(f"{listing} is missing; run with --fetch to read it (17 KB) and the 518 test videos (about 0.8 GB) from the Drive zip")
        archive = open_remote(DRIVE_URL.format(id=CELEBDF_ZIP_ID))
        root.mkdir(parents=True, exist_ok=True)
        listing.write_bytes(archive.read(CELEBDF_TEST_LIST))
    members = [line.split()[-1] for line in listing.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for member in tqdm(members, desc="test videos on disk", unit="video", disable=not fetch):
        target = root / member
        if not target.exists() and fetch:
            archive = archive or open_remote(DRIVE_URL.format(id=CELEBDF_ZIP_ID))
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as dst:
                while chunk := src.read(1 << 20):
                    dst.write(chunk)
        folder = Path(member).parts[0]
        if target.exists():
            rows.append({"video_id": Path(member).stem, "dataset": "celebdfv2", "label": int(folder in FAKE_FOLDERS), "manip_type": "fake" if folder in FAKE_FOLDERS else "real", "path": target})
    return rows


def own_test(config, dataset: str) -> list[dict]:
    root = config.paths.datasets_root / dataset
    with open(root / "labels" / "test_split.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["own_test_seed42"] == "1" and r["stem_candidates"] == "1"]
    out = []
    for r in rows:
        path = root / "raw" / "test_split" / r["member"]
        if path.exists():
            out.append({"video_id": r["sample_id"], "dataset": dataset, "label": int(r["label"]), "manip_type": r["manip_type"], "path": path})
    return out


def face_crops(path: Path, detector: FaceLandmarkDetector, frames: int) -> tuple[list[Image.Image], int]:
    """Square crops of the largest face in `frames` evenly spaced frames; frames without a face are dropped."""
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    wanted = set(np.linspace(0, max(count - 1, 0), min(frames, max(count, 1))).round().astype(int).tolist())
    crops = []
    for index in range(count):
        if not cap.grab():
            break
        if index not in wanted:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect(rgb)
        if not faces:
            continue
        x0, y0, x1, y1 = max((f.box for f in faces), key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        side = int(min(max(x1 - x0, y1 - y0), rgb.shape[0], rgb.shape[1]))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        left = int(np.clip(cx - side / 2, 0, rgb.shape[1] - side))
        top = int(np.clip(cy - side / 2, 0, rgb.shape[0] - side))
        crops.append(Image.fromarray(rgb[top:top + side, left:left + side]))
    cap.release()
    return crops, len(wanted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scores test videos with the published CLIP LN-tuning detector (Yermakov et al., trained on FaceForensics++): face crops of evenly spaced frames, the mean P(fake) over the frames is the video score. Resumable, one row per video in predictions.csv.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", required=True, choices=["celebdfv2_official_test", "own_test_hidf", "own_test_fakeavceleb"])
    parser.add_argument("--weights", type=Path, default=None, help="model.torchscript, default external/sota/deepfake-detection/weights/model.torchscript")
    parser.add_argument("--download-weights", action="store_true", help="fetch model.torchscript (607 MB) from yermandy/deepfake-detection on Hugging Face when it is missing")
    parser.add_argument("--fetch", action="store_true", help="read missing Celeb-DF test videos from the Drive zip")
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="first N videos only, for a quick check")
    args = parser.parse_args()

    config = load_config(args.config)
    weights = args.weights or config.paths.datasets_root.parent / "external" / "sota" / "deepfake-detection" / "weights" / "model.torchscript"
    if not weights.exists():
        if not args.download_weights:
            raise SystemExit(f"{weights} is missing; rerun with --download-weights to fetch it (607 MB, MIT license)")
        from huggingface_hub import hf_hub_download

        weights.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id="yermandy/deepfake-detection", filename="model.torchscript", local_dir=str(weights.parent))

    videos = celebdf_official_test(config, args.fetch) if args.scope == "celebdfv2_official_test" else own_test(config, args.scope.removeprefix("own_test_"))
    if args.limit:
        videos = videos[: args.limit]
    out_dir = config.paths.predictions_dir(f"clip_ln_tuning_on_{args.scope}")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "predictions.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as f:
            done = {r["video_id"] for r in csv.DictReader(f)}
    todo = [v for v in videos if v["video_id"] not in done]
    log = open(out_dir / "progress.log", "a", encoding="utf-8")
    log.write(f"{time.strftime('%F %T')} {len(videos)} videos, {len(todo)} to score\n")
    if not todo:
        print(f"nothing to do: {len(done)} of {len(videos)} videos scored")
        return

    device, dtype = "cuda:0", torch.bfloat16
    model = torch.jit.load(str(weights), map_location=device).to(device).to(dtype).eval()
    transform, detector = preprocess(), FaceLandmarkDetector(max_faces=5, margin=0.25)
    new_file = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for video in tqdm(todo, desc=args.scope, unit="video"):
            crops, requested = face_crops(video["path"], detector, args.frames)
            p_fake = ""
            if crops:
                batch = torch.stack([transform(c) for c in crops]).to(device).to(dtype)
                with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
                    p_fake = float(model(batch).float().softmax(dim=1)[:, 1].mean())
            writer.writerow({"video_id": video["video_id"], "dataset": video["dataset"], "label": video["label"], "manip_type": video["manip_type"],
                             "p_fake": p_fake if p_fake == "" else round(p_fake, 6), "frames_scored": len(crops), "frames_requested": requested, "path": str(video["path"])})
            f.flush()
            if not crops:
                log.write(f"{time.strftime('%T')} no face in {video['video_id']}\n")
                log.flush()


if __name__ == "__main__":
    main()
