from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import FaceLandmarkDetector
from deepfake_detection.data.segment_embedding import WHISPER_SAMPLE_RATE

OWN_PREDICTIONS = "moe_combined_retrain_on_combined_test_seed42"
COLUMNS = ["id", "dataset", "role", "label", "manip_type", "video", "full_frame", "face_crop", "box_x0", "box_y0", "box_x1", "box_y1",
           "frame_index", "audio_window_s", "logmel", "p_fake", "alpha", "why_chosen"]
FAKEAVCELEB_FAKE = re.compile(r"^\d{5}_id\d{5}_wavtolip$")


def own_test_pairs(config, dataset: str) -> list[tuple[Path, Path]]:
    """Real and fake videos of the seed-42 test split that show the same scene: for FakeAVCeleb the lip-sync fake with
    the real clip's number in the same identity folder, for HiDF the fake whose name starts with the real clip's name."""
    root = config.paths.datasets_root / dataset
    with open(root / "labels" / "test_split.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["own_test_seed42"] == "1"]
    if dataset == "hidf":
        base = root / "raw" / "test_split"
        reals = {Path(r["member"]).stem: base / r["member"] for r in rows if r["label"] == "0"}
        pairs = []
        for r in rows:
            if r["label"] == "1":
                source = Path(r["member"]).stem.split("_")[0]
                if source in reals:
                    pairs.append((reals[source], base / r["member"]))
        return sorted(pairs)
    if dataset == "fakeavceleb":
        base = config.raw_dir("fakeavceleb") / "FakeAVCeleb_v1.2"
        members = [Path(*Path(r["member"]).parts[-5:]) for r in rows]
        fakes: dict[tuple, Path] = {}
        for m in members:
            if m.parts[0] == "FakeVideo-FakeAudio" and FAKEAVCELEB_FAKE.match(m.stem):
                fakes.setdefault((*m.parts[1:4], m.stem.split("_")[0]), m)
        return sorted((base / m, base / fakes[(*m.parts[1:4], m.stem)]) for m in members if m.parts[0] == "RealVideo-RealAudio" and (*m.parts[1:4], m.stem) in fakes)
    raise SystemExit(f"no automatic pairing for {dataset}; pass --pair")


def predictions(config) -> dict[str, tuple[float, float]]:
    path = config.paths.predictions_dir(OWN_PREDICTIONS) / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[re.sub(r"_seg\d+$", "", Path(row["video"]).stem)] = (float(row["p_fake_best_f1"]), float(row["alpha_best_f1"]))
    return out


def frame_at(path: Path, index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def log_mel(path: Path, start: float, seconds: float) -> np.ndarray | None:
    import whisper

    try:
        wave = extract_audio_waveform(path, start, start + seconds, WHISPER_SAMPLE_RATE)
    except Exception:
        return None
    if wave is None or len(wave) < WHISPER_SAMPLE_RATE:
        return None
    return whisper.log_mel_spectrogram(torch.from_numpy(np.asarray(wave, dtype=np.float32)))[:, :200].numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export matched real and fake example pairs of one dataset: the full frame and the face crop at the clip's own resolution (same frame index, same crop box taken from the real frame), the log-mel of the same 2 seconds of audio when there is audio, and a manifest with the selection rule.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--failures", type=int, default=2, help="pairs whose fake is missed or whose real is flagged at 0.5, added on purpose")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-second", type=float, default=2.0)
    parser.add_argument("--audio-start", type=float, default=1.0)
    parser.add_argument("--audio", action="store_true", help="the dataset has an audio track that carries the speech")
    parser.add_argument("--pair", nargs=2, action="append", metavar=("REAL", "FAKE"), help="explicit pairs instead of the automatic test-split pairing")
    parser.add_argument("--margin", type=float, default=1.6, help="crop side as a multiple of the face box")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = args.out or config.paths.analysis_dir("example_pairs") / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = [(Path(r).resolve(), Path(f).resolve()) for r, f in args.pair] if args.pair else own_test_pairs(config, args.dataset)
    scores = predictions(config)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    def verdict(real: Path, fake: Path) -> tuple[bool, tuple, tuple]:
        r = scores.get(real.stem, (None, None))
        f = scores.get(fake.stem) or scores.get(f"fake_{fake.stem}") or scores.get(f"{fake.parent.name.lower()}_{fake.stem}", (None, None))
        wrong = (f[0] is not None and f[0] < 0.5) or (r[0] is not None and r[0] >= 0.5)
        return wrong, r, f

    detector = FaceLandmarkDetector(max_faces=1, margin=0.25)
    chosen: list[tuple[Path, Path, str, tuple, tuple]] = []
    random_picks, failure_picks = [], []
    for real, fake in pairs:
        wrong, r_score, f_score = verdict(real, fake)
        if wrong and len(failure_picks) < args.failures:
            failure_picks.append((real, fake, "failure: fake missed or real flagged at 0.5", r_score, f_score))
        elif not wrong and len(random_picks) < args.count - args.failures:
            random_picks.append((real, fake, "seeded random draw from the test-split pairs", r_score, f_score))
        if len(random_picks) + len(failure_picks) >= args.count and len(failure_picks) >= args.failures:
            break
    chosen = random_picks + failure_picks

    rows = []
    for number, (real, fake, why, r_score, f_score) in enumerate(tqdm(chosen, desc="pairs")):
        cap = cv2.VideoCapture(str(real))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        index = int(round(args.frame_second * fps))
        real_frame, fake_frame = frame_at(real, index), frame_at(fake, index)
        faces = detector.detect(cv2.cvtColor(real_frame, cv2.COLOR_BGR2RGB)) if real_frame is not None else []
        if real_frame is None or fake_frame is None or not faces:
            print(f"skipped, no frame or no face: {real.name}")
            continue
        x0, y0, x1, y1 = faces[0].box
        side = int(min(args.margin * max(x1 - x0, y1 - y0), real_frame.shape[0], real_frame.shape[1]))
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        cx0 = int(np.clip(cx - side / 2, 0, real_frame.shape[1] - side))
        cy0 = int(np.clip(cy - side / 2, 0, real_frame.shape[0] - side))
        pair_id = f"{args.dataset}_{number:02d}"
        for role, path, frame, score in (("real", real, real_frame, r_score), ("fake", fake, fake_frame, f_score)):
            full = out_dir / f"{pair_id}_{role}_full.png"
            crop = out_dir / f"{pair_id}_{role}_face.png"
            cv2.imwrite(str(full), frame)
            cv2.imwrite(str(crop), frame[cy0:cy0 + side, cx0:cx0 + side])
            mel_path = ""
            if args.audio:
                mel = log_mel(path, args.audio_start, 2.0)
                if mel is not None:
                    mel_path = str(out_dir / f"{pair_id}_{role}_logmel.npy")
                    np.save(mel_path, mel)
            rows.append({
                "id": pair_id, "dataset": args.dataset, "role": role, "label": int(role == "fake"), "manip_type": "real" if role == "real" else "fake",
                "video": str(path), "full_frame": str(full), "face_crop": str(crop), "box_x0": int(x0), "box_y0": int(y0), "box_x1": int(x1), "box_y1": int(y1),
                "frame_index": index, "audio_window_s": f"{args.audio_start}-{args.audio_start + 2.0}" if args.audio else "",
                "logmel": mel_path, "p_fake": "" if score[0] is None else round(score[0], 4), "alpha": "" if score[1] is None else round(score[1], 4), "why_chosen": why,
            })
    with open(out_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows) // 2} pairs in {out_dir}")


if __name__ == "__main__":
    main()
