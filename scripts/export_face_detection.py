from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.face_crop import MultiFaceTracker, link_boxes

SEGMENT_SECONDS = 2.0
COLORS = [(230, 40, 40), (40, 90, 230), (40, 170, 70), (230, 160, 20), (150, 60, 200)]
VIDEO_FIELDS = ["dataset", "sample_id", "label", "manip_type", "path", "width", "height", "fps", "duration_s", "segments", "segments_with_face",
                "tracks", "tracks_large", "largest_width_frac", "largest_height_frac", "largest_area_frac", "first_track_hits"]
BOX_FIELDS = ["sample_id", "frame", "time_s", "track_id", "x0", "y0", "x1", "y1"]


def test_rows(config, dataset: str) -> list[dict]:
    root = config.paths.datasets_root / dataset
    if not (root / "labels" / "test_split.csv").exists():
        return [{"path": str(p), "sample_id": p.stem, "label": "-", "manip_type": "-"} for p in sorted((root / "raw").rglob("*.mp4"))]
    with open(root / "labels" / "test_split.csv", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["own_test_seed42"] == "1"]
    for r in rows:
        r["path"] = str(root / "raw" / "test_split" / r["member"])
    return [r for r in rows if Path(r["path"]).exists()]


def draw(frame_bgr: np.ndarray, boxes: dict[int, tuple[float, float, float, float]], caption: str) -> np.ndarray:
    frame = frame_bgr.copy()
    thickness = max(1, round(min(frame.shape[:2]) / 160))
    for track_id, (x0, y0, x1, y1) in boxes.items():
        color = COLORS[track_id % len(COLORS)][::-1]
        cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), color, thickness)
        cv2.putText(frame, f"#{track_id}", (int(x0) + 3, int(y0) + 18 * thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thickness, color, thickness)
    cv2.putText(frame, caption, (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * thickness, (255, 255, 255), thickness)
    return frame


def analyse(tracker: MultiFaceTracker, path: str, max_seconds: float) -> tuple[dict, dict[int, np.ndarray], list[int], list, float]:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    last = max(min(count, int(max_seconds * fps)) - 1, 0)
    stride = max(int(round(fps / tracker.detector_fps)), 1)
    detection_indices = sorted(set(range(0, last + 1, stride)) | {last})
    needed = set(detection_indices) | set(np.linspace(0, last, 6).round().astype(int).tolist())
    frames = {}
    for index in range(last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if index in needed:
            frames[index] = frame
    cap.release()
    detection_indices = [i for i in detection_indices if i in frames]
    last = max(frames) if frames else 0
    detections = {i: tracker._detector.detect(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)) for i in detection_indices}
    tracks =[t for t in link_boxes({i: [d.box for d in faces] for i, faces in detections.items()}, tracker.iou_min) if len(t.detections) >= tracker.min_hits]
    tracks.sort(key=lambda t: t.mean_area, reverse=True)
    window = int(SEGMENT_SECONDS * fps)
    segments = max(1, (last + 1) // window)
    with_face = sum(any(any(s * window <= i < (s + 1) * window for i in t.detections) for t in tracks) for s in range(segments))
    largest = tracks[0] if tracks else None
    area = largest.mean_area if largest else 0.0
    boxes = [(b[2] - b[0], b[3] - b[1]) for b in largest.detections.values()] if largest else []
    row = {
        "width": int(width), "height": int(height), "fps": round(fps, 2), "duration_s": round((last + 1) / fps, 2), "segments": segments,
        "segments_with_face": with_face, "tracks": len(tracks), "tracks_large": sum(t.mean_area >= 0.25 * area for t in tracks) if tracks else 0,
        "largest_width_frac": round(float(np.mean([w for w, _ in boxes])) / width, 4) if boxes else 0.0,
        "largest_height_frac": round(float(np.mean([h for _, h in boxes])) / height, 4) if boxes else 0.0,
        "largest_area_frac": round(area / (width * height), 4) if tracks else 0.0,
        "first_track_hits": len(largest.detections) if largest else 0,
    }
    return row, frames, detection_indices, tracks, fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Face detection on seed-42 test-split videos of the datasets that are on this machine: rate and box size per dataset, failure examples (no face, a second large face) and box tracks over time.")
    parser.add_argument("--datasets", nargs="+", default=["hidf", "fakeavceleb"], help="a dataset without labels/test_split.csv is read whole from its raw folder")
    parser.add_argument("--topic", default="face_detection", help="folder name under results/analyses")
    parser.add_argument("--per-label", type=int, default=150, help="videos drawn per dataset and label")
    parser.add_argument("--examples", type=int, default=6, help="failure examples per dataset and failure type")
    parser.add_argument("--tracked", type=int, default=6, help="videos per dataset with a box strip and a box table")
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.analysis_dir(args.topic)
    for sub in ("failures_no_face", "failures_second_face", "tracked"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    tracker = MultiFaceTracker(detector_fps=config.preprocessing.video.face_detector_fps)
    rng = random.Random(args.seed)

    videos, boxes_table, manifest = [], [], []
    failure_pool: dict[tuple[str, str], list] = {}
    tracked_pool: dict[str, list] = {}
    for dataset in args.datasets:
        rows = test_rows(config, dataset)
        drawn = []
        for label in sorted({r["label"] for r in rows}):
            group = [r for r in rows if r["label"] == label]
            rng.shuffle(group)
            drawn += group[: args.per_label]
        for r in tqdm(drawn, desc=dataset):
            try:
                row, frames, indices, tracks, fps = analyse(tracker, r["path"], args.max_seconds)
            except Exception as error:
                print(f"{r['sample_id']}: {error}")
                continue
            record = {"dataset": dataset, "sample_id": r["sample_id"], "label": r["label"], "manip_type": r["manip_type"], "path": r["path"], **row}
            videos.append(record)
            for track in tracks:
                for index, (x0, y0, x1, y1) in track.detections.items():
                    boxes_table.append({"sample_id": r["sample_id"], "frame": index, "time_s": round(index / fps, 3), "track_id": track.track_id,
                                        "x0": round(x0, 1), "y0": round(y0, 1), "x1": round(x1, 1), "y1": round(y1, 1)})
            kind = "no_face" if row["tracks"] == 0 or row["segments_with_face"] < row["segments"] else "second_face" if row["tracks_large"] >= 2 else None
            middle = indices[len(indices) // 2] if indices else 0
            if kind and middle in frames:
                shown = {t.track_id: t.detections[min(t.detections, key=lambda i: abs(i - middle))] for t in tracks}
                failure_pool.setdefault((dataset, kind), []).append((r["sample_id"], draw(frames[middle], shown, f"{dataset} {r['sample_id']} tracks {row['tracks']}")))
            if tracks:
                kept = sorted(frames)
                strip_indices = [kept[j] for j in np.linspace(0, len(kept) - 1, 6).round().astype(int)]
                strip = []
                for i in strip_indices:
                    shown = {t.track_id: t.boxes_for_indices([i])[i] for t in tracks}
                    strip.append(draw(frames[i], shown, f"{i / fps:.1f} s"))
                tracked_pool.setdefault(dataset, []).append((r["sample_id"], strip, row["tracks"]))
            del frames

    saved = []
    for (dataset, kind), pool in failure_pool.items():
        folder = out_dir / ("failures_no_face" if kind == "no_face" else "failures_second_face")
        for sample_id, image in rng.sample(pool, min(args.examples, len(pool))):
            path = folder / f"{dataset}_{sample_id}.png"
            cv2.imwrite(str(path), image)
            saved.append({"kind": kind, "dataset": dataset, "sample_id": sample_id, "file": str(path), "available_of_this_kind": len(pool),
                          "why_chosen": f"seeded draw among the {len(pool)} drawn test videos of this failure type"})
    for dataset, pool in tracked_pool.items():
        multi = [p for p in pool if p[2] >= 2]
        pick = rng.sample(multi, min(args.tracked // 2, len(multi)))
        pick += rng.sample([p for p in pool if p not in pick], min(args.tracked - len(pick), len(pool) - len(pick)))
        for sample_id, strip, tracks in pick:
            path = out_dir / "tracked" / f"{dataset}_{sample_id}.png"
            height = min(s.shape[0] for s in strip)
            cv2.imwrite(str(path), np.hstack([cv2.resize(s, (int(s.shape[1] * height / s.shape[0]), height)) for s in strip]))
            saved.append({"kind": "tracked", "dataset": dataset, "sample_id": sample_id, "file": str(path), "available_of_this_kind": len(pool),
                          "why_chosen": f"seeded draw, up to half among videos with two or more tracks; {tracks} tracks"})

    summary = {}
    for dataset in args.datasets:
        v = [x for x in videos if x["dataset"] == dataset]
        if not v:
            continue
        summary[dataset] = {
            "videos": len(v), "videos_with_a_face": sum(x["tracks"] > 0 for x in v) / len(v),
            "segments_with_a_face": sum(x["segments_with_face"] for x in v) / sum(x["segments"] for x in v),
            "videos_with_two_large_faces": sum(x["tracks_large"] >= 2 for x in v) / len(v),
            "largest_face_area_fraction_median": float(np.median([x["largest_area_frac"] for x in v if x["tracks"]])),
            "largest_face_width_fraction_median": float(np.median([x["largest_width_frac"] for x in v if x["tracks"]])),
            "largest_face_height_fraction_median": float(np.median([x["largest_height_frac"] for x in v if x["tracks"]])),
            "per_label": {lab: {"videos": len(g), "videos_with_a_face": sum(x["tracks"] > 0 for x in g) / len(g)}
                          for lab in sorted({x["label"] for x in v}) for g in [[x for x in v if x["label"] == lab]]},
            "selection": f"up to {args.per_label} videos per label drawn with seed {args.seed} from the own seed-42 test split (every video of the raw folder when the dataset has no split file), first {args.max_seconds:g} s of each, detector at {tracker.detector_fps} per second",
        }
    for name, rows, fields in (("videos.csv", videos, VIDEO_FIELDS), ("boxes_over_time.csv", boxes_table, BOX_FIELDS),
                               ("manifest.csv", saved, ["kind", "dataset", "sample_id", "file", "available_of_this_kind", "why_chosen"])):
        with open(out_dir / name, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(summary, sort_keys=False))


if __name__ == "__main__":
    main()
