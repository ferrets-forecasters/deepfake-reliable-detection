from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.face_crop import FaceLandmarkDetector
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.reference import load_reference_repo

COLUMNS = ["image", "annotation_id", "label", "matched", "p_fake", "alpha", "iou"]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def coco_boxes(annotation: dict) -> tuple[float, float, float, float]:
    x, y, w, h = annotation["bbox"]
    return x, y, x + w, y + h


def summarize(faces_csv: Path, threshold: float) -> None:
    with open(faces_csv, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["annotation_id"] != ""]
    matched = [r for r in rows if r["matched"] == "1"]
    labels = np.array([int(r["label"]) for r in matched])
    scores = np.array([float(r["p_fake"]) for r in matched])
    print(f"annotated faces {len(rows)}, found by the detector {len(matched)} ({len(matched) / len(rows):.3f}), fake {int(labels.sum())}, real {int((1 - labels).sum())}")
    print(f"per face: AUC {roc_auc_score(labels, scores):.4f}, AP {average_precision_score(labels, scores):.4f}, "
          f"F1 at {threshold} {f1_score(labels, scores >= threshold):.4f}, accuracy {((scores >= threshold) == labels).mean():.4f}")
    by_image: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in matched:
        by_image[r["image"]].append((int(r["label"]), float(r["p_fake"])))
    multi = {k: v for k, v in by_image.items() if len(v) >= 2}
    image_label = np.array([max(l for l, _ in v) for v in multi.values()])
    image_score = np.array([max(p for _, p in v) for v in multi.values()])
    if len(set(image_label)) == 2:
        print(f"images with at least 2 matched faces ({len(multi)}): image AUC of the maximum {roc_auc_score(image_label, image_score):.4f}")
    mixed = {k: v for k, v in multi.items() if len({l for l, _ in v}) == 2}
    if mixed:
        gaps = [np.mean([p for l, p in v if l == 1]) - np.mean([p for l, p in v if l == 0]) for v in mixed.values()]
        ranked = np.mean([min(p for l, p in v if l == 1) > max(p for l, p in v if l == 0) for v in mixed.values()])
        print(f"images with both real and fake faces ({len(mixed)}): mean p_fake gap fake minus real {np.mean(gaps):.3f}, every fake above every real {ranked:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-face verdicts on a multi-face image set in COCO format (OpenForensics): the detector finds faces, each is matched to an annotated face, and the visual expert scores a static clip made of the crop.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--dataset", default="openforensics")
    parser.add_argument("--split", default="Val")
    parser.add_argument("--limit", type=int, default=0, help="first N images, 0 for all")
    parser.add_argument("--min-iou", type=float, default=0.25)
    parser.add_argument("--max-faces", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config.paths.datasets_root / args.dataset
    out_dir = config.paths.predictions_dir(f"{args.model}_on_{args.dataset}_{args.split.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    faces_csv = out_dir / "faces.csv"
    if args.summarize_only:
        summarize(faces_csv, args.threshold)
        return

    coco = json.loads((root / "labels" / f"{args.split}_poly.json").read_text(encoding="utf-8"))
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        by_image[annotation["image_id"]].append(annotation)
    images = coco["images"][: args.limit or None]

    done: set[str] = set()
    if faces_csv.exists():
        with open(faces_csv, encoding="utf-8") as f:
            done = {r["image"] for r in csv.DictReader(f)}
    else:
        with open(faces_csv, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    scorer = CheckpointScorer(load_reference_repo(_project_root() / repro["reference_repo"]), config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    detector = FaceLandmarkDetector(max_faces=args.max_faces, margin=0.25)

    for info in tqdm(images, desc=f"{args.model} on {args.dataset} {args.split}"):
        name = Path(info["file_name"]).name
        if name in done:
            continue
        bgr = cv2.imread(str(root / "raw" / args.split / name))
        if bgr is None:
            continue
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        detections = detector.detect(frame)
        annotations = by_image.get(info["id"], [])
        pairs, taken = [], set()
        for annotation in annotations:
            scores = [(iou(coco_boxes(annotation), d.box), i) for i, d in enumerate(detections) if i not in taken]
            best = max(scores, default=(0.0, -1))
            if best[0] >= args.min_iou:
                taken.add(best[1])
                pairs.append((annotation, best[1], best[0]))
            else:
                pairs.append((annotation, -1, 0.0))

        crops = []
        for _, index, _ in pairs:
            if index < 0:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in detections[index].box)
            crops.append([frame[y0:y1, x0:x1]] * VIDEOMAE_FRAMES)
        features = embedder.video_features(crops) if crops else np.zeros((0, 768), dtype=np.float32)
        cursor = 0
        with open(faces_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            for annotation, index, overlap in pairs:
                row = {"image": name, "annotation_id": annotation["id"], "label": annotation["category_id"], "matched": int(index >= 0), "p_fake": "", "alpha": "", "iou": round(overlap, 3)}
                if index >= 0:
                    p_fake, alpha = scorer.score_visual_only(features[cursor][None])
                    row["p_fake"], row["alpha"] = round(p_fake, 4), alpha
                    cursor += 1
                writer.writerow(row)
            if not pairs:
                writer.writerow({"image": name, "annotation_id": "", "label": "", "matched": 0, "p_fake": "", "alpha": "", "iou": 0})
    summarize(faces_csv, args.threshold)


if __name__ == "__main__":
    main()
