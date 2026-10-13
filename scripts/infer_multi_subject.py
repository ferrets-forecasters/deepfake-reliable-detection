from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference import MultiSubjectInference
from deepfake_detection.reference import load_reference_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="One verdict per face for each video: track the faces, embed each track, score each track with an exported checkpoint.")
    parser.add_argument("videos", nargs="+", help="video files, or folders searched for .mp4")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--name", default="multi_subject", help="dataset name used in the output folder")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    pipeline = MultiSubjectInference(
        tracker=MultiFaceTracker(config.preprocessing.video.face_detector_fps, max_faces=args.max_faces),
        embedder=SegmentEmbedder(),
        scorer=CheckpointScorer(reference, config.paths.checkpoints_dir(args.model)),
        threshold=args.threshold,
    )

    paths: list[Path] = []
    for item in map(Path, args.videos):
        paths += sorted(item.rglob("*.mp4")) if item.is_dir() else [item]

    out_dir = config.paths.predictions_dir(f"{args.model}_on_{args.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc="videos"):
        out = out_dir / f"{path.parent.name}_{path.stem}.json"
        if out.exists():
            continue
        result = pipeline.run(path)
        out.write_text(json.dumps({"video": str(path), "video_p_fake": result.video_p_fake, "flagged": result.flagged_tracks,
                                   "subjects": [asdict(s) for s in result.subjects]}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
