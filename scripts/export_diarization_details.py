from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import WHISPER_SAMPLE_RATE
from deepfake_detection.inference.diarization import EcapaDiarizer, labels_to_turns
from deepfake_detection.inference.multi_subject import frame_reader
from deepfake_detection.inference.speakers import _zscore, link_speakers_to_tracks, speech_activity

TURN_SECONDS = 4.0


def lip_scores(turns, mouth: dict[int, tuple[np.ndarray, np.ndarray]]) -> dict[str, dict[int, float]]:
    """Mean of lip motion times speech activity (both z-scored) for every speaker against every face track: the linking numbers."""
    out: dict[str, dict[int, float]] = {}
    for speaker in sorted({t.speaker for t in turns}):
        out[speaker] = {}
        for track_id, (times, opening) in mouth.items():
            if len(times) < 3:
                continue
            motion = _zscore(np.abs(np.diff(opening)))
            activity = _zscore(speech_activity(turns, speaker, (times[:-1] + times[1:]) / 2))
            out[speaker][track_id] = float(np.mean(motion * activity)) if activity.any() and motion.any() else 0.0
    return out


def analyse(path: Path, diarizer: EcapaDiarizer, tracker: MultiFaceTracker) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    reader = frame_reader(cap, count)
    tracks = tracker.tracks_for_indices(reader, [0, count - 1], fps)
    cap.release()
    waveform = extract_audio_waveform(path, 0.0, count / fps, WHISPER_SAMPLE_RATE)
    starts, kept, embeddings = diarizer.window_embeddings(waveform, WHISPER_SAMPLE_RATE)
    if not kept:
        return {"turns": [], "similarity": np.zeros((0, 0)), "kept": [], "starts": starts, "tracks": tracks, "scores": {}, "linked": {}, "fps": fps}
    norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    assigned = diarizer._cluster(embeddings, len(tracks))
    order: dict[int, int] = {}
    for label in assigned:
        order.setdefault(int(label), len(order))
    labels: list[int | None] = [None] * len(starts)
    for index, label in zip(kept, assigned):
        labels[index] = order[int(label)]
    turns = labels_to_turns(starts, diarizer.window_seconds, labels, len(waveform) / WHISPER_SAMPLE_RATE, diarizer.hop_seconds)
    mouth = {}
    for track in tracks:
        frames = sorted(track.mouth_open)
        mouth[track.track_id] = (np.array(frames) / fps, np.array([track.mouth_open[f] for f in frames]))
    return {"turns": turns, "similarity": norm @ norm.T, "kept": kept, "starts": starts, "tracks": tracks, "scores": lip_scores(turns, mouth),
            "linked": link_speakers_to_tracks(turns, mouth) if len({t.speaker for t in turns}) > 1 else {}, "fps": fps}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diarization details of two-person videos: the window cosine-similarity matrix, estimated and true turns, and the lip-motion score of every face against every speaker (the linking numbers). One npz per video and a manifest.")
    parser.add_argument("--conversation", default="fakeavceleb_conversation_own_test_lossless")
    parser.add_argument("--single-voice", default="fakeavceleb_tiled_own_test_lossless")
    parser.add_argument("--limit-single", type=int, default=40)
    parser.add_argument("--select", type=int, default=6, help="videos flagged as selected per set; failures first, then a seeded draw")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = config.paths.analysis_dir("diarization_matrices")
    out_dir.mkdir(parents=True, exist_ok=True)
    diarizer = EcapaDiarizer()
    tracker = MultiFaceTracker(config.preprocessing.video.face_detector_fps)

    rows = []
    for dataset, kind, limit in ((args.conversation, "conversation", None), (args.single_voice, "single_voice", args.limit_single)):
        root = config.paths.datasets_root / dataset
        with open(root / "labels" / "manifest.csv", encoding="utf-8", newline="") as f:
            manifest = list(csv.DictReader(f))
        if limit:
            manifest = manifest[:limit]
        for row in tqdm(manifest, desc=kind):
            path = root / row["raw_path"]
            info = analyse(path, diarizer, tracker)
            turns = info["turns"]
            speakers = sorted({t.speaker for t in turns})
            if kind == "conversation":
                true_turns = [{"start": 0.0, "end": TURN_SECONDS, "speaker": "left face"}, {"start": TURN_SECONDS, "end": 2 * TURN_SECONDS, "speaker": "right face"}]
                tracks = sorted(info["tracks"], key=lambda t: t.center_x())
                first = next((t.speaker for t in turns if t.start < 1.0), None)
                second = next((t.speaker for t in turns if t.end > TURN_SECONDS + 1.0), None)
                expected = {first: tracks[0].track_id, second: tracks[-1].track_id} if len(tracks) >= 2 and first and second and first != second else {}
                correct = sum(1 for s, t in info["linked"].items() if expected.get(s) == t)
            else:
                true_turns = [{"start": 0.0, "end": None, "speaker": f"one voice, from the {row['audio_from']} clip"}]
                correct = None
            video_id = Path(row["raw_path"]).stem
            np.savez_compressed(
                out_dir / f"{kind}_{video_id}.npz", similarity=info["similarity"].astype(np.float32), window_starts=np.array(info["starts"], dtype=np.float32),
                kept_windows=np.array(info["kept"], dtype=np.int32), estimated_turns=json.dumps([{"start": t.start, "end": t.end, "speaker": t.speaker} for t in turns]),
                true_turns=json.dumps(true_turns), lip_scores=json.dumps({s: {str(k): v for k, v in d.items()} for s, d in info["scores"].items()}),
                linked=json.dumps({s: int(t) for s, t in info["linked"].items()}), track_center_x=json.dumps({str(t.track_id): float(t.center_x()) for t in info["tracks"]}),
            )
            rows.append({"id": f"{kind}_{video_id}", "set": kind, "video": str(path), "plan": f"{row.get('left_kind', '')}+{row.get('right_kind', '')}",
                         "speakers_found": len(speakers), "estimated_turns": json.dumps([[round(t.start, 2), round(t.end, 2), t.speaker] for t in turns]),
                         "true_turns": json.dumps(true_turns), "people_linked": len(info["linked"]), "links_correct": correct, "faces": len(info["tracks"]),
                         "windows_kept": len(info["kept"]), "windows_total": len(info["starts"])})

    rng = random.Random(args.seed)
    for kind in ("conversation", "single_voice"):
        part = [r for r in rows if r["set"] == kind]
        if kind == "conversation":
            hard = [r for r in part if r["speakers_found"] != 2 or (r["links_correct"] or 0) < 2]
        else:
            hard = [r for r in part if r["speakers_found"] > 1]
        rng.shuffle(hard)
        failures = hard[: max(args.select // 2, 3)]
        rest = [r for r in part if r not in failures]
        rng.shuffle(rest)
        draws = rest[: args.select - len(failures)]
        reason = "failure: not two speakers or a link missing" if kind == "conversation" else "failure: one voice split into two speakers"
        for r in part:
            r["selected"] = int(r in failures or r in draws)
            r["why_chosen"] = reason if r in failures else ("seeded random draw" if r in draws else "")
    with open(out_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for kind in ("conversation", "single_voice"):
        part = [r for r in rows if r["set"] == kind]
        print(kind, len(part), "videos;", sum(r["speakers_found"] == 2 for r in part), "with two speakers;", sum(r["speakers_found"] == 1 for r in part), "with one;",
              "selected", sum(r["selected"] for r in part))


if __name__ == "__main__":
    main()
