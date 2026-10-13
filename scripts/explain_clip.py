from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE, SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference.multi_subject import crop_track_frames
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.xai.cross_modal import (
    audio_shift_probe, explain_segment, lip_audio_alignment, mismatch_probe, mouth_cam_share,
)
from deepfake_detection.xai.overlay import box_to_crop
from deepfake_detection.xai.report import save_explanation

SEGMENT_SECONDS = 2.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one face in one 2-second segment: video CAM, audio saliency, branch contributions, mouth overlap, audio shift sweep.")
    parser.add_argument("video")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--dataset", default="fakeavceleb", help="dataset name used in the output folder")
    parser.add_argument("--start", type=float, default=0.0, help="segment start in seconds")
    parser.add_argument("--face", type=int, default=0, help="face track index, largest face first")
    parser.add_argument("--shifts", type=int, nargs="*", default=[-300, -200, -100, -40, 0, 40, 100, 200, 300], help="audio offsets in ms")
    parser.add_argument("--other-video", default=None, help="a second clip whose audio replaces this clip's audio in the mismatch probe")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    scorer = CheckpointScorer(reference, config.paths.checkpoints_dir(args.model))
    embedder = SegmentEmbedder()
    tracker = MultiFaceTracker(config.preprocessing.video.face_detector_fps)

    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(round(args.start * fps))
    end_frame = int(round((args.start + SEGMENT_SECONDS) * fps)) - 1
    cache: dict[int, np.ndarray] = {}

    def reader(index: int) -> np.ndarray:
        if index not in cache:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            _, frame = cap.read()
            cache[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return cache[index]

    tracks = tracker.tracks_for_indices(reader, [start_frame, end_frame], fps)
    track = tracks[args.face]
    indices = np.linspace(start_frame, end_frame, VIDEOMAE_FRAMES).round().astype(int).tolist()
    face_boxes = track.boxes_for_indices(indices)
    face_frames = crop_track_frames(reader, indices, face_boxes)
    waveform = extract_audio_waveform(video, args.start, args.start + SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)

    explanation = explain_segment(embedder, scorer, face_frames, waveform)

    mouth_boxes = track.mouth_boxes_for_indices(indices)
    per_frame = [box_to_crop(mouth_boxes[i], face_boxes[i]) for i in indices]
    per_step = [tuple(np.mean([per_frame[2 * s], per_frame[2 * s + 1]], axis=0)) for s in range(len(per_frame) // 2)]
    share = mouth_cam_share(explanation.video_cam, per_step)
    opening = np.array([track.mouth_open_for_indices(indices)[i] for i in indices])
    alignment = lip_audio_alignment(opening, len(indices) / SEGMENT_SECONDS, explanation.audio_step_cam)

    video_feature = embedder.video_feature(face_frames)
    shifts = audio_shift_probe(embedder, scorer, video_feature, waveform, WHISPER_SAMPLE_RATE, args.shifts)
    mismatch = None
    if args.other_video:
        other = extract_audio_waveform(Path(args.other_video), 0.0, SEGMENT_SECONDS, WHISPER_SAMPLE_RATE)
        mismatch = mismatch_probe(embedder, scorer, video_feature, other)
    cap.release()

    out_dir = config.paths.xai_dir(f"cross_modal_{args.model}_on_{args.dataset}") / f"{video.stem}_t{args.start:.1f}_face{args.face}"
    save_explanation(explanation, face_frames, out_dir, shifts, mismatch, alignment, share)
    print(f"P(fake) {explanation.p_fake:.3f}, alpha {explanation.alpha}, video share {explanation.video_share:.2f}, "
          f"mouth CAM share {share[0]:.2f} (uniform {share[1]:.2f}), lip-audio correlation {alignment.correlation:.2f}")
    print(f"written to {out_dir}")


if __name__ == "__main__":
    main()
