from __future__ import annotations

import re
from pathlib import Path

from deepfake_detection.data.manifest import SegmentRecord
from deepfake_detection.data.segmentation import segment_windows, video_duration_seconds


def _segments_for_video(
    dataset: str, subject_id: str, video_path: Path, label: int, segment_seconds: int
) -> list[SegmentRecord]:
    duration = video_duration_seconds(video_path)
    records = []
    for i, (start, end) in enumerate(segment_windows(duration, segment_seconds)):
        segment_id = f"{video_path.stem}_{i:04d}"
        records.append(
            SegmentRecord(
                segment_id=segment_id,
                dataset=dataset,
                subject_id=subject_id,
                video_path=str(video_path),
                audio_path=str(video_path),
                start_sec=start,
                end_sec=end,
                label=label,
            )
        )
    return records


def build_ffpp_manifest(raw_dir: Path, segment_seconds: int, compression: str = "c23") -> list[SegmentRecord]:
    """Layout confirmed from ondyari/FaceForensics/dataset/README.md:
    original_sequences/youtube/<compression>/videos/<id>.mp4 (real, id 000-999);
    manipulated_sequences/<Method>/<compression>/videos/<target>_<source>.mp4 (fake).
    Subject id is the target-sequence id, the identity whose face appears."""
    records: list[SegmentRecord] = []

    real_dir = raw_dir / "original_sequences" / "youtube" / compression / "videos"
    for video_path in sorted(real_dir.glob("*.mp4")):
        subject_id = f"ffpp_{video_path.stem}"
        records += _segments_for_video("ffpp", subject_id, video_path, label=0, segment_seconds=segment_seconds)

    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"]
    for method in methods:
        fake_dir = raw_dir / "manipulated_sequences" / method / compression / "videos"
        if not fake_dir.is_dir():
            continue
        for video_path in sorted(fake_dir.glob("*.mp4")):
            target_id = video_path.stem.split("_")[0]
            subject_id = f"ffpp_{target_id}"
            records += _segments_for_video("ffpp", subject_id, video_path, label=1, segment_seconds=segment_seconds)

    return records


def build_celebdfv2_manifest(raw_dir: Path, segment_seconds: int) -> list[SegmentRecord]:
    """Layout confirmed from yuezunli/celeb-deepfakeforensics: Celeb-real/ and
    YouTube-real/ hold real videos, Celeb-synthesis/ holds fake ones. Filenames
    follow id<N>_0000.mp4 (real) and id<N>_id<M>_0000.mp4 (synthesis, source id
    N onto target id M); subject id uses the target identity."""
    records: list[SegmentRecord] = []
    id_pattern = re.compile(r"id(\d+)")

    for folder, label in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        video_dir = raw_dir / folder
        if not video_dir.is_dir():
            continue
        for video_path in sorted(video_dir.glob("*.mp4")):
            ids = id_pattern.findall(video_path.stem)
            target_id = ids[-1] if ids else video_path.stem
            subject_id = f"celebdfv2_id{target_id}"
            records += _segments_for_video(
                "celebdfv2", subject_id, video_path, label=label, segment_seconds=segment_seconds
            )

    return records


def build_fakeavceleb_manifest(raw_dir: Path, segment_seconds: int) -> list[SegmentRecord]:
    """Layout not independently verified (README is behind the dataset's request-form
    gate). Assumed from the FakeAVCeleb paper's own description: <Category>/<Race>/<Gender>/<Identity>/*.mp4,
    Category in {RealVideo-RealAudio, RealVideo-FakeAudio, FakeVideo-RealAudio,
    FakeVideo-FakeAudio}. Only RealVideo-RealAudio is label 0; the other three
    categories all carry a manipulated stream. Re-check this glob against the
    actual download once access is granted."""
    records: list[SegmentRecord] = []
    categories = {
        "RealVideo-RealAudio": 0,
        "RealVideo-FakeAudio": 1,
        "FakeVideo-RealAudio": 1,
        "FakeVideo-FakeAudio": 1,
    }
    for category, label in categories.items():
        category_dir = raw_dir / category
        if not category_dir.is_dir():
            continue
        for identity_dir in sorted(p for p in category_dir.glob("*/*/*") if p.is_dir()):
            subject_id = f"fakeavceleb_{identity_dir.relative_to(category_dir)}".replace("/", "_").replace(
                "\\", "_"
            )
            for video_path in sorted(identity_dir.glob("*.mp4")):
                records += _segments_for_video(
                    "fakeavceleb", subject_id, video_path, label=label, segment_seconds=segment_seconds
                )

    return records


def build_generic_manifest(
    dataset_name: str,
    raw_dir: Path,
    segment_seconds: int,
    real_glob: str,
    fake_glob: str,
    subject_id_from_path,
) -> list[SegmentRecord]:
    """Fallback builder for datasets whose on-disk layout has not been directly
    inspected yet (HiDF, PolyglotFake). Verify real_glob/fake_glob/subject_id_from_path
    against the actual downloaded tree before relying on the resulting manifest."""
    records: list[SegmentRecord] = []
    for video_path in sorted(raw_dir.glob(real_glob)):
        subject_id = subject_id_from_path(video_path)
        records += _segments_for_video(
            dataset_name, subject_id, video_path, label=0, segment_seconds=segment_seconds
        )
    for video_path in sorted(raw_dir.glob(fake_glob)):
        subject_id = subject_id_from_path(video_path)
        records += _segments_for_video(
            dataset_name, subject_id, video_path, label=1, segment_seconds=segment_seconds
        )
    return records
