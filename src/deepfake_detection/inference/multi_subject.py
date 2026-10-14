from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from deepfake_detection.data.audio_features import extract_audio_waveform
from deepfake_detection.data.face_crop import FaceTrack, MultiFaceTracker
from deepfake_detection.data.segment_embedding import VIDEOMAE_FRAMES, WHISPER_SAMPLE_RATE
from deepfake_detection.inference.speakers import Diarizer, SingleSpeaker, SpeechTurn, link_speakers_to_tracks, speaker_waveform


@dataclass
class SubjectResult:
    track_id: int
    center_x: float                  # mean face position, 0 left edge to 1 right edge of the frame
    segments: list[int]              # segment indices the subject appears in
    p_fake: float
    alpha: float | None
    speaker: str | None              # speaker linked to this face, None when the shared audio was used
    flagged: bool
    box: tuple[float, float, float, float] | None = None  # mean face box in pixels, x0 y0 x1 y1


@dataclass
class VideoResult:
    subjects: list[SubjectResult] = field(default_factory=list)
    video_p_fake: float = float("nan")  # maximum over subjects: the video is fake if any subject is
    threshold: float = 0.5
    turns: list[SpeechTurn] = field(default_factory=list)
    duration: float = 0.0
    fps: float = 0.0
    waveform: np.ndarray | None = None  # mono audio at 16 kHz over the scored duration
    tracks: dict[int, FaceTrack] = field(default_factory=dict)   # filled when explain=True
    cams: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)  # track id -> segment -> video CAM [steps, grid, grid]

    @property
    def flagged_tracks(self) -> list[int]:
        return [s.track_id for s in self.subjects if s.flagged]


def aggregate_subjects(subjects: list[SubjectResult], threshold: float, turns: list[SpeechTurn] | None = None, duration: float = 0.0) -> VideoResult:
    if not subjects:
        return VideoResult(threshold=threshold, turns=turns or [], duration=duration)
    return VideoResult(subjects=subjects, video_p_fake=max(s.p_fake for s in subjects), threshold=threshold, turns=turns or [], duration=duration)


def crop_track_frames(reader, indices: list[int], boxes: dict) -> list[np.ndarray]:
    crops = []
    for idx in indices:
        x0, y0, x1, y1 = (int(round(v)) for v in boxes[idx])
        crops.append(reader(idx)[y0:y1, x0:x1])
    return crops


def frame_reader(cap, frame_count: int, max_bytes: float = 600e6, forward_gap: int = 150, cache_frames: int = 240):
    """Frame accessor by index. A clip that fits in max_bytes is decoded once, in order, which is much
    faster than a seek per frame; a longer clip is read forward frame by frame for small gaps and by seeking otherwise,
    keeping the last `cache_frames` frames."""
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count * width * height * 3 <= max_bytes:
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return lambda index: frames[min(index, len(frames) - 1)]

    cache: OrderedDict[int, np.ndarray] = OrderedDict()
    position = {"next": 0}  # index of the frame the decoder returns next

    def read(index: int) -> np.ndarray:
        if index in cache:
            cache.move_to_end(index)
            return cache[index]
        gap = index - position["next"]
        if 0 <= gap <= forward_gap:
            for _ in range(gap):  # skipping ahead frame by frame is far cheaper than a seek to an earlier key frame
                cap.grab()
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            if cache:
                return next(reversed(cache.values()))
            raise ValueError(f"could not read frame {index}")
        position["next"] = index + 1
        cache[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while len(cache) > cache_frames:
            cache.popitem(last=False)
        return cache[index]

    return read


def segment_frame_range(segment: int, segment_seconds: float, fps: float) -> tuple[int, int]:
    return int(round(segment * segment_seconds * fps)), int(round((segment + 1) * segment_seconds * fps)) - 1


class MultiSubjectInference:
    """One verdict per face in a video. Faces are tracked over the whole clip, each track is cut
    into 2-second segments, every segment is embedded (VideoMAE on that face, Whisper on that face's
    audio), and the checkpoint scores each track's segment sequence. The video verdict is the maximum.

    Audio per face: with a diarizer, each speaker is linked to the face whose lip motion follows
    that speaker's turns and that face gets only that speaker's audio; faces without a linked
    speaker, and every face when there is one speaker, get the full clip audio."""

    def __init__(
        self, tracker: MultiFaceTracker, embedder, scorer, diarizer: Diarizer | None = None,
        segment_seconds: float = 2.0, threshold: float = 0.5, presence_tolerance_seconds: float = 0.6,
        link_min_correlation: float = 0.1,
    ):
        self.tracker = tracker
        self.embedder = embedder
        self.scorer = scorer
        self.diarizer = diarizer or SingleSpeaker()
        self.segment_seconds = segment_seconds
        self.threshold = threshold
        self.presence_tolerance_seconds = presence_tolerance_seconds
        self.link_min_correlation = link_min_correlation

    def _link_speakers(self, tracks: list[FaceTrack], turns, fps: float) -> dict[int, str]:
        mouth = {}
        for track in tracks:
            frames = sorted(track.mouth_open)
            mouth[track.track_id] = (np.array(frames) / fps, np.array([track.mouth_open[f] for f in frames]))
        by_speaker = link_speakers_to_tracks(turns, mouth, self.link_min_correlation)
        return {track_id: speaker for speaker, track_id in by_speaker.items()}

    def run(self, video_path: Path, explain: bool = False) -> VideoResult:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_segments = int((frame_count / fps) // self.segment_seconds)
        if n_segments == 0:
            cap.release()
            return VideoResult(threshold=self.threshold)

        reader = frame_reader(cap, frame_count)

        duration = n_segments * self.segment_seconds
        waveform = extract_audio_waveform(video_path, 0.0, duration, WHISPER_SAMPLE_RATE)
        try:
            tracks = self.tracker.tracks_for_indices(reader, [0, frame_count - 1], fps)
        except ValueError:
            cap.release()
            return VideoResult(threshold=self.threshold)
        turns = self.diarizer(waveform, WHISPER_SAMPLE_RATE, num_speakers=len(tracks))
        speaker_of = self._link_speakers(tracks, turns, fps) if len({t.speaker for t in turns}) > 1 else {}

        tolerance = int(round(self.presence_tolerance_seconds * fps))
        samples = int(self.segment_seconds * WHISPER_SAMPLE_RATE)
        shared_audio = None
        subjects = []
        cams: dict[int, dict[int, np.ndarray]] = {}
        for track in tracks:
            speaker = speaker_of.get(track.track_id)
            used, crops_per_segment = [], []
            for segment in range(n_segments):
                start_f, end_f = segment_frame_range(segment, self.segment_seconds, fps)
                if not track.present_between(start_f, end_f, tolerance):
                    continue
                indices = np.linspace(start_f, end_f, VIDEOMAE_FRAMES).round().astype(int).tolist()
                crops_per_segment.append(crop_track_frames(reader, indices, track.boxes_for_indices(indices)))
                used.append(segment)
            if not used:
                continue
            if speaker is None:
                if shared_audio is None:
                    shared_audio = self.embedder.audio_features([waveform[i * samples:(i + 1) * samples] for i in range(n_segments)])
                audio_feats = shared_audio[used]
            else:
                track_wave = speaker_waveform(waveform, turns, speaker, WHISPER_SAMPLE_RATE)
                audio_feats = self.embedder.audio_features([track_wave[i * samples:(i + 1) * samples] for i in used])
            video_feats = self.embedder.video_features(crops_per_segment)
            if not used:
                continue
            if explain:
                from deepfake_detection.xai.cross_modal import video_cam_for_segment

                cams[track.track_id] = {
                    segment: video_cam_for_segment(self.embedder, self.scorer, crops, audio_feats[row])
                    for row, (segment, crops) in enumerate(zip(used, crops_per_segment))
                }
            x = self.scorer.normalize(video_feats, audio_feats)
            p_fake, alpha = self.scorer.score(x)
            subjects.append(SubjectResult(
                track_id=track.track_id, center_x=track.center_x() / cap.get(cv2.CAP_PROP_FRAME_WIDTH), segments=used,
                p_fake=p_fake, alpha=alpha, speaker=speaker, flagged=p_fake >= self.threshold,
                box=tuple(float(v) for v in np.mean([track.detections[i] for i in track.detections], axis=0)),
            ))
        cap.release()
        result = aggregate_subjects(subjects, self.threshold, turns, duration)
        result.fps = fps
        result.waveform = waveform
        if explain:
            result.tracks, result.cams = {t.track_id: t for t in tracks}, cams
        return result
