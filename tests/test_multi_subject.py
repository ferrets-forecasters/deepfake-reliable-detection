import cv2
import numpy as np

from deepfake_detection.data.face_crop import FaceTrack
from deepfake_detection.inference import (
    MultiSubjectInference, SpeechTurn, SubjectResult, aggregate_subjects, link_speakers_to_tracks, speaker_waveform, speech_activity,
)
from deepfake_detection.inference import multi_subject


def test_speech_activity_and_speaker_waveform():
    turns = [SpeechTurn(0.0, 1.0, "A"), SpeechTurn(1.0, 2.0, "B")]
    assert speech_activity(turns, "A", np.array([0.5, 1.5])).tolist() == [1.0, 0.0]
    wave = np.ones(32, dtype=np.float32)
    kept = speaker_waveform(wave, turns, "B", sample_rate=16)
    assert kept[:16].sum() == 0 and kept[16:].sum() == 16


def test_each_speaker_is_linked_to_the_face_whose_lips_move_while_they_talk():
    times = np.arange(0, 10, 0.5)
    turns = [SpeechTurn(0, 5, "A"), SpeechTurn(5, 10, "B")]
    first = np.where(times < 5, 0.3 + 0.2 * np.sin(times * 6), 0.3)
    second = np.where(times >= 5, 0.3 + 0.2 * np.sin(times * 6), 0.3)
    assert link_speakers_to_tracks(turns, {1: (times, first), 2: (times, second)}) == {"A": 1, "B": 2}


def test_a_still_face_stays_unlinked():
    times = np.arange(0, 10, 0.5)
    turns = [SpeechTurn(0, 5, "A"), SpeechTurn(5, 10, "B")]
    assert link_speakers_to_tracks(turns, {1: (times, np.full(len(times), 0.3))}) == {}


def test_video_verdict_is_the_maximum_over_subjects():
    subjects = [SubjectResult(0, 0.25, [0], 0.1, None, None, False), SubjectResult(1, 0.75, [0], 0.8, None, None, True)]
    result = aggregate_subjects(subjects, 0.5)
    assert result.video_p_fake == 0.8 and result.flagged_tracks == [1]
    assert np.isnan(aggregate_subjects([], 0.5).video_p_fake)


def test_track_presence_and_mouth_interpolation():
    track = FaceTrack(0, {0: (0, 0, 10, 10), 100: (0, 0, 10, 10)}, mouth_open={0: 0.1, 100: 0.3})
    assert track.present_between(0, 49) and not track.present_between(40, 60)
    assert track.present_between(40, 60, tolerance=45)
    assert track.mouth_open_for_indices([50])[50] == 0.2


class FakeTracker:
    def tracks_for_indices(self, reader, indices, fps):
        left = FaceTrack(0, {0: (0, 0, 32, 64), 149: (0, 0, 32, 64)})
        right = FaceTrack(1, {0: (32, 0, 64, 64), 149: (32, 0, 64, 64)})
        return [left, right]


class FakeEmbedder:
    def video_features(self, segments):
        return np.stack([np.full(768, float(np.mean(crops[0])), dtype=np.float32) for crops in segments])

    def audio_features(self, waves):
        return np.zeros((len(waves), 1024), dtype=np.float32)


class FakeScorer:
    def normalize(self, video, audio):
        return np.concatenate([video, audio], axis=1)

    def score(self, x):
        return float(x[:, 0].mean() / 255.0), None


def test_pipeline_scores_each_face_on_its_own_segments(tmp_path, monkeypatch):
    path = tmp_path / "two_people.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64))
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:, :32] = 10
    frame[:, 32:] = 200
    for _ in range(150):
        writer.write(frame)
    writer.release()
    monkeypatch.setattr(multi_subject, "extract_audio_waveform", lambda *a, **k: np.zeros(16000 * 6, dtype=np.float32))

    result = MultiSubjectInference(FakeTracker(), FakeEmbedder(), FakeScorer()).run(path)

    assert [s.track_id for s in result.subjects] == [0, 1]
    assert all(s.segments == [0, 2] for s in result.subjects)
    assert result.subjects[0].p_fake < 0.2 and result.subjects[1].p_fake > 0.6
    assert result.flagged_tracks == [1] and result.video_p_fake == result.subjects[1].p_fake
