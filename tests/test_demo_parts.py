from pathlib import Path

import numpy as np

from deepfake_detection.inference import SpeechTurn, SubjectResult, aggregate_subjects
from deepfake_detection.inference.camera import parse_devices, record_command
from deepfake_detection.inference.diarization import labels_to_turns, voiced_mask, window_starts
from deepfake_detection.inference.summary import RollingVerdict, format_result, person_names


def test_parse_devices_reads_video_and_audio_names():
    output = '[dshow @ 000] "HD Webcam" (video)\n[dshow @ 000]   Alternative name "@device_pnp"\n[dshow @ 000] "Microphone Array (Intel)" (audio)\n'
    assert parse_devices(output) == {"video": ["HD Webcam"], "audio": ["Microphone Array (Intel)"]}


def test_record_command_names_both_devices_and_the_length():
    command = record_command(Path("ffmpeg.exe"), "HD Webcam", "Mic", 8, Path("out.mp4"), size="640x480")
    assert command[command.index("-i") + 1] == "video=HD Webcam:audio=Mic"
    assert command[command.index("-t") + 1] == "8" and command[command.index("-video_size") + 1] == "640x480"


def test_voiced_mask_marks_the_loud_part():
    wave = np.concatenate([np.zeros(16000), np.sin(np.arange(16000) * 0.2)]).astype(np.float32)
    mask = voiced_mask(wave, 16000)
    assert mask[:30].sum() == 0 and mask[-30:].all()


def test_window_starts_cover_the_whole_clip():
    starts = window_starts(3.2, 1.0, 0.5)
    assert starts[0] == 0.0 and abs(starts[-1] + 1.0 - 3.2) < 1e-6
    assert window_starts(0.8, 1.0, 0.5) == [0.0]


def test_labels_become_merged_turns_with_silence_dropped():
    starts = [0.0, 0.5, 1.0, 1.5, 2.0]
    turns = labels_to_turns(starts, 1.0, [0, 0, None, 1, 1], total_seconds=3.0, slot_seconds=0.5)
    assert turns[0].speaker == "speaker_0" and turns[0].start == 0.0
    assert {t.speaker for t in turns} == {"speaker_0", "speaker_1"}
    assert all(a.end <= b.start + 1e-9 for a, b in zip(turns, turns[1:]))


def test_format_result_lists_faces_left_to_right_and_the_video_verdict():
    subjects = [SubjectResult(1, 0.75, [0, 1], 0.8, 0.9, "speaker_1", True), SubjectResult(0, 0.25, [0, 1], 0.1, 0.2, "speaker_0", False)]
    text = format_result(aggregate_subjects(subjects, 0.5), "clip.mp4")
    lines = text.splitlines()
    assert lines[2].startswith("Person A") and lines[3].startswith("Person B")
    assert lines[3].split()[-1] == "FAKE" and lines[-1].startswith("video: FAKE")
    assert person_names(subjects) == {0: "Person B", 1: "Person A"} or person_names(subjects) == {1: "Person B", 0: "Person A"}


def test_rolling_verdict_averages_per_position_and_resets_when_faces_change():
    one = aggregate_subjects([SubjectResult(0, 0.2, [0], 0.2, None, None, False), SubjectResult(1, 0.8, [0], 0.9, None, None, True)], 0.5)
    two = aggregate_subjects([SubjectResult(0, 0.2, [0], 0.4, None, None, False), SubjectResult(1, 0.8, [0], 0.7, None, None, True)], 0.5)
    rolling = RollingVerdict(window=2)
    rolling.update(one)
    assert np.allclose(rolling.update(two), [0.3, 0.8])
    single = aggregate_subjects([SubjectResult(0, 0.5, [0], 0.6, None, None, True)], 0.5)
    assert rolling.update(single) == [0.6]


def test_render_draws_person_labels_and_waveform_rows():
    from deepfake_detection.inference.render import render_result, waveform_columns

    wave = np.sin(np.linspace(0, 200, 16000 * 4)).astype(np.float32)
    subjects = [SubjectResult(0, 0.25, [0], 0.1, None, "speaker_0", False, (10, 10, 60, 60)), SubjectResult(1, 0.75, [0], 0.9, None, "speaker_1", True, (110, 10, 160, 60))]
    result = aggregate_subjects(subjects, 0.5, [SpeechTurn(0, 2, "speaker_0"), SpeechTurn(2, 4, "speaker_1")], 4.0)
    result.waveform = wave
    image = render_result(np.full((80, 180, 3), 128, dtype=np.uint8), result, 1.0)
    assert image.shape[1] == 180 and image.shape[0] > 80 + 40
    assert waveform_columns(wave, 50).max() <= 1.0 and waveform_columns(None, 5).shape == (5,)
