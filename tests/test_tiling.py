from pathlib import Path

from deepfake_detection.data.tiling import ffmpeg_tile_command, index_clips, pick_pairs


def make_clip(root: Path, category: str, race: str, gender: str, speaker: str, name: str) -> Path:
    path = root / category / race / gender / speaker / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_pairs_share_race_and_gender_and_differ_in_speaker(tmp_path):
    for speaker in ("id1", "id2", "id3"):
        make_clip(tmp_path, "RealVideo-RealAudio", "African", "men", speaker, "a.mp4")
        make_clip(tmp_path, "FakeVideo-FakeAudio", "African", "men", speaker, "b.mp4")
    make_clip(tmp_path, "FakeVideo-FakeAudio", "Asian (East)", "women", "id9", "c.mp4")
    index = index_clips(tmp_path)
    pairs = pick_pairs(index, "fake_av", "real", count=4, seed=1)
    assert len(pairs) == 4
    for left, right in pairs:
        assert left.parts[-4:-2] == right.parts[-4:-2]
        assert left.parts[-2] != right.parts[-2]


def test_command_keeps_the_chosen_audio_and_stops_at_the_shorter_clip():
    command = ffmpeg_tile_command(Path("ffmpeg"), Path("l.mp4"), Path("r.mp4"), Path("o.mp4"), "right", 360)
    assert command[command.index("-map", command.index("[v]")) + 1] == "1:a:0"
    assert "hstack=inputs=2:shortest=1" in command[command.index("-filter_complex") + 1]
    assert "-shortest" in command
