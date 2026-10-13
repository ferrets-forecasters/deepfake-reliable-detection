import importlib.util
import json
from pathlib import Path

from deepfake_detection.data.tiling import ffmpeg_tile_command, index_clips, own_test_clips, pick_pairs
from deepfake_detection.inference import SpeechTurn, SubjectResult, aggregate_subjects


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


def test_own_test_clips_keep_only_flagged_members_under_the_raw_root(tmp_path):
    csv_path = tmp_path / "test_split.csv"
    csv_path.write_text(
        "member,sample_id,label,manip_type,subject,own_test_seed42,official_test,stem_candidates\n"
        "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio/African/men/id1/a.mp4,a,0,real,fac_a,1,0,1\n"
        "FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/African/men/id2/b.mp4,b,1,audio_video,fac_b,0,1,1\n",
        encoding="utf-8",
    )
    raw = tmp_path / "raw"
    kept = make_clip(raw, "RealVideo-RealAudio", "African", "men", "id1", "a.mp4")
    make_clip(raw, "FakeVideo-FakeAudio", "African", "men", "id2", "b.mp4")
    allowed = own_test_clips(raw, csv_path)
    assert allowed == {kept}
    assert [p for paths in index_clips(raw, allowed).values() for p in paths] == [kept]


def test_saved_result_carries_the_speaker_turns_and_duration(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    result = aggregate_subjects([SubjectResult(0, 0.2, [0], 0.1, None, "speaker_0", False)], 0.5, [SpeechTurn(0.0, 4.0, "speaker_0")], 8.0)
    demo.save_result(result, "clip.mkv", tmp_path, "clip")
    payload = json.loads((tmp_path / "clip.json").read_text(encoding="utf-8"))
    assert payload["duration"] == 8.0 and payload["turns"] == [{"start": 0.0, "end": 4.0, "speaker": "speaker_0"}]


def test_command_keeps_the_chosen_audio_and_stops_at_the_shorter_clip():
    command = ffmpeg_tile_command(Path("ffmpeg"), Path("l.mp4"), Path("r.mp4"), Path("o.mp4"), "right", 360)
    assert command[command.index("-map", command.index("[v]")) + 1] == "1:a:0"
    assert "hstack=inputs=2:shortest=1" in command[command.index("-filter_complex") + 1]
    assert "-shortest" in command
