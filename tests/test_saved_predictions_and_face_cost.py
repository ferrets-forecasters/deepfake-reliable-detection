import importlib.util
import json
from pathlib import Path

import numpy as np

from deepfake_detection.evaluation.saved_predictions import load_saved_predictions

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_csv_predictions_use_the_best_checkpoint_columns_and_nan_for_a_missing_gate(tmp_path):
    (tmp_path / "predictions.csv").write_text(
        "video,dataset,subject,label,manip_type,audio_avail,p_fake_best_f1,alpha_best_f1,p_fake_swa,alpha_swa\n"
        "a,hidf,s1,1,video_only,1,0.9,0.4,0.1,0.1\n"
        "b,hidf,s2,0,real,1,0.2,,0.3,\n", encoding="utf-8")
    data = load_saved_predictions(tmp_path)
    assert data["probs"].tolist() == [0.9, 0.2]
    assert data["labels"].tolist() == [1, 0]
    assert data["alpha"][0] == 0.4 and np.isnan(data["alpha"][1])
    assert data["manip_type"].tolist() == ["video_only", "real"]


def test_npz_predictions_win_over_the_csv(tmp_path):
    np.savez(tmp_path / "predictions.npz", probs=np.array([0.7]), labels=np.array([1]), alpha=np.array([0.5]), manip_type=np.array(["audio_video"]))
    (tmp_path / "predictions.csv").write_text("video,label,p_fake_best_f1\nz,0,0.1\n", encoding="utf-8")
    assert load_saved_predictions(tmp_path)["probs"].tolist() == [0.7]


def test_face_cost_ratio_counts_only_clips_with_one_and_two_tracks_found():
    summarize = load_script("benchmark_faces").summarize

    def row(clip, faces, total, tracks, **stages):
        base = {"clip": clip, "faces": str(faces), "diarizer": "single", "total_s": str(total), "tracks_found": str(tracks), "other_s": "0"}
        base.update({f"{s}_s": str(stages.get(s, 0.0)) for s in ["detect_track", "diarize", "audio_embed", "video_embed", "score"]})
        return base

    rows = [
        row("a", 1, 2.0, 1, video_embed=1.0), row("a", 2, 3.0, 2, video_embed=2.0),
        row("b", 1, 2.0, 1, video_embed=1.0), row("b", 2, 4.0, 2, video_embed=2.0),
        row("c", 1, 2.0, 1), row("c", 2, 9.0, 1),
    ]
    entry = summarize(rows)["single"]
    assert entry["clips_with_one_and_two_faces_found"] == 2
    assert entry["time_ratio_two_over_one"]["mean"] == 1.75
    assert entry["stage_ratio_two_over_one"]["video_embed"] == 2.0


def test_speaker_split_summary_counts_by_plan():
    summarize = load_script("count_speaker_splits").summarize
    rows = [{"plan": "real_real_left", "speakers": 1}, {"plan": "real_real_left", "speakers": 2}, {"plan": "fake_av_real_left", "speakers": 1}]
    out = summarize(rows)
    assert out["split_into_two_or_more"] == 1 and out["by_plan"]["real_real_left"] == {"videos": 2, "split_into_two_or_more": 1}


def test_json_turn_counter_reads_distinct_speakers(tmp_path):
    counter = load_script("count_speaker_splits").speakers_in
    path = tmp_path / "v.json"
    path.write_text(json.dumps({"turns": [{"start": 0, "end": 1, "speaker": "speaker_0"}, {"start": 1, "end": 2, "speaker": "speaker_1"}, {"start": 2, "end": 3, "speaker": "speaker_0"}]}), encoding="utf-8")
    assert counter(path) == (2, 3)


def test_viral_case_summary_row_compares_the_verdict_with_the_claim():
    summary_row = load_script("score_viral_cases").summary_row
    case = {"case_id": "c1", "title": "t", "date": "2026-01-01", "claim": "Fake"}
    payload = {"video_p_fake": 0.8, "threshold": 0.5, "duration": 12.34, "subjects": [
        {"center_x": 0.7, "p_fake": 0.8, "alpha": 0.2}, {"center_x": 0.2, "p_fake": 0.1, "alpha": None}]}
    row = summary_row(case, payload, {"width": 640, "height": 360, "fps": 25.0})
    assert row["video_verdict"] == "fake" and row["agrees_with_claim"] == "true"
    assert row["p_fake_per_face"] == "0.1000;0.8000" and row["alpha_per_face"] == "-;0.200"
    unknown = summary_row({**case, "claim": ""}, payload, {"width": 1, "height": 1, "fps": 1})
    assert unknown["agrees_with_claim"] == "-"
    no_face = summary_row(case, {"video_p_fake": float("nan"), "threshold": 0.5, "duration": 3.0, "subjects": []}, {"width": 1, "height": 1, "fps": 1})
    assert no_face["video_verdict"] == "no_face" and no_face["agrees_with_claim"] == "-"
