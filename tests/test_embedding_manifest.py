from deepfake_detection.config import load_config
from deepfake_detection.data.embedding_manifest import build_embedding_manifest, manifest_totals
from deepfake_detection.data.subject_rules import subject_id


def test_subject_rules():
    assert subject_id("polyglotfake", "fake_ar_10_to_en_MicroTts_seg002") == "10"
    assert subject_id("hidf", "c00003_c11507_seg001") == "c00003"
    assert subject_id("celebdfv2", "id0_id16_0000_seg000") == "celeb_id0"
    assert subject_id("celebdfv2", "00000_seg000") == "celeb_yt_00000"
    assert subject_id("fakeavceleb", "00001_0_id06152_wavtolip_seg000") == "fac_00001"
    assert subject_id("ffpp", "deepfakes_000_003_seg004") == "ffpp_000"
    assert subject_id("ffpp", "000_seg000") == "ffpp_000"
    assert subject_id("ffpp", "deepfakedetection_01_02__walk_seg000") == "ffpp_dfd_01"


def test_manifest_from_embedding_folder(tmp_path):
    config = load_config()
    for folder, names in {"real": ["c00003_seg000", "c00003_seg001"], "fake": ["c00003_c11507_seg000"]}.items():
        (tmp_path / folder).mkdir()
        for name in names:
            (tmp_path / folder / f"{name}.npz").touch()
    manifest = build_embedding_manifest(config.dataset("hidf"), tmp_path, config.manifest_columns)
    assert list(manifest.columns) == config.manifest_columns
    assert manifest["sample_id"].tolist() == ["c00003", "c00003_c11507"]
    assert manifest["embedding_pattern"].tolist() == ["real/c00003_seg*.npz", "fake/c00003_c11507_seg*.npz"]
    assert manifest["manipulation_type"].tolist() == ["real", "video_only"]
    assert manifest_totals(manifest) == {"real": 2, "fake": 1, "segments": 3}


def test_paths_follow_layout():
    config = load_config()
    assert config.embedding_dir("ffpp").name == "embedding"
    assert config.manifest_path("ffpp").parts[-3:] == ("ffpp", "labels", "manifest.csv")


def test_results_folders():
    config = load_config()
    assert config.paths.results_dir("benchmark_inference_speed").parts[-3:] == ("results", "metrics", "benchmark_inference_speed")
    assert config.paths.predictions_dir("moe_combined_on_ffpp").parts[-3:-1] == ("results", "inferences")
    assert config.paths.analysis_dir("embedding_separation").parts[-3:-1] == ("results", "analyses")
    assert config.paths.xai_dir("gradcam_moe_combined_on_ffpp").parts[-3:-1] == ("results", "xai")
    assert config.paths.checkpoints_dir("moe_combined").parts[-2:] == ("models", "moe_combined")


def test_subjects_70_30_keeps_whole_subjects():
    from deepfake_detection.evaluation.reproduce import test_subjects_70_30
    records = [([], 0, f"s{i // 2}", "real", True) for i in range(20)]
    test = test_subjects_70_30(records, seed=42)
    assert {r[2] for r in test} and len(test) == 2 * int(10 * 0.30)
    assert all(sum(1 for r in test if r[2] == s) == 2 for s in {r[2] for r in test})
