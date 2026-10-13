import numpy as np

from deepfake_detection.training.reference_training import threaded_norm_stats


def test_threaded_norm_stats_match_numpy(tmp_path):
    rng = np.random.default_rng(0)
    video = rng.normal(2.0, 3.0, size=(30, 768)).astype(np.float32)
    audio = rng.normal(-1.0, 0.5, size=(30, 1024)).astype(np.float32)
    records = []
    for index in range(0, 30, 5):
        files = []
        for k in range(index, index + 5):
            path = tmp_path / f"{k}.npz"
            np.savez(path, video_feat=video[k], audio_feat=audio[k])
            files.append(path)
        records.append((files, 0, "subject", "real", True))
    stats = threaded_norm_stats(records, workers=4)
    assert np.allclose(stats["video_mean"], video.mean(0), atol=1e-5)
    assert np.allclose(stats["video_std"], video.std(0), atol=1e-4)
    assert np.allclose(stats["audio_mean"], audio.mean(0), atol=1e-5)
    assert np.allclose(stats["audio_std"], audio.std(0), atol=1e-4)


def test_numpy_stand_in_serves_cached_arrays_and_falls_back(tmp_path):
    from deepfake_detection.training.reference_training import _NumpyWithFeatureCache

    cached_path = tmp_path / "cached.npz"
    disk_path = tmp_path / "disk.npz"
    np.savez(disk_path, video_feat=np.ones(3, dtype=np.float32), audio_feat=np.zeros(2, dtype=np.float32))
    stand_in = _NumpyWithFeatureCache({str(cached_path): {"video_feat": np.full(3, 7.0), "audio_feat": np.full(2, 5.0)}})
    assert stand_in.load(cached_path, allow_pickle=True)["video_feat"][0] == 7.0
    with stand_in.load(disk_path, allow_pickle=True) as data:
        assert data["video_feat"][0] == 1.0
    assert stand_in.zeros(2).shape == (2,)


def test_shared_cache_serves_the_same_arrays_and_is_built_once(tmp_path):
    from deepfake_detection.training.reference_training import _shared_cache

    rng = np.random.default_rng(0)
    expected = {}
    for k in range(6):
        path = tmp_path / f"{k}.npz"
        expected[str(path)] = (rng.normal(size=768).astype(np.float32), rng.normal(size=1024).astype(np.float32))
        np.savez(path, video_feat=expected[str(path)][0], audio_feat=expected[str(path)][1])
    missing = str(tmp_path / "missing.npz")
    paths = sorted([*expected, missing])
    folder = tmp_path / "cache"
    first = _shared_cache(paths, folder, workers=2)
    assert len(first) == 6 and first.get(missing) is None
    built = sorted(p.name for p in folder.iterdir())
    second = _shared_cache(paths, folder, workers=2)
    assert sorted(p.name for p in folder.iterdir()) == built
    for path, (video, audio) in expected.items():
        assert np.array_equal(second.get(path)["video_feat"], video)
        assert np.array_equal(second.get(path)["audio_feat"], audio)
