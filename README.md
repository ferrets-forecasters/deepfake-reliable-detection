# Audio-visual deepfake detection with a gated Mixture-of-Experts fusion

Code behind the paper *Deteksi Deepfake Audio-Visual Menggunakan MoEFusionClassifier dengan Mekanisme Gate Modalitas Dinamis* (GEMASTIK XIX 2026, Karya Tulis Ilmiah, Telkom University). It holds the whole experiment pipeline: embedding extraction, training, scoring, the analyses of the paper, multi-person inference, and the check that recomputes every number of the result tables.

## What it does

A video is cut into 2-second segments. Frozen VideoMAE-base and Whisper-medium encoders give one feature vector per segment. An audio-visual branch and a visual branch each produce a score, and a small gate (4,417 parameters) turns the two branch scores into one weight α per video, so every verdict reports which branch drove it. The gate is trained with an auxiliary loss on the manipulation type of the training dataset (audio and video, or video only). The same model scores each face of a clip with several people: faces are tracked, each face gets its own speaker through ECAPA-TDNN diarization, and the video verdict is the maximum over people. A static concatenation of the same features is the baseline.

## Results of the paper

| Result | Value |
|---|---|
| Macro-F1, three seeds, gated fusion against static concatenation | 0.9215 ± 0.0046 against 0.9147 ± 0.0123 (paired t-test p = 0.35) |
| Macro-F1, five cross-validation folds | 0.9210 ± 0.0059 against 0.9193 ± 0.0088 (p = 0.40) |
| Two-person videos, video AUC, one face against one verdict per person | 0.7900 against 0.9190 |
| Real-time factor, RTX 4050 laptop GPU, 120 s clip | 0.132 (15.9 s) |

Each number is recomputed from saved predictions by `scripts/verify_paper_tables.py` (values in `configs/paper_tables.yaml`).

## Setup

Python 3.10 or newer, [uv](https://docs.astral.sh/uv/), and ffmpeg and ffprobe on `PATH` (or `FFMPEG_PATH`, `UNRAR_PATH`). The lock file pins the CUDA 12.8 build of PyTorch.

```
uv sync
uv run python -m pytest -q
```

The scripts that download archives or videos need the extra packages: `uv sync --extra download`. Face detection uses the official MediaPipe [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) model, placed at `<root>/external/mediapipe/face_landmarker.task` (see Project root).

## Project root and data layout

Paths resolve from a project root: the folder two levels above the clone, or the folder in `DEEPFAKE_PROJECT_ROOT`.

```
<root>/datasets/<name>/raw            downloaded dataset
<root>/datasets/<name>/embedding      one .npz per 2-second segment (video_feat 768, audio_feat 1024)
<root>/datasets/<name>/labels         manifests, folds, split files
<root>/models/<run>/                  checkpoints and results.yaml of one training run
<root>/results/inferences|metrics|analyses|xai
<root>/external/mediapipe/face_landmarker.task
<root>/temp/                          caches and working files
```

The five datasets are not included: PolyGlotFake, HiDF, FakeAVCeleb, Celeb-DF v2 and FaceForensics++. `configs/datasets.yaml` lists the access page and release of each. Every dataset keeps its own licence (HiDF is CC BY-NC 4.0). Checkpoints (`*.pt`) and embeddings are not in git.

Scripts that read dataset archives from a shared Google Drive take the file ids from `configs/drive_archives.local.yaml`, which is not tracked. Copy `configs/drive_archives.example.yaml` to that name and fill in the ids of your own copies.

## Pipeline

Run every command from the repository root.

| Step | Command | Output |
|---|---|---|
| Extract embeddings from the raw videos | `uv run python scripts/extract_embeddings.py --dataset <name>` (all five: `scripts/extract_all_datasets.py`) | `datasets/<name>/embedding/{real,fake}/*.npz` |
| Write the manifest of each dataset | `uv run python scripts/build_embedding_manifest.py` | `datasets/<name>/labels/manifest.csv` |
| Subject-level k-fold assignment | `uv run python scripts/make_cv_folds.py --folds 5 --seed 42` | `datasets/<name>/labels/cv_folds_k5_seed42.csv` |
| Train one model | `uv run python scripts/train_reference.py moe_fusion.yaml --name <run>` (static baseline: `concat.yaml`) | `models/<run>/` |
| Train every run of a queue, resumable | `uv run python scripts/train_queue.py --queue configs/training_queue.yaml --max-active 3` | `models/<run>/` for each entry |
| Score the test split of finished runs | `uv run python scripts/export_predictions.py --models <run>` | `results/inferences/<run>_on_<scope>/predictions.csv` |
| Re-score checkpoints and compare with their stored `results.yaml` | `uv run python scripts/reproduce_inference.py --models <run>` | one MATCH or MISMATCH line per run |
| Recompute the numbers of the paper | `uv run python scripts/verify_paper_tables.py` | one ok or MISMATCH line per number |
| Score videos with several people | `uv run python scripts/infer_multi_subject.py <video or folder> --model <run>` | one verdict per face in `results/inferences/` |
| Upload page for one video | `uv run python scripts/demo.py --serve --model <run>`, then open `http://127.0.0.1:8765/` | verdict per face in the browser |

The training configs are in `configs/training/` (`moe_fusion.yaml`, `concat.yaml`, the five leave-one-out configs, the single-branch ablations). `configs/training_queue*.yaml` list the runs of the paper with their overrides (seeds 42 to 44, folds 0 to 4, leave-one-out re-runs, ablations). `--override key.sub=value` changes any setting of a config.

Checks made on this code: `reproduce_inference.py` re-scored the exported FakeAVCeleb 70/30 checkpoint on its 5,574 test videos with macro-F1, AP, AUC and mean α equal to the stored values. Embeddings extracted again from a FakeAVCeleb clip match the stored ones exactly for audio and with cosine similarity 0.996 for video, because the face crop now comes from the current MediaPipe `FaceLandmarker` instead of the removed `FaceMesh` solution.

A second, smaller pipeline works on its own feature cache and is not the source of the paper's numbers: `extract_features.py`, `train.py` and `evaluate.py` (modules in `src/deepfake_detection/modeling`).

## Layout

| Path | Holds |
|---|---|
| `src/deepfake_detection/reference` | Data loading, models, losses and training loop that produced the paper's checkpoints, and the loader that scripts use |
| `src/deepfake_detection/modeling` | The gated model, the static baseline and the domain classifier of the second pipeline |
| `src/deepfake_detection/{data,encoders,experts,gate}` | Segment embedding, face tracking, tiling, frozen encoders, branches, gate |
| `src/deepfake_detection/inference` | Multi-person pipeline, diarization, upload page, rendering |
| `src/deepfake_detection/{training,evaluation,xai}` | Resumable training, cross-validation splits, scoring, Grad-CAM |
| `scripts` | One script per step or analysis, listed below |
| `configs` | Paths, datasets, training configs and queues, the numbers the paper prints |
| `tests` | Unit tests |

## Scripts

Every script prints its options with `--help`.

| Group | Scripts |
|---|---|
| Embeddings and manifests | `extract_embeddings`, `extract_all_datasets`, `build_embedding_manifest`, `build_feature_cache`, `build_manifests`, `extract_features`, `make_cv_folds`, `export_dataset_registry`, `fetch_official_splits` |
| Training | `train_reference`, `train_queue`, `train`, `stop_training` (Windows only), `watch_progress` |
| Scoring and result tables | `export_predictions`, `reproduce_inference`, `evaluate`, `analyze_training`, `compare_runs`, `calibration`, `prediction_curves`, `rerun_inference`, `run_remaining_checks`, `verify_paper_tables`, `find_values_in_paper` |
| Model analysis | `analyze_gate_branches`, `analyze_gate_vs_static`, `analyze_embeddings`, `expert_tests`, `audit_shortcuts`, `complexity_table`, `benchmark_inference`, `benchmark_faces`, `compression_sweep` |
| Data audits | `audit_duplicate_videos`, `audit_split_leakage`, `check_polyglot_split`, `plan_b_filtered_metrics`, `rescore_leave_one_out_dedup`, `count_partitions`, `check_dataset_releases` |
| Several people | `infer_multi_subject`, `build_tiled_set`, `build_conversation_set`, `build_reencoded_control`, `score_tiled_set`, `check_face_tracks`, `check_diarization`, `count_faces`, `count_speaker_splits`, `render_multi_subject`, `summarize_multi_subject`, `export_diarization_details` |
| Explanations | `explain_clip`, `explain_video`, `export_cam_coverage`, `export_audio_conditions` |
| Other test sets | `score_images`, `score_still_frames`, `score_categories`, `find_viral_videos`, `download_viral_cases`, `trim_viral_clips`, `make_contact_sheets`, `score_viral_cases` |
| Examples and figures data | `export_example_pairs`, `export_pair_frames`, `export_remote_example_pairs`, `export_face_detection`, `export_cross_language_transcripts` |
| Archive helpers | `download_drive_archives`, `fetch_test_videos`, `remote_zip`, `remote_rar`, `extract_partial_zip` |
| Demo | `demo` |

## Tests

```
uv run python -m pytest -q
```
