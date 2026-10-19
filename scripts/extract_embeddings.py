"""
Unified feature-extraction pipeline - all four supported datasets.

Produces per-segment .npz files (2-second windows) containing:
    video_feat : float32 [768]   - VideoMAE-base mean-pooled embedding
    audio_feat : float32 [1024]  - Whisper-medium encoder mean-pooled embedding
    seg_idx    : int             - window index within the source video
    orig_path  : str             - absolute path to the source file
    duration   : float           - source duration in seconds

All embeddings land flat inside  <dataset>/embedding/real/  or  .../fake/
regardless of the raw folder hierarchy.

Usage (from the repository root, paths from deepfake_detection.reference.config)
--------------------------
Named datasets, reading datasets/<name>/raw and writing datasets/<name>/embedding:
    uv run python scripts/extract_embeddings.py --dataset polyglotfake
    uv run python scripts/extract_embeddings.py --dataset hidf
    uv run python scripts/extract_embeddings.py --dataset celebdfv2
    uv run python scripts/extract_embeddings.py --dataset fakeavceleb
    uv run python scripts/extract_embeddings.py --dataset ffpp

Custom path (must supply --output and --label):
    uv run python scripts/extract_embeddings.py --input /path/to/videos --output /path/to/embeddings --label fake

Optional flags:
    --ffmpeg-timeout 120    ffmpeg audio-decode timeout in seconds (default: 120)
    --probe-timeout  30     ffprobe info timeout in seconds (default: 30)

Failed files are logged to  <output_root>/failed_files.tsv  with reason codes:
    probe_timeout       ffprobe timed out getting video/audio info
    probe_error         ffprobe returned unusable output
    audio_timeout       ffmpeg timed out decoding audio
    audio_error         ffmpeg failed or returned empty audio
    frame_error         OpenCV failed to read frames
    too_short           video shorter than MIN_DURATION
"""
import sys
import argparse
import zipfile
import datetime
import numpy as np
import subprocess
from pathlib import Path

import torch
import cv2
import whisper
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel

# -- Path setup ---------------------------------------------------------------
from deepfake_detection.data.landmark_mesh import LandmarkMesh
from deepfake_detection.reference.config import (
    AUDIO_DIM, BASE, CELEBDFV2_ROOT, FAKEAVCELEB_ROOT, FFPP_ROOT, HIDF_ROOT, POLYGLOT_ROOT, VIDEO_DIM,
)

# -- Constants ----------------------------------------------------------------
WINDOW_SEC      = 2.0
VIDEOMAE_FRAMES = 16
WHISPER_SR      = 16000
MIN_DURATION    = 1.0
MP_DETECT_FPS   = 2
VIDEO_EXTS      = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS      = {".wav", ".flac", ".m4a", ".aac", ".ogg"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Dataset path registry ----------------------------------------------------
DATASET_PATHS = {
    "polyglotfake": (BASE / "datasets" / "polyglotfake" / "raw", POLYGLOT_ROOT),
    "hidf":         (BASE / "datasets" / "hidf" / "raw", HIDF_ROOT),
    "celebdfv2":    (BASE / "datasets" / "celebdfv2" / "raw", CELEBDFV2_ROOT),
    "fakeavceleb":  (BASE / "datasets" / "fakeavceleb" / "raw", FAKEAVCELEB_ROOT),
    "ffpp":         (BASE / "datasets" / "ffpp" / "raw", FFPP_ROOT),
}


# -- Failure logger -----------------------------------------------------------

def _log_failure(log_path: Path, video_path: Path, reason: str):
    """Append one failed-file record to the TSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"{ts}\t{reason}\t{video_path}\n")


# -- ffprobe / ffmpeg helpers -------------------------------------------------

def get_video_info(path, timeout=30):
    """Returns (fps, duration) or (None, None) on any failure."""
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=r_frame_rate,duration",
           "-of", "csv=p=0", str(path)]
    try:
        out   = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        lines = [l.strip() for l in out.stdout.strip().splitlines() if l.strip()]
        if not lines:
            return None, None
        parts    = lines[0].split(",")
        num, den = map(float, parts[0].split("/"))
        return num / den, float(parts[1])
    except subprocess.TimeoutExpired:
        return None, "probe_timeout"
    except Exception:
        return None, "probe_error"


def get_audio_duration(path, timeout=30):
    """Returns duration in seconds or None on failure."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "csv=p=0", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return float(out.stdout.strip())
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def load_audio(path, timeout=120):
    """
    Decode audio to float32 mono at WHISPER_SR.
    Returns numpy array or None on timeout/error.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-f", "f32le", "-ac", "1", "-ar", str(WHISPER_SR), "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout)
        arr = np.frombuffer(out.stdout, dtype=np.float32)
        if len(arr) == 0:
            return None
        return arr
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


# -- Frame extraction + face crop ---------------------------------------------

def read_and_crop_frames(path, fps, face_mesh):
    """Returns list of cropped RGB frames, or None on read failure."""
    try:
        cap    = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    except Exception:
        return None

    if not frames:
        return None

    detect_interval = max(1, int(fps / MP_DETECT_FPS))
    detected_boxes  = {}
    for i in range(0, len(frames), detect_interval):
        results = face_mesh.process(frames[i])
        if results.multi_face_landmarks:
            h, w, _ = frames[i].shape
            ls       = results.multi_face_landmarks[0].landmark
            xs       = [int(l.x * w) for l in ls]
            ys       = [int(l.y * h) for l in ls]
            detected_boxes[i] = (
                max(min(xs) - 15, 0), max(min(ys) - 15, 0),
                min(max(xs) + 15, w), min(max(ys) + 15, h),
            )

    def interpolate_box(idx):
        keys = sorted(detected_boxes.keys())
        if not keys:
            return None
        lo = max((k for k in keys if k <= idx), default=keys[0])
        hi = min((k for k in keys if k >= idx), default=keys[-1])
        if lo == hi:
            return detected_boxes[lo]
        t  = (idx - lo) / (hi - lo)
        b0, b1 = detected_boxes[lo], detected_boxes[hi]
        return tuple(int(b0[j] + t * (b1[j] - b0[j])) for j in range(4))

    cropped = []
    for i, frame in enumerate(frames):
        box = interpolate_box(i)
        if box:
            x1, y1, x2, y2 = box
            cropped.append(cv2.resize(frame[y1:y2, x1:x2], (224, 224)))
        else:
            cropped.append(cv2.resize(frame, (224, 224)))
    return cropped


# -- Feature extraction -------------------------------------------------------

def extract_video_feat(cropped_frames, start, fps, processor, vmae_model):
    start_f = int(start * fps)
    end_f   = int((start + WINDOW_SEC) * fps)
    chunk   = cropped_frames[start_f:end_f]
    while len(chunk) < VIDEOMAE_FRAMES:
        chunk.append(chunk[-1] if chunk else np.zeros((224, 224, 3), np.uint8))
    idx     = np.linspace(0, len(chunk) - 1, VIDEOMAE_FRAMES).astype(int)
    sampled = [chunk[i] for i in idx]
    inputs  = processor(sampled, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        return vmae_model(**inputs).last_hidden_state.squeeze(0).mean(dim=0).float().cpu().numpy()


def extract_audio_feat(audio, start, whisper_model):
    start_s  = int(start * WHISPER_SR)
    end_s    = int((start + WINDOW_SEC) * WHISPER_SR)
    chunk    = audio[start_s:end_s].astype(np.float32)
    n_target = int(WINDOW_SEC * WHISPER_SR)
    if len(chunk) < n_target:
        chunk = np.pad(chunk, (0, n_target - len(chunk)))
    chunk        = chunk[:n_target]
    mel          = whisper.log_mel_spectrogram(chunk).to(DEVICE)
    n_mel_needed = 3000
    if mel.shape[-1] < n_mel_needed:
        mel = torch.nn.functional.pad(mel, (0, n_mel_needed - mel.shape[-1]))
    mel = mel[..., :n_mel_needed]
    with torch.no_grad():
        return whisper_model.encoder(mel.unsqueeze(0)).squeeze(0).mean(dim=0).float().cpu().numpy()


# -- Per-file processors ------------------------------------------------------

def process_video(path, out_dir, face_mesh, processor, vmae_model, whisper_model,
                  fail_log, probe_timeout=30, ffmpeg_timeout=120, stem_prefix="",
                  skip_audio=False, legacy_prefix=""):
    """
    Extract all 2-second segment embeddings from a video file.
    stem_prefix   : prepended to output filename (FF++ method tagging)
    legacy_prefix : if set, also treat legacy-named segments as already done.
                    Used for PolyglotFake where old extractions wrote
                    real_ar_1_seg000.npz but new naming is ar_1_seg000.npz.
    Returns (n_windows, skipped).  On any unrecoverable error: (0, True).
    """
    result = get_video_info(path, timeout=probe_timeout)

    if isinstance(result[1], str):
        _log_failure(fail_log, path, result[1])
        return 0, True
    fps, duration = result
    if fps is None or duration is None:
        _log_failure(fail_log, path, "probe_error")
        return 0, True
    if duration < MIN_DURATION:
        _log_failure(fail_log, path, "too_short")
        return 0, True

    n_windows = int(duration // WINDOW_SEC)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem        = f"{stem_prefix}{path.stem}" if stem_prefix else path.stem
    legacy_stem = f"{legacy_prefix}{path.stem}" if legacy_prefix else None

    def _seg_exists(i):
        if (out_dir / f"{stem}_seg{i:03d}.npz").exists():
            return True
        if legacy_stem and (out_dir / f"{legacy_stem}_seg{i:03d}.npz").exists():
            return True
        return False

    missing = [i for i in range(n_windows) if not _seg_exists(i)]
    if not missing:
        return n_windows, False

    cropped = read_and_crop_frames(path, fps, face_mesh)
    if cropped is None:
        _log_failure(fail_log, path, "frame_error")
        return 0, True

    if skip_audio:
        audio = None
    else:
        audio = load_audio(path, timeout=ffmpeg_timeout)
        if audio is None:
            _log_failure(fail_log, path, "audio_error")
            return 0, True

    for seg_idx in tqdm(missing, desc=stem, unit="seg", leave=False):
        out_path   = out_dir / f"{stem}_seg{seg_idx:03d}.npz"
        start      = seg_idx * WINDOW_SEC
        video_feat = extract_video_feat(cropped, start, fps, processor, vmae_model)
        audio_feat = (np.zeros(AUDIO_DIM, dtype=np.float32) if audio is None
                      else extract_audio_feat(audio, start, whisper_model))
        np.savez_compressed(
            out_path,
            video_feat=video_feat,
            audio_feat=audio_feat,
            seg_idx=seg_idx,
            orig_path=str(path),
            duration=duration,
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return n_windows, False


def process_audio_only(path, out_dir, whisper_model,
                       fail_log, probe_timeout=30, ffmpeg_timeout=120, stem_prefix=""):
    """Audio-only files: video_feat is zeroed (no face track)."""
    duration = get_audio_duration(path, timeout=probe_timeout)
    if duration is None or duration < MIN_DURATION:
        _log_failure(fail_log, path, "probe_error" if duration is None else "too_short")
        return 0, True

    n_windows = int(duration // WINDOW_SEC)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{stem_prefix}{path.stem}" if stem_prefix else path.stem

    missing = [i for i in range(n_windows)
               if not (out_dir / f"{stem}_seg{i:03d}.npz").exists()]
    if not missing:
        return n_windows, False

    audio = load_audio(path, timeout=ffmpeg_timeout)
    if audio is None:
        _log_failure(fail_log, path, "audio_timeout")
        return 0, True

    for seg_idx in tqdm(missing, desc=stem, unit="seg", leave=False):
        out_path   = out_dir / f"{stem}_seg{seg_idx:03d}.npz"
        start      = seg_idx * WINDOW_SEC
        audio_feat = extract_audio_feat(audio, start, whisper_model)
        np.savez_compressed(
            out_path,
            video_feat=np.zeros(VIDEO_DIM, dtype=np.float32),
            audio_feat=audio_feat,
            seg_idx=seg_idx,
            orig_path=str(path),
            duration=duration,
        )

    return n_windows, False


# -- Dataset-specific task collectors ----------------------------------------
# Each returns 5-tuples: (source_path, output_dir, stem_prefix, skip_audio, legacy_prefix)
#   output_dir     : embeddings/real/ or embeddings/fake/ (always flat)
#   stem_prefix    : prepended to output filename (FF++ method tagging)
#   skip_audio     : True for FF++ and CelebDFv2 (no meaningful audio)
#   legacy_prefix  : if old extractions used a different prefix, check both
#                    e.g. PolyglotFake old format: real_ar_1_seg000.npz

def _glob_media(root: Path):
    return sorted(f for f in root.rglob("*")
                  if f.suffix.lower() in VIDEO_EXTS | AUDIO_EXTS)


def collect_polyglotfake(input_root, output_root):
    # Legacy format: real_ar_1_seg000.npz / fake_ar_10_to_en_MicroTts_seg000.npz
    # New format:    ar_1_seg000.npz      / ar_10_to_en_MicroTts_seg000.npz
    tasks = []
    for label, subdir, legacy in [
        ("real", "real-video", "real_"),
        ("fake", "fake-video", "fake_"),
    ]:
        src = input_root / subdir
        if not src.exists():
            print(f"  [polyglotfake] WARNING: expected folder not found: {src}")
            continue
        for f in _glob_media(src):
            tasks.append((f, output_root / label, "", False, legacy))
    return tasks


def collect_hidf(input_root, output_root):
    tasks = []
    for label, zip_name in [("real", "Real-vid.zip"), ("fake", "Fake-vid.zip")]:
        extracted = input_root / label
        zip_path  = input_root / zip_name
        if not extracted.exists():
            if zip_path.exists():
                print(f"  [hidf] Extracting {zip_name} -> {extracted} ...")
                with zipfile.ZipFile(zip_path) as zf:
                    zf.extractall(extracted)
                print(f"  [hidf] Extraction complete.")
            else:
                print(f"  [hidf] WARNING: neither {extracted} nor {zip_path} found - skipping {label}")
                continue
        for f in _glob_media(extracted):
            tasks.append((f, output_root / label, "", False, ""))
    return tasks


def collect_celebdfv2(input_root, output_root):
    # CelebDFv2 has no audio track - skip_audio=True zeros audio_feat
    tasks = []
    for subdir in ["Celeb-real", "YouTube-real"]:
        src = input_root / subdir
        if src.exists():
            for f in _glob_media(src):
                tasks.append((f, output_root / "real", "", True, ""))
        else:
            print(f"  [celebdfv2] WARNING: folder not found (skipping): {src}")
    for subdir in ["Celeb-synthesis"]:
        src = input_root / subdir
        if src.exists():
            for f in _glob_media(src):
                tasks.append((f, output_root / "fake", "", True, ""))
        else:
            print(f"  [celebdfv2] WARNING: folder not found (skipping): {src}")
    return tasks


def collect_fakeavceleb(input_root, output_root):
    mapping = {
        "RealVideo-RealAudio": "real",
        "FakeVideo-FakeAudio": "fake",
        "FakeVideo-RealAudio": "fake",
        "RealVideo-FakeAudio": "fake",
    }
    tasks      = []
    found_real = False
    for subdir, label in mapping.items():
        src = input_root / subdir
        if src.exists():
            if label == "real":
                found_real = True
            for f in _glob_media(src):
                tasks.append((f, output_root / label, "", False, ""))
    if not found_real:
        print("  [fakeavceleb] WARNING: RealVideo-RealAudio not found.")
        print("                Processing fakes only.\n")
    return tasks


def collect_ffpp(input_root, output_root):
    """
    FF++ structure (dataset-ff++/raw/ contains all method folders):
        original/           -> embeddings/real/   (no prefix)
        Deepfakes/          -> embeddings/fake/   prefix: deepfakes_
        Face2Face/          -> embeddings/fake/   prefix: face2face_
        FaceShifter/        -> embeddings/fake/   prefix: faceshifter_
        FaceSwap/           -> embeddings/fake/   prefix: faceswap_
        NeuralTextures/     -> embeddings/fake/   prefix: neuraltextures_
        DeepFakeDetection/  -> embeddings/fake/   prefix: deepfakedetection_

    Fake files from different methods share identical filenames (e.g. 000_003.mp4),
    so the method prefix is mandatory to avoid collisions in the flat fake/ folder.

    NOTE: FF++ has no meaningful audio. Whisper will encode silence/background
    to near-identical embeddings across all files. The gate should produce α≈1.0
    (route to VisualArtifactExpert) for this dataset.
    """
    REAL_DIRS = ["original"]
    FAKE_DIRS = [
        ("Deepfakes",         "deepfakes_"),
        ("Face2Face",         "face2face_"),
        ("FaceShifter",       "faceshifter_"),
        ("FaceSwap",          "faceswap_"),
        ("NeuralTextures",    "neuraltextures_"),
        ("DeepFakeDetection", "deepfakedetection_"),
    ]

    tasks = []
    for subdir in REAL_DIRS:
        src = input_root / subdir
        if src.exists():
            for f in _glob_media(src):
                tasks.append((f, output_root / "real", "", True, ""))
        else:
            print(f"  [ff++] WARNING: {src} not found - skipping real")

    for subdir, prefix in FAKE_DIRS:
        src = input_root / subdir
        if src.exists():
            for f in _glob_media(src):
                tasks.append((f, output_root / "fake", prefix, True, ""))
        else:
            print(f"  [ff++] WARNING: {src} not found - skipping {subdir}")

    return tasks


DATASET_COLLECTORS = {
    "polyglotfake": collect_polyglotfake,
    "hidf":         collect_hidf,
    "celebdfv2":    collect_celebdfv2,
    "fakeavceleb":  collect_fakeavceleb,
    "ffpp":         collect_ffpp,
}


# -- Entry point --------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract VideoMAE + Whisper embeddings -> real/ and fake/ folders"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", choices=list(DATASET_PATHS.keys()))
    group.add_argument("--input",   type=Path)
    parser.add_argument("--output",          type=Path,  default=None)
    parser.add_argument("--label",           choices=["real", "fake"], default=None)
    parser.add_argument("--ffmpeg-timeout",  type=int,   default=120,
                        help="ffmpeg audio decode timeout in seconds (default: 120)")
    parser.add_argument("--probe-timeout",   type=int,   default=30,
                        help="ffprobe info timeout in seconds (default: 30)")
    args = parser.parse_args()

    # -- Resolve paths and task list ------------------------------------------
    if args.dataset:
        input_root, output_root = DATASET_PATHS[args.dataset]
        tasks = DATASET_COLLECTORS[args.dataset](input_root, output_root)
    else:
        if args.output is None:
            parser.error("--output is required with --input")
        if args.label is None:
            parser.error("--label real|fake is required with --input")
        input_root  = args.input
        output_root = args.output
        out_dir     = output_root / args.label
        tasks       = [(f, out_dir, "", False, "") for f in _glob_media(input_root)]

    fail_log = output_root / "failed_files.tsv"

    # -- Summary --------------------------------------------------------------
    print(f"Device          : {DEVICE}")
    if args.dataset:
        n_real    = sum(1 for _, d, _, _, _ in tasks if d.name == "real")
        n_fake    = sum(1 for _, d, _, _, _ in tasks if d.name == "fake")
        n_noaudio = sum(1 for _, _, _, sa, _ in tasks if sa)
        print(f"Dataset         : {args.dataset}")
        print(f"Files           : {len(tasks)}  ({n_real} real / {n_fake} fake)")
        if n_noaudio:
            print(f"Audio           : zeroed for all {n_noaudio} files (no audio stream)")
    else:
        print(f"Input           : {input_root}")
        print(f"Output          : {output_root / args.label}")
        print(f"Files           : {len(tasks)}")
    print(f"ffmpeg timeout  : {args.ffmpeg_timeout}s")
    print(f"ffprobe timeout : {args.probe_timeout}s")
    print(f"Failure log     : {fail_log}")
    print()

    if not tasks:
        print("No files found - check dataset paths and folder names.")
        return

    # -- Load models ----------------------------------------------------------
    print("Loading models (VideoMAE-base + Whisper-medium)...")
    processor     = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    vmae_model    = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(DEVICE).eval()
    whisper_model = whisper.load_model("medium").to(DEVICE).eval()
    face_mesh     = LandmarkMesh()
    print("Models loaded.\n")

    # -- Pre-flight check -----------------------------------------------------
    # Run the full pipeline on 3 random files before committing to the full run.
    # Each sample file is extracted to a temp folder and the output npz is verified.
    # If any sample fails, the script aborts with a clear error.
    import random, tempfile, shutil
    PREFLIGHT_N = min(3, len(tasks))
    print(f"Pre-flight check - testing {PREFLIGHT_N} random files before full run...")
    sample_tasks = random.sample(tasks, PREFLIGHT_N)
    preflight_ok = True

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        tmp_path = Path(tmpdir)
        for path, _, stem_prefix, skip_audio, legacy_prefix in sample_tasks:
            tmp_out = tmp_path / "out"
            stem    = f"{stem_prefix}{path.stem}" if stem_prefix else path.stem
            tmp_fail_log = tmp_path / "fail.tsv"

            n, failed = process_video(
                path, tmp_out, face_mesh, processor, vmae_model, whisper_model,
                fail_log=tmp_fail_log,
                probe_timeout=args.probe_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
                stem_prefix=stem_prefix,
                skip_audio=skip_audio,
                legacy_prefix=legacy_prefix,
            ) if path.suffix.lower() in VIDEO_EXTS else process_audio_only(
                path, tmp_out, whisper_model,
                fail_log=tmp_fail_log,
                probe_timeout=args.probe_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
                stem_prefix=stem_prefix,
            )

            if failed or n == 0:
                reason = "unknown"
                if tmp_fail_log.exists():
                    reason = tmp_fail_log.read_text().strip().split("\t")[1]
                print(f"  [failed] FAIL  [{reason}]  {path.name}")
                preflight_ok = False
                continue

            # Verify at least one output npz has correct shapes
            npz_files = list(tmp_out.glob("*.npz"))
            sample_npz = npz_files[0] if npz_files else None
            if sample_npz is None:
                print(f"  [failed] FAIL  [no output npz created]  {path.name}")
                preflight_ok = False
                continue

            data = np.load(sample_npz, allow_pickle=True)
            vf   = data.get("video_feat")
            af   = data.get("audio_feat")
            data.close()   # release file handle before Windows tries to delete the temp dir
            shape_ok = (vf is not None and vf.shape == (VIDEO_DIM,) and
                        af is not None and af.shape == (AUDIO_DIM,))
            if not shape_ok:
                print(f"  [failed] FAIL  [wrong output shape: video={getattr(vf,'shape',None)} "
                      f"audio={getattr(af,'shape',None)}]  {path.name}")
                preflight_ok = False
                continue

            has_nan = not (np.all(np.isfinite(vf)) and np.all(np.isfinite(af)))
            if has_nan:
                print(f"  [failed] FAIL  [NaN/Inf in output]  {path.name}")
                preflight_ok = False
                continue

            audio_note = " (audio zeroed - no audio stream)" if skip_audio else ""
            print(f"  [ok] OK    {path.name}  ->  {n} segment(s){audio_note}")

            # Clean up between samples so each is independent
            shutil.rmtree(tmp_out, ignore_errors=True)

    if not preflight_ok:
        print("\n[ABORT] Pre-flight check failed - fix the issues above before running the full extraction.")
        print(f"        Failure details logged to: {fail_log}")
        return

    print(f"Pre-flight passed - starting full extraction of {len(tasks)} files.\n")

    # -- Process --------------------------------------------------------------
    total_segs, n_skipped, n_failed = 0, 0, 0
    EARLY_CHECK = 20   # after this many files, warn if failure rate is high

    for i, (path, out_dir, stem_prefix, skip_audio, legacy_prefix) in enumerate(
            tqdm(tasks, desc="Files", unit="file")):

        if path.suffix.lower() in VIDEO_EXTS:
            n, skip = process_video(
                path, out_dir, face_mesh, processor, vmae_model, whisper_model,
                fail_log=fail_log,
                probe_timeout=args.probe_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
                stem_prefix=stem_prefix,
                skip_audio=skip_audio,
                legacy_prefix=legacy_prefix,
            )
        else:
            n, skip = process_audio_only(
                path, out_dir, whisper_model,
                fail_log=fail_log,
                probe_timeout=args.probe_timeout,
                ffmpeg_timeout=args.ffmpeg_timeout,
                stem_prefix=stem_prefix,
            )
        total_segs += n
        if skip:
            if n == 0:
                n_failed += 1
            else:
                n_skipped += 1

        # Early failure warning - catch catastrophic misconfig before wasting hours
        if i + 1 == EARLY_CHECK:
            fail_rate = n_failed / EARLY_CHECK
            if fail_rate >= 0.5:
                tqdm.write(f"\n[WARNING] {n_failed}/{EARLY_CHECK} files failed in the "
                           f"first {EARLY_CHECK} files ({fail_rate*100:.0f}% failure rate).")
                tqdm.write(f"[WARNING] Check {fail_log} for reason codes.")
                tqdm.write(f"[WARNING] Stopping early - fix the issue before continuing.\n")
                break

    # -- Final report ---------------------------------------------------------
    print(f"\n{'-'*50}")
    print(f"Done.")
    print(f"  Segments written : {total_segs}")
    print(f"  Already done     : {n_skipped}")
    print(f"  Failed / skipped : {n_failed}")
    if n_failed > 0:
        print(f"  Failure log      : {fail_log}")
        print(f"  Re-run the same command to retry - already-done segments are skipped.")
    print(f"{'-'*50}")


if __name__ == "__main__":
    main()
