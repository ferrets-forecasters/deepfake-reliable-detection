from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import asdict
from pathlib import Path


from deepfake_detection.config import load_config
from deepfake_detection.config.config import _project_root
from deepfake_detection.data.face_crop import MultiFaceTracker
from deepfake_detection.data.segment_embedding import SegmentEmbedder
from deepfake_detection.evaluation.checkpoint_scoring import CheckpointScorer
from deepfake_detection.inference import MultiSubjectInference, SingleSpeaker
from deepfake_detection.inference.camera import list_devices, record_chunk
from deepfake_detection.inference.summary import RollingVerdict, format_result
from deepfake_detection.reference import load_reference
from deepfake_detection.utils.gpu import cap_gpu_memory
from deepfake_detection.utils.tools import find_tool


def build_pipeline(args) -> MultiSubjectInference:
    config = load_config(args.config)
    reference = load_reference()
    if args.no_diarization:
        diarizer = SingleSpeaker()
    else:
        from deepfake_detection.inference.diarization import EcapaDiarizer

        diarizer = EcapaDiarizer()
    return MultiSubjectInference(
        tracker=MultiFaceTracker(args.detector_fps, max_faces=args.max_faces),
        embedder=SegmentEmbedder(),
        scorer=CheckpointScorer(reference, config.paths.checkpoints_dir(args.model)),
        diarizer=diarizer,
        threshold=args.threshold,
    )


def save_result(result, source: str, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source, "video_p_fake": result.video_p_fake, "flagged": result.flagged_tracks, "duration": result.duration,
        "turns": [asdict(t) for t in result.turns], "subjects": [asdict(s) for s in result.subjects],
    }
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_video(pipeline, path: Path, out_dir: Path) -> None:
    started = time.time()
    result = pipeline.run(path)
    print(format_result(result, str(path)))
    print(f"time: {time.time() - started:.1f} s")
    save_result(result, str(path), out_dir, path.stem)


def run_camera(pipeline, args, out_dir: Path) -> None:
    devices = list_devices()
    video_device = args.video_device or (devices["video"][0] if devices["video"] else None)
    audio_device = args.audio_device or (devices["audio"][0] if devices["audio"] else None)
    if not video_device or not audio_device:
        raise SystemExit(f"no camera or microphone found: {devices}")
    print(f"camera: {video_device}\nmicrophone: {audio_device}\nchunks of {args.seconds} s, Ctrl-C stops")
    rolling = RollingVerdict(args.rolling)
    with tempfile.TemporaryDirectory(dir=_project_root() / "temp") as tmp:
        chunk = 0
        try:
            while args.chunks == 0 or chunk < args.chunks:
                path = Path(tmp) / f"chunk_{chunk:04d}.mp4"
                record_chunk(find_tool("ffmpeg"), video_device, audio_device, args.seconds, path, args.size)
                started = time.time()
                result = pipeline.run(path)
                print(format_result(result, f"chunk {chunk}"))
                print(f"time: {time.time() - started:.1f} s, rolling p_fake per face (left to right): {[round(v, 4) for v in rolling.update(result)]}")
                save_result(result, f"camera chunk {chunk}", out_dir, f"camera_{int(started)}_{chunk:04d}")
                chunk += 1
        except KeyboardInterrupt:
            print("stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake verdict per face from a video file or the camera.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", nargs="+", help="video files to score")
    source.add_argument("--serve", action="store_true", help="open the upload demo: a local page where a video file is uploaded and scored")
    source.add_argument("--camera", action="store_true", help="record chunks from the camera and microphone and score each one")
    source.add_argument("--list-devices", action="store_true", help="print the camera and microphone names and exit")
    parser.add_argument("--host", default="127.0.0.1", help="address of the upload demo")
    parser.add_argument("--port", type=int, default=8765, help="port of the upload demo")
    parser.add_argument("--keep-uploads", action="store_true", help="keep each uploaded video in temp/demo_uploads instead of deleting it after scoring")
    parser.add_argument("--seconds", type=float, default=10.0, help="camera chunk length, at least 2")
    parser.add_argument("--chunks", type=int, default=0, help="camera chunks to run, 0 until Ctrl-C")
    parser.add_argument("--rolling", type=int, default=5, help="camera chunks in the rolling mean")
    parser.add_argument("--video-device", default=None)
    parser.add_argument("--audio-device", default=None)
    parser.add_argument("--size", default=None, help="camera frame size, for example 640x480")
    parser.add_argument("--model", default="moe_combined")
    parser.add_argument("--max-faces", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--detector-fps", type=float, default=8.0, help="face detections per second, lip-motion linking needs more than 2")
    parser.add_argument("--no-diarization", action="store_true", help="give every face the full audio")
    parser.add_argument("--name", default="demo", help="dataset name used in the output folder")
    parser.add_argument("--skip-done", action="store_true", help="skip a video whose result file already exists, to resume a long list")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None, help="cap this process's CUDA memory, for running beside trainings")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.list_devices:
        for kind, names in list_devices().items():
            print(kind + ":", *names, sep="\n  ")
        return

    cap_gpu_memory(args.gpu_memory_fraction)
    pipeline = build_pipeline(args)
    config = load_config(args.config)
    out_dir = config.paths.predictions_dir(f"{args.model}_on_{args.name}")
    if args.serve:
        from deepfake_detection.inference.upload_server import serve

        serve(pipeline, args.host, args.port, _project_root() / "temp" / "demo_uploads", out_dir, args.keep_uploads)
        return
    if args.camera:
        run_camera(pipeline, args, out_dir)
        return
    for path in map(Path, args.video):
        if args.skip_done and (out_dir / f"{path.stem}.json").exists():
            continue
        run_video(pipeline, path, out_dir)


if __name__ == "__main__":
    main()
