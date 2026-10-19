from __future__ import annotations

import re
import subprocess
from pathlib import Path

from deepfake_detection.utils.tools import find_tool


def list_devices_command(ffmpeg: Path | None = None) -> list[str]:
    return [str(ffmpeg or find_tool("ffmpeg")), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"]


def parse_devices(ffmpeg_output: str) -> dict[str, list[str]]:
    """Names of the DirectShow video and audio devices from the output of ffmpeg -list_devices."""
    found: dict[str, list[str]] = {"video": [], "audio": []}
    for name, kind in re.findall(r'"([^"]+)" \((video|audio)\)', ffmpeg_output):
        found[kind].append(name)
    return found


def list_devices(ffmpeg: Path | None = None) -> dict[str, list[str]]:
    result = subprocess.run(list_devices_command(ffmpeg), capture_output=True)
    return parse_devices(decode_console(result.stderr + result.stdout))


def decode_console(data: bytes) -> str:
    """ffmpeg writes device names in UTF-8 or in the Windows ANSI code page, depending on the build."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("mbcs", errors="replace")


def record_command(ffmpeg: Path, video_device: str, audio_device: str, seconds: float, out_path: Path, size: str | None = None, fps: int = 25) -> list[str]:
    """Records a chunk of camera and microphone into an mp4 with 16 kHz mono audio."""
    command = [str(ffmpeg), "-y", "-loglevel", "error", "-f", "dshow", "-rtbufsize", "200M", "-framerate", str(fps)]
    if size:
        command += ["-video_size", size]
    command += [
        "-i", f"video={video_device}:audio={audio_device}", "-t", str(seconds),
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "16000", "-ac", "1", str(out_path),
    ]
    return command


def record_chunk(ffmpeg: Path, video_device: str, audio_device: str, seconds: float, out_path: Path, size: str | None = None) -> Path:
    subprocess.run(record_command(ffmpeg, video_device, audio_device, seconds, out_path, size), check=True)
    return out_path
