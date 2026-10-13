from __future__ import annotations

import random
import subprocess
from collections import defaultdict
from pathlib import Path

# FakeAVCeleb folder per clip kind: the first word is the video, the second the audio.
CATEGORY_DIRS = {
    "real": "RealVideo-RealAudio",
    "fake_av": "FakeVideo-FakeAudio",
    "fake_v": "FakeVideo-RealAudio",
    "fake_a": "RealVideo-FakeAudio",
}


def index_clips(raw_root: Path) -> dict[tuple[str, str, str], list[Path]]:
    """Clips grouped by (kind, race, gender). Layout: <raw_root>/<category>/<race>/<gender>/<id>/<clip>.mp4"""
    index: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for kind, category in CATEGORY_DIRS.items():
        for path in sorted((raw_root / category).glob("*/*/*/*.mp4")):
            race, gender = path.parts[-4], path.parts[-3]
            index[(kind, race, gender)].append(path)
    return index


def pick_pairs(
    index: dict[tuple[str, str, str], list[Path]], left_kind: str, right_kind: str, count: int, seed: int
) -> list[tuple[Path, Path]]:
    """Pairs of clips of the given kinds with the same race and gender and different speakers,
    so the two faces differ in identity but not in demographics."""
    rng = random.Random(seed)
    groups = sorted({(race, gender) for kind, race, gender in index if kind == left_kind} &
                    {(race, gender) for kind, race, gender in index if kind == right_kind})
    pairs: list[tuple[Path, Path]] = []
    attempts = 0
    while len(pairs) < count and attempts < count * 50:
        attempts += 1
        race, gender = rng.choice(groups)
        left = rng.choice(index[(left_kind, race, gender)])
        right = rng.choice(index[(right_kind, race, gender)])
        if left.parts[-2] != right.parts[-2] and (left, right) not in pairs:
            pairs.append((left, right))
    return pairs


LOSSY_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac"]
LOSSLESS_CODEC = ["-c:v", "ffv1", "-c:a", "pcm_s16le"]  # use with a .mkv output


def ffmpeg_tile_command(ffmpeg: Path, left: Path, right: Path, out: Path, audio_from: str, height: int, lossless: bool = False) -> list[str]:
    """Places the clips side by side at one height. Output ends with the shorter clip,
    and carries the audio of the chosen side."""
    audio_index = {"left": "0:a:0", "right": "1:a:0"}[audio_from]
    graph = (
        f"[0:v]scale=-2:{height},setsar=1,fps=25[l];[1:v]scale=-2:{height},setsar=1,fps=25[r];"
        "[l][r]hstack=inputs=2:shortest=1[v]"
    )
    return [
        str(ffmpeg), "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
        "-filter_complex", graph, "-map", "[v]", "-map", audio_index,
        *(LOSSLESS_CODEC if lossless else LOSSY_CODEC), "-shortest", str(out),
    ]


def tile_pair(ffmpeg: Path, left: Path, right: Path, out: Path, audio_from: str, height: int, lossless: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ffmpeg_tile_command(ffmpeg, left, right, out, audio_from, height, lossless), check=True)


def ffmpeg_turn_taking_command(ffmpeg: Path, left: Path, right: Path, out: Path, turn_seconds: float, height: int, lossless: bool = False) -> list[str]:
    """A two-person conversation: the left person talks for turn_seconds with the right face still,
    then the right person talks for turn_seconds with the left face still. The audio is the left
    clip's audio followed by the right clip's, so each speaker's voice is heard only in their turn."""
    t = turn_seconds
    scale = f"scale=-2:{height},setsar=1,fps=25"
    graph = (
        f"[0:v]{scale},trim=duration={t},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={t}[l];"
        f"[1:v]{scale},trim=duration={t},setpts=PTS-STARTPTS,tpad=start_mode=clone:start_duration={t}[r];"
        "[l][r]hstack=inputs=2:shortest=1[v];"
        f"[0:a]atrim=duration={t},asetpts=PTS-STARTPTS,aresample=16000[a0];"
        f"[1:a]atrim=duration={t},asetpts=PTS-STARTPTS,aresample=16000[a1];"
        "[a0][a1]concat=n=2:v=0:a=1[a]"
    )
    return [
        str(ffmpeg), "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        *(LOSSLESS_CODEC if lossless else LOSSY_CODEC), "-t", str(2 * t), str(out),
    ]


def conversation_pair(ffmpeg: Path, left: Path, right: Path, out: Path, turn_seconds: float, height: int, lossless: bool = False) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(ffmpeg_turn_taking_command(ffmpeg, left, right, out, turn_seconds, height, lossless), check=True)


def ffmpeg_reencode_command(ffmpeg: Path, clip: Path, out: Path, height: int) -> list[str]:
    """One clip through the same scale and encoder settings as the tiling, without a second person.
    Scores of this control against the original clip isolate what the re-encoding does."""
    return [
        str(ffmpeg), "-y", "-loglevel", "error", "-i", str(clip), "-vf", f"scale=-2:{height},setsar=1",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", str(out),
    ]
