from __future__ import annotations

import argparse
import statistics
import time

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from deepfake_detection.config import load_config
from deepfake_detection.reference import load_reference

VIDEO_DIM, AUDIO_DIM = 768, 1024


def gflops(fn) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 1e9


def median_ms(fn, repeats: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameters, GFLOP and time of the audio-visual branch, the visual branch and the gate of the gated classifier, for a clip of N segments, so each can be reported per 2-second segment.")
    parser.add_argument("--model", default="moe_combined_retrain")
    parser.add_argument("--segments", type=int, default=60)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    config = load_config()
    meta = yaml.safe_load((config.paths.checkpoints_dir(args.model) / "results.yaml").read_text(encoding="utf-8"))
    model = load_reference().build_model(meta["model"]).eval()
    n = args.segments
    mask, lengths = torch.ones(1, n, dtype=torch.bool), torch.tensor([n])
    video, audio = torch.randn(1, n, VIDEO_DIM), torch.randn(1, n, AUDIO_DIM)
    scores = torch.zeros(1)
    parts = {
        "audio_visual_branch": (model.sync_expert, lambda m, v, a, mk, ln: m(a, v, mk, ln)),
        "visual_branch": (model.visual_expert, lambda m, v, a, mk, ln: m(v, mk)),
        "gate": (model.gate, lambda m, v, a, mk, ln: m(scores.to(v.device), scores.to(v.device))),
    }
    report = {"model": args.model, "segments": n, "device": torch.cuda.get_device_name(0)}
    for name, (module, call) in parts.items():
        cpu_args = (module, video, audio, mask, lengths)
        flops = gflops(lambda: call(*cpu_args))
        module = module.cuda()
        gpu_args = (module, video.cuda(), audio.cuda(), mask.cuda(), lengths.cuda())
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            ms = median_ms(lambda: call(*gpu_args), args.repeats)
        module.cpu()
        report[name] = {"parameters": sum(p.numel() for p in module.parameters()), "gflop_total": round(flops, 4), "gflop_per_segment": round(flops / n, 5),
                        "ms_total": round(ms, 3), "ms_per_segment": round(ms / n, 4)}
    out_dir = config.paths.results_dir("complexity")
    (out_dir / "branches.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
