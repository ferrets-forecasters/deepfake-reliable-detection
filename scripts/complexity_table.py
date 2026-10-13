from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.flop_counter import FlopCounterMode

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.encoders.videomae_weights import load_videomae
from deepfake_detection.reference import load_reference_repo

SEGMENT_FRAMES, SEGMENT_SIZE, WHISPER_MELS, WHISPER_STEPS = 16, 224, 80, 3000


def count(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def gflops(fn) -> float:
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return counter.get_total_flops() / 1e9


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameters, checkpoint size and floating point operations of each component, counted on CPU. Matrix multiplications and convolutions are counted; element-wise operations are not.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--models", nargs="+", default=["moe_combined", "concat_combined"])
    parser.add_argument("--segments", type=int, default=60, help="sequence length for the classifier FLOPs")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    report: dict = {}

    videomae = load_videomae("MCG-NJU/videomae-base").eval()
    import whisper

    encoder = whisper.load_model("medium", device="cpu").encoder.eval()
    frames = torch.zeros(1, SEGMENT_FRAMES, 3, SEGMENT_SIZE, SEGMENT_SIZE)
    mel = torch.zeros(1, WHISPER_MELS, WHISPER_STEPS)
    report["VideoMAE-base encoder, frozen"] = {"parameters_M": round(count(videomae) / 1e6, 1), "gflops_per_2s_segment": round(gflops(lambda: videomae(pixel_values=frames)), 1)}
    report["Whisper-medium encoder, frozen"] = {"parameters_M": round(count(encoder) / 1e6, 1), "gflops_per_2s_segment": round(gflops(lambda: encoder(mel)), 1)}

    for name in args.models:
        model_dir = config.paths.checkpoints_dir(name)
        meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
        model = reference.build_model(meta["model"]).eval()
        x = torch.zeros(1, args.segments, 1792)
        mask = torch.ones(1, args.segments, dtype=torch.bool)
        lengths = torch.tensor([args.segments])
        flops = gflops(lambda: model(x, mask, lengths, lambda_=0.0, audio_avail=torch.tensor([True])))
        parts = {child: round(count(module) / 1e6, 3) for child, module in model.named_children() if count(module)}
        report[name] = {
            "classifier_parameters_M": round(count(model) / 1e6, 3),
            "parts_parameters_M": parts,
            "checkpoint_MB": round((model_dir / "best_f1.pt").stat().st_size / 1e6, 1),
            f"classifier_gflops_{args.segments}_segments": round(flops, 3),
        }

    out_dir = config.paths.results_dir("complexity")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "complexity.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False))


if __name__ == "__main__":
    main()
