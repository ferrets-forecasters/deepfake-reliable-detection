from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.data import CachedFeatureDataset, read_manifest
from deepfake_detection.evaluation import compute_metrics
from deepfake_detection.modeling import ConcatFusionClassifier, MoEFusionClassifier
from deepfake_detection.utils.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on a manifest split.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="combined.csv")
    parser.add_argument("--model", choices=["moe", "concat"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--eval-dataset", default=None, help="Restrict evaluation to one dataset (e.g. an LOO held-out set).")
    parser.add_argument(
        "--zero-modality", default=None, choices=["visual", "audio"], help="Modality-ablation: zero this stream."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = read_manifest(config.paths.manifests_dir / args.manifest)
    manifest = manifest[manifest["split"] == args.split]
    if args.eval_dataset:
        manifest = manifest[manifest["dataset"] == args.eval_dataset]

    dataset = CachedFeatureDataset(manifest, config.paths.features_dir)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=False, num_workers=4)

    model_cls = MoEFusionClassifier if args.model == "moe" else ConcatFusionClassifier
    model = model_cls(config).to(device)
    state = load_checkpoint(args.checkpoint)
    model.load_state_dict(state["model_state"])
    model.eval()

    all_probs, all_targets, all_alpha = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            visual_seq = batch["visual_seq"].to(device)
            audio_seq = batch["audio_seq"].to(device)
            concat_repr = batch["concat_repr"].to(device)
            if args.zero_modality == "visual":
                visual_seq = torch.zeros_like(visual_seq)
            elif args.zero_modality == "audio":
                audio_seq = torch.zeros_like(audio_seq)

            if args.model == "moe":
                out = model(visual_seq, audio_seq)
                all_alpha.append(out.alpha.cpu().numpy())
            else:
                out = model(concat_repr)

            all_probs.append(torch.sigmoid(out.final_logit).cpu().numpy())
            all_targets.append(batch["label"].numpy())

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    metrics = compute_metrics(probs, targets)

    results_dir = config.paths.results_dir(args.run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics.as_dict() | {"n_segments": len(targets)}, f, indent=2)

    predictions_dir = config.paths.predictions_dir(args.run_name)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions = {"segment_id": manifest["segment_id"].to_numpy(), "prob": probs, "label": targets}
    if all_alpha:
        predictions["alpha"] = np.concatenate(all_alpha)
    np.savez_compressed(predictions_dir / f"{args.split}_predictions.npz", **predictions)

    print(json.dumps(metrics.as_dict(), indent=2))


if __name__ == "__main__":
    main()
