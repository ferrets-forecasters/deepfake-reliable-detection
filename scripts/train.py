from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from deepfake_detection.config import load_config
from deepfake_detection.data import CachedFeatureDataset, read_manifest
from deepfake_detection.modeling import ConcatFusionClassifier, MoEFusionClassifier
from deepfake_detection.training import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MoEFusionClassifier or ConcatFusionClassifier on cached features.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default="combined.csv")
    parser.add_argument("--model", choices=["moe", "concat"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--domain-adversarial", action="store_true")
    parser.add_argument(
        "--held-out",
        default=None,
        help="Dataset name to exclude from training entirely, for a Leave-One-Dataset-Out run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = read_manifest(config.paths.manifests_dir / args.manifest)
    if args.held_out:
        manifest = manifest[manifest["dataset"] != args.held_out]

    domain_names = sorted(manifest["dataset"].unique())
    domain_map = {name: i for i, name in enumerate(domain_names)} if args.domain_adversarial else None
    use_domain_adversarial = args.domain_adversarial and len(domain_names) > 1

    train_set = CachedFeatureDataset(manifest[manifest["split"] == "train"], config.paths.features_dir, domain_map)
    val_set = CachedFeatureDataset(manifest[manifest["split"] == "val"], config.paths.features_dir, domain_map)
    train_loader = DataLoader(train_set, batch_size=config.train.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False, num_workers=4)

    num_domains = len(domain_names) if use_domain_adversarial else None
    if args.model == "moe":
        model = MoEFusionClassifier(config, num_domains=num_domains)
    else:
        model = ConcatFusionClassifier(config, num_domains=num_domains)

    trainer = Trainer(config, model, args.run_name, device, use_domain_adversarial=use_domain_adversarial)
    best_checkpoint = trainer.fit(train_loader, val_loader)
    print(f"best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
