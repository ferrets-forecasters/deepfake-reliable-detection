from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.reference import load_reference_repo
from deepfake_detection.training.cv_splits import load_kfold_splits
from deepfake_detection.training.reference_training import load_reference_training, train_resumable


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one model with the old repo's training code on the saved embeddings, resumable after every epoch. Output goes to models/<name>/.")
    parser.add_argument("config", help="a YAML under the old repo's src/experiments/configs, or a path")
    parser.add_argument("--name", required=True, help="output folder name under models/")
    parser.add_argument("--override", nargs="*", default=[], metavar="key.sub=value")
    parser.add_argument("--project-config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    args = parser.parse_args()

    config = load_config(args.project_config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    root = _project_root() / repro["reference_repo"]
    run_module, _ = load_reference_training(root, config, _project_root() / "temp" / "training_cache")

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = root / "src" / "experiments" / "configs" / config_path.name
    cfg = run_module.apply_overrides(run_module.load_yaml(config_path), args.override)
    save_dir = config.paths.checkpoints_dir(args.name)
    if (save_dir / "results.yaml").exists():
        print(f"{args.name}: finished, nothing to do")
        return

    seed = int(cfg["data"].get("seed", 42))
    model = run_module.build_model(cfg["model"])
    if cfg["data"].get("experiment") == "all_datasets_kfold":
        data = cfg["data"]
        train, val, test, domain_labels, norm_overrides = load_kfold_splits(load_reference_repo(root), config, int(data["fold"]), int(data.get("folds", 5)), seed)
        print(f"fold {data['fold']}: train {len(train)} val {len(val)} test {len(test)} videos", flush=True)
    else:
        train, val, test, domain_labels, norm_overrides = run_module.load_splits(cfg["data"], cfg.get("performance", {}))
    result = train_resumable(args.name, model, train, val, test, save_dir, run_module.flatten_training_cfg(cfg), domain_labels, norm_overrides, seed)

    summary = {"name": args.name, "timestamp": datetime.now().isoformat(timespec="seconds"), "best_f1": round(float(result["best_f1"]), 4),
               "best_loss": round(float(result["best_loss"]), 4), "test_results": result["test_results"], "model": cfg["model"], "data": cfg["data"],
               "training": cfg.get("training", {}), "loss": cfg.get("loss", {}), "swa": cfg.get("swa", {}), "performance": cfg.get("performance", {})}
    (save_dir / "results.yaml").write_text(yaml.dump(summary, default_flow_style=False, sort_keys=False), encoding="utf-8")
    print(f"{args.name}: best_f1 {summary['best_f1']}, test {result['test_results']}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # the interpreter teardown after training crashed with a fail-fast status on this machine


if __name__ == "__main__":
    main()
