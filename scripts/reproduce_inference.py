from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import (
    compare_with_stored, evaluate_model, save_outputs, test_records, timed,
)
from deepfake_detection.reference import load_reference_repo


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score the exported checkpoints and compare with their results.yaml.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--models", nargs="+", default=None, help="default: every model in reproduction.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])

    summary = []
    for name in tqdm(args.models or list(repro["models"]), desc="models"):
        experiment = repro["models"][name]
        model_dir = config.paths.checkpoints_dir(name)
        stored = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))["test_results"]["primary_best_f1"]
        records = test_records(reference, config, experiment, repro["split"])
        result, seconds = timed(evaluate_model, reference, model_dir, records, repro["loader"])
        comparison = compare_with_stored(result, stored, repro["tolerance"])
        scope = experiment["scope"]
        save_outputs(config, name, scope, result, comparison, seconds)
        ok = all(row["ok"] for row in comparison.values())
        summary.append((name, len(records), ok, comparison))
        tqdm.write(f"{name}: {len(records)} videos, {'MATCH' if ok else 'MISMATCH'}, {seconds:.0f} s")

    tqdm.write("\nmodel | videos | metric: measured vs stored (delta)")
    for name, count, ok, comparison in summary:
        cells = ", ".join(f"{k} {v['measured']} vs {v['expected']} ({v['delta']})" for k, v in comparison.items())
        tqdm.write(f"{name} | {count} | {'MATCH' if ok else 'MISMATCH'} | {cells}")


if __name__ == "__main__":
    main()
