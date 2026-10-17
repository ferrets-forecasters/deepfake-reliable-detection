"""The two branch scores the gate reads (audio anomaly, visual anomaly) and the gate weight alpha, per video and dataset, for the
combined model and the five leave-one-out models. Shows why alpha of a held-out dataset differs from alpha of the same dataset
in the combined model: the gate is a function of the two scores only, and the scores of an unseen dataset land elsewhere.
Inference only, no training. Resumable: a finished model's csv is skipped."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from deepfake_detection.config import load_config
from deepfake_detection.config.config import REPO_ROOT, _project_root
from deepfake_detection.evaluation.reproduce import test_records
from deepfake_detection.reference import load_reference_repo
sys.path.insert(0, str(Path(__file__).parent))
from export_predictions import records_of_run  # noqa: E402  (same record rules as the prediction export)

OUT = "gate_branch_scores"


def score_records(reference, model_dir: Path, records: list, loader_cfg: dict) -> dict:
    meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
    model = reference.build_model(meta["model"]).to(reference.device)
    state = torch.load(model_dir / "best_f1.pt", map_location=reference.device, weights_only=True)
    head = getattr(model, "domain_head", None)
    if "domain_head.3.weight" in state and isinstance(head, torch.nn.Sequential) and head[-1].out_features != state["domain_head.3.weight"].shape[0]:
        head[-1] = torch.nn.Linear(state["domain_head.3.weight"].shape[1], state["domain_head.3.weight"].shape[0])
    model.to(reference.device).load_state_dict(state)
    model.eval()
    stats = torch.load(model_dir / "norm_stats.pt", map_location="cpu", weights_only=False)["train_stats"]
    loader, _ = reference.make_loader(records, stats, loader_cfg["batch_size"], shuffle=False,
                                      performance={"num_workers": loader_cfg["num_workers"], "persistent_workers": False})
    columns = {"audio_anomaly": [], "visual_anomaly": [], "alpha_gate": []}
    with torch.no_grad():
        for batch in tqdm(loader, desc=model_dir.name, leave=False):
            x, _, lengths = batch[0].to(reference.device), batch[1], batch[2]
            mask = torch.arange(x.shape[1], device=reference.device).unsqueeze(0) < lengths.to(reference.device).unsqueeze(1)
            with torch.autocast(device_type=reference.device.type, dtype=torch.float16, enabled=loader_cfg["amp"]):
                diag = model.get_gate_diagnostics(x, mask, lengths)
            for key, name in (("audio_anomaly", "audio_anomaly"), ("visual_anomaly", "visual_anomaly"), ("alpha", "alpha_gate")):
                columns[name].append(diag[key].float().numpy())
    return {k: np.concatenate(v) for k, v in columns.items()}


def summarize(out_dir: Path) -> None:
    """Mean branch scores and alpha per model, scope, dataset and label, and for each held-out dataset the distance, in standard
    deviations of the scores of the seen datasets, from its centre to the centre of every seen dataset (fakes and reals together)."""
    import pandas as pd
    rows, near = [], []
    for path in sorted(out_dir.glob("moe_*.csv")):
        frame = pd.read_csv(path)
        model = path.stem
        grouped = frame.groupby(["scope", "dataset", "label"])[["audio_anomaly", "visual_anomaly", "alpha_gate"]].agg(["mean", "std"])
        for (scope, dataset, label), row in grouped.iterrows():
            rows.append({"model": model, "scope": scope, "dataset": dataset, "label": label, "videos": int(((frame.scope == scope) & (frame.dataset == dataset) & (frame.label == label)).sum()),
                         **{f"{c}_{s}": round(float(row[(c, s)]), 4) for c in ("audio_anomaly", "visual_anomaly", "alpha_gate") for s in ("mean", "std")}})
        seen = frame[frame.scope == "seen"]
        if seen.empty:
            continue
        spread = seen[["audio_anomaly", "visual_anomaly"]].std()
        for held in sorted(frame[frame.scope == "held_out"].dataset.unique()):
            centre = frame[(frame.scope == "held_out") & (frame.dataset == held)][["audio_anomaly", "visual_anomaly"]].mean()
            for dataset, part in seen.groupby("dataset"):
                distance = float((((part[["audio_anomaly", "visual_anomaly"]].mean() - centre) / spread) ** 2).sum() ** 0.5)
                near.append({"model": model, "held_out": held, "seen_dataset": dataset, "distance_sd": round(distance, 3),
                             "seen_alpha_mean": round(float(part.alpha_gate.mean()), 3)})
    pd.DataFrame(rows).to_csv(out_dir / "scores_per_dataset.csv", index=False)
    near = pd.DataFrame(near)
    near.to_csv(out_dir / "held_out_distance_to_seen.csv", index=False)
    # a check of the reading "alpha of an unseen dataset follows the seen datasets whose scores are nearest": alpha predicted as the mean
    # of the seen datasets' alpha weighted by 1 / distance, against the alpha the gate gave
    predicted = []
    for (model, held), part in near.groupby(["model", "held_out"]):
        weights = 1.0 / part.distance_sd
        observed = pd.read_csv(out_dir / f"{model}.csv")
        observed = observed[(observed.scope == "held_out") & (observed.dataset == held)].alpha_gate.mean()
        predicted.append({"model": model, "held_out": held, "predicted_alpha": round(float((weights * part.seen_alpha_mean).sum() / weights.sum()), 3),
                          "observed_alpha": round(float(observed), 3)})
    pd.DataFrame(predicted).to_csv(out_dir / "held_out_alpha_prediction.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--reproduction", default=str(REPO_ROOT / "configs" / "reproduction.yaml"))
    parser.add_argument("--models", nargs="+", default=["moe_combined_retrain", "moe_leave_polyglotfake_retrain", "moe_leave_hidf_retrain",
                                                          "moe_leave_fakeavceleb_retrain", "moe_leave_celebdfv2_retrain", "moe_leave_ffpp_retrain"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    repro = yaml.safe_load(Path(args.reproduction).read_text(encoding="utf-8"))
    reference = load_reference_repo(_project_root() / repro["reference_repo"])
    out_dir = config.paths.analysis_dir(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets_root = config.paths.datasets_root
    seen = test_records(reference, config, {"experiment": "combined"}, {"val_ratio": 0.2, "seed": 42})

    for name in args.models:
        target = out_dir / f"{name}.csv"
        if target.exists() and not args.force:
            print(f"{name}: already done")
            continue
        model_dir = config.paths.models_root / name
        meta = yaml.safe_load((model_dir / "results.yaml").read_text(encoding="utf-8"))
        records, _ = records_of_run(reference, config, repro, name, meta)
        held_out = {Path(r[0][0]).relative_to(datasets_root).parts[0] for r in records}
        scope = "held_out" if "leave" in name else "combined_test"
        rows = [(r, scope) for r in records]
        if scope == "held_out":
            rows += [(r, "seen") for r in seen if Path(r[0][0]).relative_to(datasets_root).parts[0] not in held_out]
        scores = score_records(reference, model_dir, [r for r, _ in rows], repro["loader"])
        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video", "dataset", "label", "manip_type", "audio_avail", "scope", "audio_anomaly", "visual_anomaly", "alpha_gate"])
            for i, ((files, label, _subject, manip, audio), part) in enumerate(rows):
                relative = Path(files[0]).relative_to(datasets_root)
                writer.writerow([relative.as_posix(), relative.parts[0], int(label), manip, int(bool(audio)), part,
                                 f"{scores['audio_anomaly'][i]:.6f}", f"{scores['visual_anomaly'][i]:.6f}", f"{scores['alpha_gate'][i]:.6f}"])
        print(f"{name}: {len(rows)} videos")
    summarize(out_dir)


if __name__ == "__main__":
    main()
