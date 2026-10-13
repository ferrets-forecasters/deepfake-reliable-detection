from __future__ import annotations

import argparse

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

from deepfake_detection.config import load_config

HELD_OUT = {"moe_leave_polyglotfake": ("polyglotfake", "PolyglotFake"), "moe_leave_hidf": ("hidf", "HiDF"), "moe_leave_fakeavceleb": ("fakeavceleb", "FakeAVCeleb"),
            "moe_leave_celebdfv2": ("celebdfv2", "CelebDFv2"), "moe_leave_ffpp": ("ffpp", "FF++")}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ece(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probs, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(probs[mask].mean() - labels[mask].mean())
    return float(total)


def best_threshold(margin: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(np.quantile(margin, np.linspace(0.01, 0.99, 99)))
    scores = [f1_score(labels, margin >= c, average="macro") for c in candidates]
    return float(candidates[int(np.argmax(scores))])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recalibration of the leave-one-out models on a small labelled sample of the held-out dataset: a re-fitted threshold and a logistic (Platt) rescaling of the margin, scored on the remaining videos. Uses the saved predictions only.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20, 50, 100, 500])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)
    report: dict = {}
    print("dataset | labelled videos | macro-F1 at 0.5 | re-fitted threshold | Platt | ECE before | ECE after Platt")
    for model, (scope, label) in tqdm(HELD_OUT.items(), desc="held-out datasets"):
        with np.load(config.paths.predictions_dir(f"{model}_on_{scope}") / "predictions.npz", allow_pickle=False) as data:
            probs, labels = data["probs"].astype(float), data["labels"].astype(int)
        margin = logit(probs)
        base_f1 = f1_score(labels, probs >= 0.5, average="macro")
        base_ece = ece(probs, labels)
        report[label] = {"videos": int(len(labels)), "macro_f1_at_0.5": round(float(base_f1), 4), "ece": round(base_ece, 4), "by_labelled_videos": {}}
        for size in args.sizes:
            thresholds, platts, eces = [], [], []
            for _ in range(args.repeats):
                chosen = np.zeros(len(labels), dtype=bool)
                for cls in (0, 1):
                    pool = np.flatnonzero(labels == cls)
                    take = max(2, int(round(size * len(pool) / len(labels))))
                    chosen[rng.choice(pool, size=min(take, len(pool)), replace=False)] = True
                rest = ~chosen
                thresholds.append(f1_score(labels[rest], margin[rest] >= best_threshold(margin[chosen], labels[chosen]), average="macro"))
                model_platt = LogisticRegression(C=1e4).fit(margin[chosen, None], labels[chosen])
                calibrated = model_platt.predict_proba(margin[rest, None])[:, 1]
                platts.append(f1_score(labels[rest], calibrated >= 0.5, average="macro"))
                eces.append(ece(calibrated, labels[rest]))
            report[label]["by_labelled_videos"][size] = {
                "macro_f1_threshold": [round(float(np.mean(thresholds)), 4), round(float(np.std(thresholds)), 4)],
                "macro_f1_platt": [round(float(np.mean(platts)), 4), round(float(np.std(platts)), 4)],
                "ece_platt": round(float(np.mean(eces)), 4),
            }
            print(f"{label} | {size} | {base_f1:.4f} | {np.mean(thresholds):.4f} ± {np.std(thresholds):.4f} | {np.mean(platts):.4f} ± {np.std(platts):.4f} | {base_ece:.4f} | {np.mean(eces):.4f}")

    out_dir = config.paths.results_dir("calibration")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
