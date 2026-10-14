from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, roc_auc_score

from deepfake_detection.config import load_config

SEED = 42
FAMILY = {"polyglotfake": 1, "fakeavceleb": 1, "hidf": 0, "celebdfv2": 0, "ffpp": 0}
RUNS = {"gated": "moe_combined_retrain", "static": "concat_combined_retrain", "video_only": "video_only_pure", "audio_only": "audio_only_pure", "no_gate_loss": "moe_no_gate_loss"}


def load_run(config, run: str) -> dict[tuple[str, str], tuple[int, float, float]]:
    path = config.paths.predictions_dir(f"{run}_on_combined_test_seed{SEED}") / "predictions.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            alpha = float(r["alpha_best_f1"]) if r["alpha_best_f1"] not in ("", "nan") else float("nan")
            out[(r["dataset"], re.sub(r"_seg\d+$", "", Path(r["video"]).stem))] = (int(float(r["label"])), float(r["p_fake_best_f1"]), alpha)
    return out


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(np.asarray(y).tolist())) == 2 else float("nan")


def macro(y, p) -> float:
    return float(f1_score(y, np.asarray(p) >= 0.5, average="macro"))


def bootstrap_gap(y, a, b, fn, resamples=1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    base = fn(y, a) - fn(y, b)
    draws = []
    for _ in range(resamples):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) == 2:
            draws.append(fn(y[i], a[i]) - fn(y[i], b[i]))
    return [round(base, 4), round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Shortcut checks on the saved test predictions of the combined split: clip length (R1), what the gate weight identifies (R2) and where the audio helps (R3). Reads predictions, manifests and audio embedding norms only.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    config = load_config(args.config)

    preds = {name: load_run(config, run) for name, run in RUNS.items()}
    keys = sorted(set.intersection(*[set(p) for p in preds.values()]))
    segments: dict[tuple[str, str], int] = {}
    for dataset in FAMILY:
        with open(config.paths.datasets_root / dataset / "labels" / "manifest.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                segments[(dataset, r["sample_id"])] = int(r["num_segments"])
    keys = [k for k in keys if k in segments]
    datasets = list(FAMILY)
    report: dict = {"seed": SEED, "videos": len(keys)}

    # R1
    r1 = {}
    for ds in datasets:
        k = [x for x in keys if x[0] == ds]
        y = np.array([preds["gated"][x][0] for x in k]); n = np.array([segments[x] for x in k], dtype=float)
        entry = {"videos": len(k), "length_only_auc": round(auc(y, n), 4), "gated_auc": round(auc(y, [preds["gated"][x][1] for x in k]), 4),
                 "static_auc": round(auc(y, [preds["static"][x][1] for x in k]), 4)}
        for cls, name in ((0, "real"), (1, "fake")):
            sel = y == cls
            if sel.sum() > 5 and n[sel].std() > 0:
                entry[f"spearman_p_fake_vs_length_{name}"] = round(float(spearmanr(n[sel], [preds["gated"][x][1] for x, s in zip(k, sel) if s])[0]), 3)
        rng = np.random.default_rng(SEED)
        picked = []
        for value in sorted(set(n.tolist())):
            real = [x for x, yy, nn in zip(k, y, n) if nn == value and yy == 0]
            fake = [x for x, yy, nn in zip(k, y, n) if nn == value and yy == 1]
            m = min(len(real), len(fake))
            if m:
                picked += [real[i] for i in rng.choice(len(real), m, replace=False)] + [fake[i] for i in rng.choice(len(fake), m, replace=False)]
        if picked:
            yy = np.array([preds["gated"][x][0] for x in picked])
            entry["length_matched"] = {"videos": len(picked), "real": int((yy == 0).sum()), "fake": int((yy == 1).sum()),
                                       "length_only_auc": round(auc(yy, [segments[x] for x in picked]), 4),
                                       "gated": {"macro_f1": round(macro(yy, [preds["gated"][x][1] for x in picked]), 4), "auc": round(auc(yy, [preds["gated"][x][1] for x in picked]), 4)},
                                       "static": {"macro_f1": round(macro(yy, [preds["static"][x][1] for x in picked]), 4), "auc": round(auc(yy, [preds["static"][x][1] for x in picked]), 4)}}
        r1[ds] = entry
    report["R1_length"] = r1

    # R2
    r2 = {}
    for name in ("gated", "no_gate_loss"):
        fam = np.array([FAMILY[x[0]] for x in keys]); lab = np.array([preds[name][x][0] for x in keys]); alpha = np.array([preds[name][x][2] for x in keys])
        entry = {"alpha_mean": round(float(np.nanmean(alpha)), 4)}
        for cls, cname in ((None, "all"), (0, "real only"), (1, "fake only")):
            sel = np.ones(len(keys), bool) if cls is None else lab == cls
            entry[f"family_auc_from_alpha ({cname})"] = round(auc(fam[sel], -alpha[sel]), 4)
        per = {}
        for ds in datasets:
            sel = np.array([x[0] == ds for x in keys])
            a, y, p = alpha[sel], lab[sel], np.array([preds[name][x][1] for x in keys])[sel]
            per[ds] = {"alpha_mean_real": round(float(np.nanmean(a[y == 0])), 3), "alpha_mean_fake": round(float(np.nanmean(a[y == 1])), 3),
                       "auc_label_from_alpha": round(auc(y, a), 4), "spearman_alpha_p_fake": round(float(spearmanr(a, p)[0]), 3) if np.nanstd(a) > 0 else None}
        entry["per_dataset"] = per
        r2[name] = entry
    clips = config.paths.predictions_dir("fakeavceleb_categories_retrain") / "clips.csv"
    if clips.exists():
        with open(clips, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0])
        cat = next((c for c in cols if "categ" in c.lower() or c == "kind"), None)
        alpha_col = next((c for c in cols if "alpha" in c.lower()), None)
        p_col = next((c for c in cols if c.lower().startswith("p_fake")), None)
        if cat and alpha_col and p_col:
            groups: dict[str, list] = {}
            for r in rows:
                if r[alpha_col] not in ("", "nan"):
                    groups.setdefault(r[cat], []).append((float(r[alpha_col]), float(r[p_col])))
            r2["fakeavceleb_categories_gated"] = {c: {"clips": len(v), "alpha_mean": round(float(np.mean([a for a, _ in v])), 3), "p_fake_mean": round(float(np.mean([p for _, p in v])), 3)} for c, v in groups.items()}
    report["R2_alpha"] = r2

    # R3
    r3 = {"per_dataset": {}}
    for ds in datasets:
        k = [x for x in keys if x[0] == ds]
        y = np.array([preds["gated"][x][0] for x in k])
        g, v, a = (np.array([preds[n][x][1] for x in k]) for n in ("gated", "video_only", "audio_only"))
        r3["per_dataset"][ds] = {"videos": len(k), "gated_minus_video_only_macro_f1": bootstrap_gap(y, g, v, macro), "gated_minus_video_only_auc": bootstrap_gap(y, g, v, auc),
                                 "audio_only_auc": round(auc(y, a), 4), "video_only_auc": round(auc(y, v), 4)}
    k = [x for x in keys if x[0] != "polyglotfake"]
    y = np.array([preds["gated"][x][0] for x in k])
    g, v = (np.array([preds[n][x][1] for x in k]) for n in ("gated", "video_only"))
    r3["without_polyglotfake"] = {"videos": len(k), "gated_minus_video_only_macro_f1": bootstrap_gap(y, g, v, macro), "gated_minus_video_only_auc": bootstrap_gap(y, g, v, auc)}
    k = [x for x in keys if FAMILY[x[0]] == 1 and x[0] != "polyglotfake"]
    y = np.array([preds["gated"][x][0] for x in k])
    g, v = (np.array([preds[n][x][1] for x in k]) for n in ("gated", "video_only"))
    r3["fakeavceleb_only"] = {"videos": len(k), "gated_minus_video_only_macro_f1": bootstrap_gap(y, g, v, macro), "gated_minus_video_only_auc": bootstrap_gap(y, g, v, auc)}
    # trivial audio cues: first-segment audio embedding norm and clip length
    norms = {}
    for ds in ("polyglotfake", "fakeavceleb", "hidf"):
        k = [x for x in keys if x[0] == ds]
        values, labels = [], []
        for x in k:
            label = "real" if preds["gated"][x][0] == 0 else "fake"
            path = config.embedding_dir(ds) / label / f"{x[1]}_seg000.npz"
            try:
                with np.load(path, allow_pickle=True) as d:  # the project's own embedding files
                    values.append(float(np.linalg.norm(d["audio_feat"])))
                    labels.append(preds["gated"][x][0])
            except Exception:
                continue
        norms[ds] = {"videos": len(values), "audio_embedding_norm_auc": round(auc(labels, values), 4)}
    r3["trivial_audio_cue"] = norms
    report["R3_audio"] = r3

    out = config.paths.analysis_dir("split_leakage_audit") / "shortcuts_seed42.yaml"
    out.write_text(yaml.safe_dump(report, sort_keys=False, width=140), encoding="utf-8")
    print(yaml.safe_dump(report, sort_keys=False, width=140))


if __name__ == "__main__":
    main()
