from __future__ import annotations

import argparse
import csv
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deepfake_detection.config import load_config

DATASETS = {"polyglotfake": "PolyglotFake", "hidf": "HiDF", "fakeavceleb": "FakeAVCeleb", "celebdfv2": "CelebDFv2", "ffpp": "FF++"}
COLOURS = {"polyglotfake": "#c8463c", "hidf": "#3c78c8", "fakeavceleb": "#d29b2d", "celebdfv2": "#4a9a5b", "ffpp": "#7a5aa8"}


def load_dataset(config, dataset: str, per_class: int, rng: random.Random, workers: int) -> dict:
    """A seeded sample of segments per class: video and audio vectors, label, subject, audio availability."""
    manifest = {}
    with open(config.paths.datasets_root / dataset / "labels" / "manifest.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            manifest[row["sample_id"]] = row
    files, labels = [], []
    for folder, label in (("real", 0), ("fake", 1)):
        found = sorted((config.embedding_dir(dataset) / folder).glob("*.npz"))
        pick = rng.sample(found, min(per_class, len(found)))
        files += pick
        labels += [label] * len(pick)

    def read(path):
        try:
            with np.load(path, allow_pickle=True) as data:  # embeddings written by this project's own extraction
                return data["video_feat"].astype(np.float32), data["audio_feat"].astype(np.float32)
        except Exception:
            return None  # the few files listed in failed_files.tsv

    with ThreadPoolExecutor(workers) as pool:
        results = list(tqdm(pool.map(read, files), total=len(files), desc=dataset, unit="file", mininterval=10, leave=False))
    keep = [i for i, r in enumerate(results) if r is not None]
    loaded = [results[i] for i in keep]
    files, labels = [files[i] for i in keep], [labels[i] for i in keep]
    subjects = []
    for path in files:
        row = manifest.get(re.sub(r"_seg\d+$", "", path.stem))
        subjects.append(row["subject_id"] if row else path.stem)
    audio = int(next(iter(manifest.values()))["audio_available"]) if manifest else 0
    return {"video": np.stack([v for v, _ in loaded]), "audio": np.stack([a for _, a in loaded]), "label": np.array(labels), "subject": np.array(subjects), "has_audio": bool(audio)}


def cohens_d(real: np.ndarray, fake: np.ndarray) -> float:
    pooled = np.sqrt((real.var(ddof=1) * (len(real) - 1) + fake.var(ddof=1) * (len(fake) - 1)) / (len(real) + len(fake) - 2))
    return float((fake.mean() - real.mean()) / pooled) if pooled > 0 else 0.0


def probe_auc(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> float | None:
    """Logistic-regression probe, trained and tested on disjoint subjects."""
    train, test = next(GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(x, y, groups))
    if len(set(y[train])) < 2 or len(set(y[test])) < 2:
        return None
    scaler = StandardScaler().fit(x[train])
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(x[train]), y[train])
    return float(roc_auc_score(y[test], model.predict_proba(scaler.transform(x[test]))[:, 1]))


def cross_auc(source: dict, target: dict, key: str) -> float | None:
    if len(set(source["label"])) < 2 or len(set(target["label"])) < 2:
        return None
    scaler = StandardScaler().fit(source[key])
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(source[key]), source["label"])
    return float(roc_auc_score(target["label"], model.predict_proba(scaler.transform(target[key]))[:, 1]))


def mmd2(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    """Biased RBF-kernel estimate of the squared maximum mean discrepancy."""
    def kernel(a, b):
        d = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.exp(-np.maximum(d, 0) / (2 * bandwidth ** 2))

    return float(kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean())


def matrix_rows(names: list[str], values: np.ndarray) -> list[list]:
    return [[DATASETS[n]] + [None if np.isnan(v) else round(float(v), 4) for v in row] for n, row in zip(names, values)]


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding separation and domain gap from the saved embeddings, no training: Cohen's d, linear probes (within and across datasets), centroid distances, MMD, PCA and t-SNE scatter. Writes results/analyses/embedding_separation/ and results/analyses/domain_gap/.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--per-class", type=int, default=1000, help="segments sampled per class and dataset")
    parser.add_argument("--tsne-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = random.Random(args.seed)
    data = {name: load_dataset(config, name, args.per_class, rng, args.workers) for name in DATASETS}
    names = list(DATASETS)
    separation = config.paths.analysis_dir("embedding_separation")
    gap = config.paths.analysis_dir("domain_gap")
    separation.mkdir(parents=True, exist_ok=True)
    gap.mkdir(parents=True, exist_ok=True)

    report: dict = {"sample": {"segments_per_class": args.per_class, "seed": args.seed}, "per_dataset": {}}
    for name, d in data.items():
        real, fake = d["label"] == 0, d["label"] == 1
        entry = {"real_segments": int(real.sum()), "fake_segments": int(fake.sum()), "has_audio": d["has_audio"],
                 "cohens_d_video_norm": round(cohens_d(np.linalg.norm(d["video"][real], axis=1), np.linalg.norm(d["video"][fake], axis=1)), 4),
                 "probe_auc_video": probe_auc(d["video"], d["label"], d["subject"], args.seed)}
        if d["has_audio"]:
            entry["cohens_d_audio_norm"] = round(cohens_d(np.linalg.norm(d["audio"][real], axis=1), np.linalg.norm(d["audio"][fake], axis=1)), 4)
            entry["probe_auc_audio"] = probe_auc(d["audio"], d["label"], d["subject"], args.seed)
            entry["probe_auc_both"] = probe_auc(np.hstack([d["video"], d["audio"]]), d["label"], d["subject"], args.seed)
        report["per_dataset"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items()}

    for key in ("video", "audio"):
        usable = [n for n in names if key == "video" or data[n]["has_audio"]]
        matrix = np.full((len(usable), len(usable)), np.nan)
        for i, a in enumerate(tqdm(usable, desc=f"cross-dataset probe {key}", leave=False)):
            for j, b in enumerate(usable):
                if i != j:
                    value = cross_auc(data[a], data[b], key)
                    matrix[i, j] = np.nan if value is None else value
        write_csv(separation / f"cross_dataset_probe_auc_{key}.csv", ["trained on \\ tested on"] + [DATASETS[n] for n in usable], matrix_rows(usable, matrix))

    pooled = {key: np.vstack([data[n][key] for n in names]) for key in ("video", "audio")}
    scaler = {key: StandardScaler().fit(pooled[key]) for key in pooled}
    centroids = {key: np.stack([scaler[key].transform(data[n][key]).mean(0) for n in names]) for key in pooled}
    for key in ("video", "audio"):
        usable = [i for i, n in enumerate(names) if key == "video" or data[n]["has_audio"]]
        distance = np.array([[np.linalg.norm(centroids[key][i] - centroids[key][j]) for j in usable] for i in usable])
        write_csv(gap / f"centroid_distance_{key}.csv", ["dataset"] + [DATASETS[names[i]] for i in usable], matrix_rows([names[i] for i in usable], distance))
        sample = {names[i]: scaler[key].transform(data[names[i]][key])[: 600] for i in usable}
        stacked = np.vstack(list(sample.values()))
        pairs = rng.sample(range(len(stacked)), min(400, len(stacked)))
        bandwidth = float(np.median(np.linalg.norm(stacked[pairs][:, None] - stacked[pairs][None], axis=2)) or 1.0)
        values = np.array([[mmd2(sample[names[i]], sample[names[j]], bandwidth) for j in usable] for i in usable])
        write_csv(gap / f"mmd_{key}.csv", ["dataset"] + [DATASETS[names[i]] for i in usable], matrix_rows([names[i] for i in usable], values))
        fig, ax = plt.subplots(figsize=(4.6, 4))
        image = ax.imshow(values, cmap="Greys")
        ax.set_xticks(range(len(usable)), [DATASETS[names[i]] for i in usable], rotation=45, ha="right")
        ax.set_yticks(range(len(usable)), [DATASETS[names[i]] for i in usable])
        fig.colorbar(image, label="MMD²")
        fig.tight_layout()
        fig.savefig(gap / f"mmd_{key}.png", dpi=200)
        plt.close(fig)

    small = {n: {k: (v[np.r_[np.where(d['label'] == 0)[0][: args.tsne_per_class], np.where(d['label'] == 1)[0][: args.tsne_per_class]]] if isinstance(v, np.ndarray) else v) for k, v in d.items()} for n, d in data.items()}
    points = scaler["video"].transform(np.vstack([small[n]["video"] for n in names]))
    owner = np.concatenate([[n] * len(small[n]["video"]) for n in names])
    klass = np.concatenate([small[n]["label"] for n in names])
    embeddings = {"pca": PCA(2, random_state=args.seed).fit_transform(points), "tsne": TSNE(2, random_state=args.seed, init="pca", perplexity=30).fit_transform(points)}
    for method, xy in embeddings.items():
        fig, ax = plt.subplots(figsize=(5.4, 4.4))
        for n in names:
            for label, marker in ((0, "o"), (1, "x")):
                mask = (owner == n) & (klass == label)
                ax.scatter(xy[mask, 0], xy[mask, 1], s=8, marker=marker, color=COLOURS[n], alpha=0.6, linewidths=0.8, label=f"{DATASETS[n]} {'real' if label == 0 else 'fake'}")
        ax.legend(frameon=False, fontsize=6, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(separation / f"{method}_video_features.png", dpi=200)
        plt.close(fig)

    (separation / "separation.yaml").write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump(report["per_dataset"], sort_keys=False))


if __name__ == "__main__":
    main()
