from __future__ import annotations

import hashlib
import importlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, SWALR

from deepfake_detection.config import Config
from deepfake_detection.utils.progress import progress

DATASET_ROOTS = {"POLYGLOT_ROOT": "polyglotfake", "HIDF_ROOT": "hidf", "CELEBDFV2_ROOT": "celebdfv2", "FAKEAVCELEB_ROOT": "fakeavceleb", "FFPP_ROOT": "ffpp"}
feature_cache_dir: Path | None = None


def load_reference_training(config: Config, cache_dir: Path):
    """Imports the reference training modules with the dataset roots pointed at this project's
    embedding folders and the split and norm-stat caches inside this project. Returns the run
    module (config loading, model building, split scanning) and the trainer module."""
    global feature_cache_dir
    feature_cache_dir = cache_dir / "features"
    import deepfake_detection.reference.config as reference_config

    for attribute, dataset in DATASET_ROOTS.items():
        setattr(reference_config, attribute, config.embedding_dir(dataset))
    reference_config.CACHE_DIR = cache_dir
    run_module = importlib.import_module("deepfake_detection.reference.run")
    import deepfake_detection.reference.training.trainer as trainer_module

    trainer_module.compute_norm_stats = threaded_norm_stats
    return run_module, trainer_module


def threaded_norm_stats(records, workers: int = 16) -> dict:
    """The reference compute_norm_stats with the file reads spread over threads and the sums kept in
    float64: per-feature mean and std over every segment of the training records."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    def load(path):
        try:
            with np.load(path, allow_pickle=True) as data:  # embeddings written by this project's own extraction
                return data["video_feat"].astype(np.float64), data["audio_feat"].astype(np.float64)
        except Exception:
            return None

    paths = [f for row in records for f in row[0]]
    totals = {key: [0.0, 0.0] for key in ("video", "audio")}
    count = 0
    with ThreadPoolExecutor(workers) as pool:
        for item in tqdm(pool.map(load, paths), total=len(paths), desc="Norm stats", unit="segment", leave=False, mininterval=20):
            if item is None:
                continue
            count += 1
            for key, value in zip(("video", "audio"), item):
                totals[key][0] = totals[key][0] + value
                totals[key][1] = totals[key][1] + value * value
    stats = {}
    for key in ("video", "audio"):
        mean = totals[key][0] / count
        stats[f"{key}_mean"] = mean.astype(np.float32)
        stats[f"{key}_std"] = np.sqrt(np.maximum(totals[key][1] / count - mean * mean, 0.0)).astype(np.float32).clip(1e-6)
    return stats


class _CachedNpz:
    def __init__(self, arrays: dict):
        self._arrays = arrays

    def __getitem__(self, key):
        return self._arrays[key]

    def close(self) -> None:
        pass


class _NumpyWithFeatureCache:
    """Stand-in for the `np` name inside the old dataset module: everything is numpy's own, except
    that np.load of an embedding file already in memory returns the stored arrays."""

    def __init__(self, cache: dict):
        self._cache = cache

    def __getattr__(self, name):
        return getattr(np, name)

    def load(self, path, *args, **kwargs):
        hit = self._cache.get(str(path))
        return _CachedNpz(hit) if hit is not None else np.load(path, *args, **kwargs)


def _read_embeddings(paths: list, workers: int):
    """Yields (path, video, audio) per file in order, float32, with None arrays for a file that cannot be read."""
    from concurrent.futures import ThreadPoolExecutor

    from tqdm import tqdm

    def load(path):
        try:
            with np.load(path, allow_pickle=True) as data:  # embeddings written by this project's own extraction
                return path, data["video_feat"].astype(np.float32), data["audio_feat"].astype(np.float32)
        except Exception:
            return path, None, None

    with ThreadPoolExecutor(workers) as pool:
        yield from tqdm(pool.map(load, paths), total=len(paths), desc="Feature cache", unit="file", mininterval=20)


class _MappedFeatures:
    """The embeddings of one set of files as two memory-mapped matrices. Every training process on the
    machine maps the same files, so the operating system holds one copy of the pages."""

    def __init__(self, index: dict, video: np.ndarray, audio: np.ndarray):
        self._index, self._video, self._audio = index, video, audio

    def get(self, path):
        row = self._index.get(path)
        return None if row is None else {"video_feat": self._video[row], "audio_feat": self._audio[row]}

    def __len__(self) -> int:
        return len(self._index)


def _build_shared_cache(paths: list, folder: Path, key: str, workers: int) -> None:
    """Writes the embeddings once into two .npy matrices and a path index. The index is written last, so its
    existence means the matrices are complete."""
    kept, video, audio = [], None, None
    for path, v, a in _read_embeddings(paths, workers):
        if v is None:
            continue
        if video is None:
            video = np.lib.format.open_memmap(folder / f"{key}.video.tmp.npy", mode="w+", dtype=np.float32, shape=(len(paths), *v.shape))
            audio = np.lib.format.open_memmap(folder / f"{key}.audio.tmp.npy", mode="w+", dtype=np.float32, shape=(len(paths), *a.shape))
        if v.shape != video.shape[1:] or a.shape != audio.shape[1:]:
            raise ValueError(f"embedding shapes differ at {path}")
        video[len(kept)], audio[len(kept)] = v, a
        kept.append(path)
    video.flush(), audio.flush()
    del video, audio
    for kind in ("video", "audio"):
        (folder / f"{key}.{kind}.tmp.npy").replace(folder / f"{key}.{kind}.npy")
    temporary = folder / f"{key}.index.tmp"
    temporary.write_text(json.dumps(kept), encoding="utf-8")
    temporary.replace(folder / f"{key}.index.json")


def _shared_cache(paths: list, folder: Path, workers: int) -> _MappedFeatures:
    """The memory-mapped cache of `paths` under `folder`, built by the first process that needs it while the
    others wait for it. The file name is the hash of the path list."""
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1("\n".join(paths).encode("utf-8")).hexdigest()[:16]
    index_file, lock = folder / f"{key}.index.json", folder / f"{key}.lock"
    waiting = False
    while not index_file.exists():
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            if not waiting:
                print("feature cache: another run is building it, waiting", flush=True)
                waiting = True
            if time.time() - lock.stat().st_mtime > 3600:
                lock.unlink(missing_ok=True)
            time.sleep(5)
            continue
        try:
            if not index_file.exists():
                _build_shared_cache(paths, folder, key, workers)
        finally:
            lock.unlink(missing_ok=True)
    kept = json.loads(index_file.read_text(encoding="utf-8"))
    return _MappedFeatures({p: i for i, p in enumerate(kept)}, np.load(folder / f"{key}.video.npy", mmap_mode="r"), np.load(folder / f"{key}.audio.npy", mmap_mode="r"))


def install_feature_cache(record_sets: list, workers: int = 16, cache_dir: Path | None = None) -> int:
    """Makes every embedding file of the given records available to the old dataset class without reading the
    files again. With a cache folder the arrays are memory-mapped from one shared copy; without one, or when
    the shared copy cannot be built, each process holds its own float32 copy. Returns the number of files held.
    The training loop then reads memory instead of about 340 small files per batch."""
    import deepfake_detection.reference.data.dataset as dataset_module

    paths = sorted({str(f) for records in record_sets for row in records for f in row[0]})
    cache = None
    if cache_dir is not None:
        # every run over the same datasets maps the same files, whichever videos it trains and tests on
        roots = {Path(p).parents[1] for p in paths}
        paths = sorted(str(f) for root in roots for f in root.rglob("*.npz"))
        try:
            cache = _shared_cache(paths, cache_dir, workers)
        except Exception as error:
            print(f"feature cache: shared copy failed ({error}), reading into this process", flush=True)
    if cache is None:
        cache = {path: {"video_feat": v, "audio_feat": a} for path, v, a in _read_embeddings(paths, workers) if v is not None}
    dataset_module.np = _NumpyWithFeatureCache(cache)
    return len(cache)


def _save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def train_resumable(name, model, train_records, val_records, test_records, save_dir: Path, cfg: dict, domain_labels, norm_overrides, seed: int) -> dict:
    """The reference run_experiment with one addition: the full training state (model, optimiser,
    schedulers, scaler, SWA, best scores, patience) is written after every epoch, so a restarted
    run continues at the next epoch. Hyperparameters, losses, loaders and the final evaluation are
    the reference helpers. Per-epoch scores go to history.csv, progress with ETA to progress.log."""
    import deepfake_detection.reference.training.trainer as t
    from deepfake_detection.reference.training.engine import run_epoch
    from deepfake_detection.reference.training.losses import FocalLoss

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)

    get = lambda key, default: cfg.get(key, default)
    device = get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epochs, batch_size = get("epochs", 60), get("batch_size", 16)
    performance = get("performance", {})
    amp_enabled = bool(performance.get("amp", t.AMP))
    amp_dtype = t._amp_dtype(performance.get("amp_dtype", t.AMP_DTYPE))
    cache_norm = bool(performance.get("cache_norm_stats", t.CACHE_NORM_STATS))
    pin_enabled = bool(performance.get("pin_memory", t.PIN_MEMORY)) and device.type == "cuda"
    gate_weight = get("gate_supervision_weight", 0.1)

    assert {r[2] for r in train_records}.isdisjoint({r[2] for r in val_records} | {r[2] for r in test_records}), "subject leakage"
    sanity = get("sanity_fraction", None)
    if sanity is not None:
        keep = lambda records: records[: max(2, int(len(records) * sanity))]
        train_records, val_records, test_records = keep(train_records), keep(val_records), keep(test_records)

    extra_sets = [s["records"] for s in (norm_overrides or {}).get("extra_test_sets", [])]
    held = install_feature_cache([train_records, val_records, test_records, *extra_sets], cache_dir=feature_cache_dir)
    print(f"feature cache: {held} embedding files available")
    train_stats = t._compute_norm_stats(train_records, use_cache=cache_norm)
    primary_test_stats = t._compute_norm_stats(norm_overrides["test_records_for_stats"], use_cache=cache_norm) if norm_overrides and "test_records_for_stats" in norm_overrides else train_stats

    domain_ids = domain_labels.tolist() if domain_labels is not None else None
    train_loader, _ = t._make_loader(train_records, train_stats, batch_size, shuffle=True, modal_drop_p=get("modal_drop_p", 0.25),
                                     mixup_alpha=get("mixup_alpha", 0.2), feat_noise=get("feat_noise", 0.01), domain_ids=domain_ids, performance=performance)
    val_loader, _ = t._make_loader(val_records, train_stats, batch_size, shuffle=False, performance=performance)
    test_loader, _ = t._make_loader(test_records, primary_test_stats, batch_size, shuffle=False, performance=performance)

    criterion = FocalLoss(gamma=get("focal_gamma", 1.0), alpha=get("focal_alpha", 0.75), label_smoothing=get("label_smoothing", 0.05))
    base_model = model.to(device)
    num_domains = int(domain_labels.max().item() + 1) if domain_labels is not None and len(domain_labels) else None
    t._resize_domain_head(base_model, num_domains)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=get("lr", 1e-4), weight_decay=get("weight_decay", 1e-3))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")
    swa_model = AveragedModel(base_model)
    swa_start = int(epochs * get("swa_start", 0.75))
    swa_sched = SWALR(optimizer, swa_lr=get("swa_lr", 5e-5), anneal_epochs=5)
    state = {"epoch": 0, "swa_started": False, "best_f1": 0.0, "best_loss": float("inf"), "patience_counter": 0, "step": 0, "stopped": False}

    state_path = save_dir / "train_state.pt"
    if state_path.exists():
        # the state file is written by this function, so full unpickling of it is safe
        saved = torch.load(state_path, map_location=device, weights_only=False)
        base_model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        swa_model.load_state_dict(saved["swa_model"])
        swa_sched.load_state_dict(saved["swa_sched"])
        state = saved["progress"]
        print(f"resumed {name} after epoch {state['epoch']}")

    history_path = save_dir / "history.csv"
    if not history_path.exists():
        history_path.write_text("epoch,train_loss,train_f1,val_loss,val_f1,alpha,seconds\n", encoding="utf-8")
    total_steps = epochs * len(train_loader)
    bar = progress(epochs, state["epoch"], save_dir / "progress.log", f"{name} epochs")
    import time

    for epoch in range(state["epoch"] + 1, epochs + 1):
        if state["stopped"]:
            break
        started = time.time()
        lambda_ = get("dann_lambda", 0.1) * (2.0 / (1.0 + np.exp(-10 * state["step"] / max(total_steps, 1))) - 1.0)
        state["step"] += len(train_loader)
        tr = run_epoch(base_model, train_loader, criterion, optimizer, is_training=True, desc="Train", lambda_=lambda_,
                       gate_supervision_weight=gate_weight, amp_enabled=amp_enabled, amp_dtype=amp_dtype, scaler=scaler, non_blocking=pin_enabled)
        if epoch >= swa_start:
            swa_model.update_parameters(base_model)
            swa_sched.step()
            state["swa_started"] = True
        else:
            scheduler.step()
        vl = run_epoch(base_model, val_loader, criterion, optimizer=None, is_training=False, desc="Val  ", amp_enabled=amp_enabled,
                       amp_dtype=amp_dtype, non_blocking=pin_enabled)

        if vl["f1"] > state["best_f1"]:
            state["best_f1"], state["patience_counter"] = vl["f1"], 0
            torch.save(base_model.state_dict(), save_dir / "best_f1.pt")
        if vl["loss"] < state["best_loss"]:
            state["best_loss"] = vl["loss"]
            torch.save(base_model.state_dict(), save_dir / "best_loss.pt")
        else:
            state["patience_counter"] += 1
            state["stopped"] = state["patience_counter"] >= get("patience", 12)
        state["epoch"] = epoch

        with open(history_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{tr['loss']:.5f},{tr['f1']:.5f},{vl['loss']:.5f},{vl['f1']:.5f},{vl['alpha_mean'] if vl['alpha_mean'] is not None else ''},{time.time() - started:.1f}\n")
        _save_state(state_path, {"model": base_model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                                 "scaler": scaler.state_dict(), "swa_model": swa_model.state_dict(), "swa_sched": swa_sched.state_dict(), "progress": state})
        bar.update(1)
    bar.close()

    if state["swa_started"] and not (save_dir / "swa.pt").exists():
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), save_dir / "swa.pt")
    torch.save({"train_stats": train_stats, "primary_test_stats": primary_test_stats}, save_dir / "norm_stats.pt")

    results = {}
    for checkpoint in ("best_f1", "swa"):
        path = save_dir / f"{checkpoint}.pt"
        if not path.exists():
            continue
        base_model.load_state_dict(torch.load(path, map_location=device))
        res = run_epoch(base_model, test_loader, criterion, optimizer=None, is_training=False, desc="Test ", amp_enabled=amp_enabled,
                        amp_dtype=amp_dtype, non_blocking=pin_enabled)
        if checkpoint == "best_f1":
            t._save_alpha_analysis(save_dir, "primary", res)
        results[f"primary_{checkpoint}"] = {key: (round(res[key], 4) if res.get(key) is not None else None) for key in ("ap", "auc", "f1", "f1_opt", "thresh_opt", "alpha_mean")}
        results[f"primary_{checkpoint}"]["per_type_f1"] = res["per_type_f1"]
    return {"best_f1": state["best_f1"], "best_loss": state["best_loss"], "test_results": results}
