"""
Reusable training / evaluation runner.

Multi-test support
───────────────────
When norm_overrides contains "extra_test_sets", the final evaluation runs on
each additional test set after the primary one.  Each entry is a dict:
    {"name": str, "records": list, "stats_records": list | None}

If stats_records is None, the primary train_stats are used.  For truly unseen
datasets (E3 FakeAVCeleb, E4 FF++, E5 CelebDFv2) stats_records = all records
of that dataset so it is normalised by its own distribution.

Reported metrics per test set
───────────────────────────────
    AP       — average precision (area under precision–recall curve)
    AUC      — area under ROC curve
    F1       — macro-F1 (real + fake averaged)
    α_mean   — mean gate weighting (MoE only; interpretability result)
               α ≈ 0 → sync expert dominant (audio-video fake)
               α ≈ 1 → visual expert dominant (video-only fake)
"""
from __future__ import annotations

import hashlib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import classification_report
from tqdm import tqdm

from deepfake_detection.reference.data.dataset import VideoSequenceDataset, collate_fn, compute_norm_stats
from deepfake_detection.reference.training.losses import FocalLoss
from deepfake_detection.reference.training.engine import run_epoch
from deepfake_detection.reference.config import (
    AMP, AMP_DTYPE, TORCH_COMPILE, NUM_WORKERS, PIN_MEMORY,
    PERSISTENT_WORKERS, PREFETCH_FACTOR, CACHE_NORM_STATS, CACHE_DIR,
)


def _resolve(cfg: dict, key: str, default):
    return cfg.get(key, default)


def _amp_dtype(name: str) -> torch.dtype:
    if str(name).lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


def _make_loader(records, stats, batch_size, shuffle, performance=None, **aug_kwargs):
    ds = VideoSequenceDataset(records, stats, **aug_kwargs)
    perf = performance or {}
    num_workers = int(perf.get("num_workers", NUM_WORKERS))
    pin_memory = bool(perf.get("pin_memory", PIN_MEMORY)) and torch.cuda.is_available()
    persistent = bool(perf.get("persistent_workers", PERSISTENT_WORKERS)) and num_workers > 0
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(perf.get("prefetch_factor", PREFETCH_FACTOR))
    return DataLoader(ds, **kwargs), ds


def _records_cache_key(records: list) -> str:
    h = hashlib.blake2b(digest_size=16)
    for row in records:
        h.update(str(row[1]).encode())
        h.update(str(row[2]).encode())
        h.update(str(row[3] if len(row) > 3 else "unknown").encode())
        h.update(str(row[4] if len(row) > 4 else True).encode())
        for f in row[0]:
            p = Path(f)
            try:
                st = p.stat()
                h.update(str(p.resolve()).encode())
                h.update(str(st.st_size).encode())
                h.update(str(st.st_mtime_ns).encode())
            except OSError:
                h.update(str(p).encode())
    return h.hexdigest()


def _compute_norm_stats(records: list, use_cache: bool = True):
    if not use_cache:
        return compute_norm_stats(records)
    cache_dir = Path(CACHE_DIR) / "norm_stats"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{_records_cache_key(records)}.pt"
    if cache_path.exists():
        try:
            return torch.load(cache_path, map_location="cpu", weights_only=False)
        except Exception:
            pass
    stats = compute_norm_stats(records)
    torch.save(stats, cache_path)
    return stats


def _resize_domain_head(model: nn.Module, num_domains: int | None):
    if not num_domains or num_domains <= 0 or not hasattr(model, "domain_head"):
        return
    head = getattr(model, "domain_head")
    if not isinstance(head, nn.Sequential) or not head:
        return
    last = head[-1]
    if not isinstance(last, nn.Linear) or last.out_features == num_domains:
        return
    head[-1] = nn.Linear(last.in_features, num_domains).to(last.weight.device)
    tqdm.write(f"  [dann] domain_head outputs={num_domains}")


def _save_alpha_analysis(save_dir: Path, test_name: str, result: dict):
    """
    Save per-sample interpretability arrays to <save_dir>/alpha_<test_name>.npz.

    Arrays saved
    ────────────
    alpha     : gate α per sample — how much the model relied on video vs audio
                (0 = audio/sync expert dominant, 1 = visual/artifact expert dominant)
                Shape [N], dtype float32.  All-zeros if model has no gate.
    labels    : true binary label per sample  (0=real, 1=fake)  shape [N] int32
    probs     : P(fake) score per sample       shape [N] float32
    manip_type: manipulation type code per sample  shape [N] int32
                codes defined in src.data.dataset.MANIP_TYPE_MAP:
                  0=real, 1=audio_video, 2=video_only, -1=unknown
    """
    safe_name = test_name.replace("/", "_").replace(" ", "_")
    out_path  = save_dir / f"alpha_{safe_name}.npz"
    alpha = result["alpha_arr"] if result["alpha_arr"] is not None \
            else np.zeros(len(result["labels_arr"]), dtype=np.float32)
    np.savez_compressed(
        out_path,
        alpha      = alpha,
        labels     = result["labels_arr"],
        probs      = result["probs_arr"],
        manip_type = result["manip_arr"],
    )
    tqdm.write(f"  [alpha] saved -> {out_path.name}  "
               f"(N={len(alpha)}  alpha_mean={alpha.mean():.3f})")


def _print_test_results(name: str, result: dict, ckpt_label: str):
    """Print a standardised test block for one test set."""
    tqdm.write(f"\n--- Test [{ckpt_label}]  {name} ---")
    opt_str = ""
    if abs(result.get("thresh_opt", 0.5) - 0.5) > 0.01:
        opt_str = (f"  F1_opt={result['f1_opt']:.4f}"
                   f"@t={result['thresh_opt']:.3f}")
    tqdm.write(f"  AP={result['ap']:.4f}  AUC={result['auc']:.4f}  "
               f"F1={result['f1']:.4f}{opt_str}"
               + (f"  alpha_mean={result['alpha_mean']:.3f}"
                  if result["alpha_mean"] is not None else ""))
    print(classification_report(result["labels"], result["preds"],
                                 target_names=["real", "fake"], labels=[0, 1], zero_division=0))
    if result["per_type_f1"]:
        tqdm.write("  Per-manipulation-type macro-F1:")
        for t, v in result["per_type_f1"].items():
            tqdm.write(f"    {t:<15} {v:.4f}")


def run_experiment(
    name:           str,
    model:          nn.Module,
    train_records:  list,
    val_records:    list,
    test_records:   list,
    save_dir:       Path,
    cfg:            dict,
    domain_labels:  torch.Tensor | None = None,
    norm_overrides: dict | None         = None,
) -> dict:
    """
    Train model, validate, and evaluate on primary + any extra test sets.

    Parameters
    ----------
    name            : experiment label
    model           : nn.Module from factory.build_model
    train_records   : list of 5-tuples from dataset.scan
    val_records     : same format
    test_records    : same format (primary test — used for checkpoint selection)
    save_dir        : directory for checkpoints and results
    cfg             : flat training hyperparameter dict
    domain_labels   : [N_train] LongTensor  per-sample domain IDs for DANN
    norm_overrides  : optional dict, may contain:
        "test_records_for_stats"  — records for computing target-domain norm stats
        "extra_test_sets"         — list of {name, records, stats_records} dicts
                                    for evaluating additional unseen test sets

    Returns
    -------
    dict with best_f1, best_loss, save_dir, and per-test-set results
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Hyperparameters ───────────────────────────────────────────────────────
    device          = _resolve(cfg, "device",               torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    epochs          = _resolve(cfg, "epochs",               60)
    batch_size      = _resolve(cfg, "batch_size",           16)
    lr              = _resolve(cfg, "lr",                   1e-4)
    weight_decay    = _resolve(cfg, "weight_decay",         1e-3)
    patience        = _resolve(cfg, "patience",             12)
    label_smoothing = _resolve(cfg, "label_smoothing",      0.05)
    modal_drop_p    = _resolve(cfg, "modal_drop_p",         0.25)
    feat_noise      = _resolve(cfg, "feat_noise",           0.01)
    mixup_alpha     = _resolve(cfg, "mixup_alpha",          0.2)
    dann_lambda     = _resolve(cfg, "dann_lambda",          0.1)
    focal_gamma     = _resolve(cfg, "focal_gamma",          1.0)
    focal_alpha     = _resolve(cfg, "focal_alpha",          0.75)
    swa_start_frac  = _resolve(cfg, "swa_start",            0.75)
    swa_lr          = _resolve(cfg, "swa_lr",               5e-5)
    sanity_fraction = _resolve(cfg, "sanity_fraction",      None)
    gate_sup_weight = _resolve(cfg, "gate_supervision_weight", 0.1)
    performance     = _resolve(cfg, "performance",          {})
    amp_enabled     = bool(performance.get("amp", AMP))
    amp_dtype       = _amp_dtype(performance.get("amp_dtype", AMP_DTYPE))
    compile_enabled = bool(performance.get("torch_compile", TORCH_COMPILE))
    cache_norm      = bool(performance.get("cache_norm_stats", CACHE_NORM_STATS))
    pin_enabled     = bool(performance.get("pin_memory", PIN_MEMORY)) and device.type == "cuda"

    # ── Header ────────────────────────────────────────────────────────────────
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Experiment : {name}")
    tqdm.write(f"  train={len(train_records)}  val={len(val_records)}  "
               f"test={len(test_records)}")
    tqdm.write(
        "  perf="
        f"amp={amp_enabled} dtype={str(amp_dtype).replace('torch.', '')} "
        f"compile={compile_enabled} workers={performance.get('num_workers', NUM_WORKERS)} "
        f"pin={pin_enabled} cache_norm={cache_norm}"
    )

    # ── Subject-leakage guard (primary test only) ─────────────────────────────
    train_subs = {r[2] for r in train_records}
    val_subs   = {r[2] for r in val_records}
    test_subs  = {r[2] for r in test_records}
    assert train_subs.isdisjoint(val_subs),  "Subject leakage: train/val"
    assert train_subs.isdisjoint(test_subs), "Subject leakage: train/test"
    assert val_subs.isdisjoint(test_subs),   "Subject leakage: val/test"

    # ── Sanity fraction ───────────────────────────────────────────────────────
    if sanity_fraction is not None:
        train_records = train_records[:max(2, int(len(train_records) * sanity_fraction))]
        val_records   = val_records  [:max(2, int(len(val_records)   * sanity_fraction))]
        test_records  = test_records [:max(2, int(len(test_records)  * sanity_fraction))]
        tqdm.write(f"[sanity] {sanity_fraction*100:.1f}%")

    # ── Normalisation ─────────────────────────────────────────────────────────
    tqdm.write("Computing norm stats from train split...")
    train_stats = _compute_norm_stats(train_records, use_cache=cache_norm)

    if norm_overrides and "test_records_for_stats" in norm_overrides:
        tqdm.write("Computing separate norm stats for primary test domain...")
        primary_test_stats = _compute_norm_stats(
            norm_overrides["test_records_for_stats"], use_cache=cache_norm)
    else:
        primary_test_stats = train_stats

    # ── Datasets & loaders ────────────────────────────────────────────────────
    domain_id_list = domain_labels.tolist() if domain_labels is not None else None
    train_loader, train_ds = _make_loader(
        train_records, train_stats, batch_size, shuffle=True,
        modal_drop_p=modal_drop_p, mixup_alpha=mixup_alpha, feat_noise=feat_noise,
        domain_ids=domain_id_list, performance=performance)
    val_loader,  val_ds  = _make_loader(
        val_records, train_stats, batch_size, shuffle=False,
        performance=performance)
    test_loader, test_ds = _make_loader(
        test_records, primary_test_stats, batch_size, shuffle=False,
        performance=performance)

    # ── Loss, optimiser, scheduler ────────────────────────────────────────────
    criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha,
                          label_smoothing=label_smoothing)
    base_model = model.to(device)
    num_domains = int(domain_labels.max().item() + 1) if domain_labels is not None and len(domain_labels) else None
    _resize_domain_head(base_model, num_domains)
    model = torch.compile(base_model) if compile_enabled and hasattr(torch, "compile") else base_model
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and device.type == "cuda")

    swa_model   = AveragedModel(base_model)
    swa_start   = int(epochs * swa_start_frac)
    swa_sched   = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)
    swa_started = False

    best_f1, best_loss = 0.0, float("inf")
    patience_counter   = 0
    total_steps        = epochs * len(train_loader)
    step               = 0

    # ── Training loop ─────────────────────────────────────────────────────────
    epoch_bar = tqdm(range(1, epochs + 1), desc=f"[{name}]", unit="epoch")
    for epoch in epoch_bar:
        p       = step / max(total_steps, 1)
        lambda_ = dann_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        step   += len(train_loader)

        tr = run_epoch(
            model, train_loader, criterion, optimizer,
            is_training=True, desc="Train",
            lambda_=lambda_,
            gate_supervision_weight=gate_sup_weight,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
            non_blocking=pin_enabled,
        )

        if epoch >= swa_start:
            swa_model.update_parameters(base_model)
            swa_sched.step()
            swa_started = True
        else:
            scheduler.step()

        vl = run_epoch(
            model, val_loader, criterion, optimizer=None,
            is_training=False, desc="Val  ",
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            non_blocking=pin_enabled,
        )

        postfix = dict(
            tr_loss=f"{tr['loss']:.4f}", tr_f1=f"{tr['f1']:.4f}",
            vl_loss=f"{vl['loss']:.4f}", vl_f1=f"{vl['f1']:.4f}",
        )
        if "audio_video" in vl["per_type_f1"]:
            postfix["vl_av"] = f"{vl['per_type_f1']['audio_video']:.4f}"
        if "video_only" in vl["per_type_f1"]:
            postfix["vl_vo"] = f"{vl['per_type_f1']['video_only']:.4f}"
        if vl["alpha_mean"] is not None:
            postfix["alpha"] = f"{vl['alpha_mean']:.3f}"
        if tr["dann_acc"] is not None:
            postfix["dann"] = f"{tr['dann_acc']:.3f}"
        epoch_bar.set_postfix(**postfix)

        # ── Checkpointing ──────────────────────────────────────────────────────
        torch.save(
            {"epoch": epoch, "model": base_model.state_dict(),
             "optimizer": optimizer.state_dict(),
             "train_stats": train_stats, "primary_test_stats": primary_test_stats},
            save_dir / "latest.pt",
        )
        if vl["f1"] > best_f1:
            best_f1          = vl["f1"]
            patience_counter = 0
            torch.save(base_model.state_dict(), save_dir / "best_f1.pt")
            vl_t = vl["per_type_f1"]
            tqdm.write(
                f"  [+] f1={best_f1:.4f}"
                + (f"  av={vl_t.get('audio_video','?')}"
                   f"  vo={vl_t.get('video_only','?')}" if vl_t else "")
                + (f"  a={vl['alpha_mean']:.3f}" if vl["alpha_mean"] is not None else "")
            )

        if vl["loss"] < best_loss:
            best_loss = vl["loss"]
            torch.save(base_model.state_dict(), save_dir / "best_loss.pt")
            tqdm.write(f"  [-] loss={best_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write(f"Early stopping at epoch {epoch}")
                break

    # ── SWA ───────────────────────────────────────────────────────────────────
    if swa_started:
        tqdm.write("Updating SWA batch norm...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        torch.save(swa_model.module.state_dict(), save_dir / "swa.pt")
        tqdm.write("SWA saved.")

    torch.save({"train_stats": train_stats, "primary_test_stats": primary_test_stats},
               save_dir / "norm_stats.pt")
    tqdm.write(f"Final  f1={best_f1:.4f}  loss={best_loss:.4f}")

    # ── Final test evaluation ─────────────────────────────────────────────────
    all_test_results = {}   # keyed by "primary" or dataset name

    ckpt_pairs = [("best_f1", save_dir / "best_f1.pt")]
    if (save_dir / "swa.pt").exists():
        ckpt_pairs.append(("swa", save_dir / "swa.pt"))

    # Extra test sets from norm_overrides (unseen datasets)
    extra_test_sets = []
    if norm_overrides and "extra_test_sets" in norm_overrides:
        extra_test_sets = norm_overrides["extra_test_sets"]

    for ckpt_name, ckpt_path in ckpt_pairs:
        if not ckpt_path.exists():
            continue
        base_model.load_state_dict(torch.load(ckpt_path, map_location=device))

        # Primary test set
        res = run_epoch(model, test_loader, criterion, optimizer=None,
                        is_training=False, desc="Test ",
                        amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                        non_blocking=pin_enabled)
        _print_test_results("Primary", res, ckpt_name)
        if ckpt_name == "best_f1":
            _save_alpha_analysis(save_dir, "primary", res)
        all_test_results[f"primary_{ckpt_name}"] = {
            "ap": round(res["ap"], 4), "auc": round(res["auc"], 4),
            "f1": round(res["f1"], 4),
            "f1_opt": round(res["f1_opt"], 4),
            "thresh_opt": round(res["thresh_opt"], 4),
            "alpha_mean": round(res["alpha_mean"], 4) if res["alpha_mean"] is not None else None,
            "per_type_f1": res["per_type_f1"],
        }

        # Extra unseen test sets (only evaluate from best_f1 checkpoint for brevity)
        if ckpt_name == "best_f1":
            for ts in extra_test_sets:
                ts_name       = ts["name"]
                ts_records    = ts["records"]
                ts_stats_recs = ts.get("stats_records") or ts_records
                tqdm.write(f"\nComputing norm stats for {ts_name}...")
                ts_stats = _compute_norm_stats(ts_stats_recs, use_cache=cache_norm)
                ts_loader, ts_ds = _make_loader(ts_records, ts_stats,
                                                batch_size, shuffle=False,
                                                performance=performance)
                ts_res = run_epoch(model, ts_loader, criterion, optimizer=None,
                                   is_training=False, desc=f"Test-{ts_name[:8]}",
                                   amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                                   non_blocking=pin_enabled)
                _print_test_results(ts_name, ts_res, ckpt_name)
                _save_alpha_analysis(save_dir, ts_name, ts_res)
                all_test_results[ts_name] = {
                    "ap": round(ts_res["ap"], 4), "auc": round(ts_res["auc"], 4),
                    "f1": round(ts_res["f1"], 4),
                    "f1_opt": round(ts_res["f1_opt"], 4),
                    "thresh_opt": round(ts_res["thresh_opt"], 4),
                    "alpha_mean": round(ts_res["alpha_mean"], 4) if ts_res["alpha_mean"] is not None else None,
                    "per_type_f1": ts_res["per_type_f1"],
                }

    return {
        "best_f1":      best_f1,
        "best_loss":    best_loss,
        "save_dir":     save_dir,
        "test_results": all_test_results,
    }
