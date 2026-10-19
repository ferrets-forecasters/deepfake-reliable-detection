"""
Single entry point for all experiments.

Usage (from project root):
    uv run python -m deepfake_detection.reference.run configs/training/moe_fusion.yaml
    uv run python -m deepfake_detection.reference.run configs/training/moe_fusion.yaml \
        --override training.epochs=5 debug.sanity_fraction=0.1

Experiment types (data.experiment in YAML)
-------------------------------------------
  all_datasets_combined
                        Train on all 5 datasets (PolyglotFake + HiDF + FakeAVCeleb
                        + CelebDFv2 + FF++), each subject-split 60/20/20, combined.
                        DANN domain labels per source dataset.
                        Used by: moe_fusion, concat, audio_only, video_only.

  lodo_leave_polyglot   LODO: train on HiDF+FakeAVCeleb+CelebDFv2+FF++ (80% each).
                        Test = ALL PolyglotFake (held-out domain).

  lodo_leave_hidf       LODO: train on PolyglotFake+FakeAVCeleb+CelebDFv2+FF++ (80% each).
                        Test = ALL HiDF (held-out domain).

  lodo_leave_fakeavceleb  LODO: train on PolyglotFake+HiDF+CelebDFv2+FF++ (80% each).
                          Test = ALL FakeAVCeleb (held-out domain).

  lodo_leave_celebdfv2  LODO: train on PolyglotFake+HiDF+FakeAVCeleb+FF++ (80% each).
                        Test = ALL CelebDFv2 (held-out domain).

  lodo_leave_ffpp       LODO: train on PolyglotFake+HiDF+FakeAVCeleb+CelebDFv2 (80% each).
                        Test = ALL FF++ (held-out domain).

  ffpp_official_to_celebdfv2
                        Train on FF++ official split (video IDs 000-719, 4 standard
                        manipulation types only). Val = IDs 720-859.
                        Test = ALL CelebDFv2 (full benchmark, no split applied).
                        Audio zeroed, gate alpha forced visual.
                        Used by: comp_ffpp_celebdfv2_moe, comp_ffpp_celebdfv2_concat.

  ffpp5_official_to_celebdfv2
                        CADDM-matched variant of FF++ to CelebDFv2. Same official
                        split, but includes FaceShifter with the four standard types.

  fakeavceleb_70_30    CAD-matched FakeAVCeleb split. Deterministic subject split:
                        56% train, 14% validation, 30% test.
                        Full audio+video mode.

  fakeavceleb_indist    Train on FakeAVCeleb (subject_split 60/20/20).
                        Test = FakeAVCeleb 20% held-out test split.
                        Full audio+video mode (audio IS manipulation signal).
                        Internal same-split result only; not used as literature comparison.

Outputs
-------
  models/<name>/
      best_f1.pt   best_loss.pt   swa.pt   norm_stats.pt   latest.pt
      results.yaml
"""
import argparse
import hashlib
import random
import yaml
import torch
from pathlib import Path
from datetime import datetime

from deepfake_detection.reference.config import (
    POLYGLOT_ROOT, HIDF_ROOT, CELEBDFV2_ROOT, FAKEAVCELEB_ROOT, FFPP_ROOT,
    VAL_RATIO, SEED, DEVICE, CACHE_SPLITS, CACHE_DIR, SAVE_BASE,
)
from deepfake_detection.reference.data.dataset import (
    scan, subject_split,
    _extract_subject_polyglot, _extract_subject_hidf,
    _extract_subject_celebdfv2, _extract_subject_fakeavceleb, _extract_subject_ffpp,
)
from deepfake_detection.reference.models.factory import build_model
from deepfake_detection.reference.training.trainer import run_experiment

_SPLIT_MEMO = {}
_ROOT_FINGERPRINT_MEMO = {}
_SCAN_MEMO = {}
CACHE_VERSION = "splits_v2"


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """Apply dot-notation CLI overrides, e.g. "training.epochs=10"."""
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        key_path, raw_value = override.split("=", 1)
        keys   = key_path.split(".")
        target = cfg
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        if raw_value.lower() in ("null", "none", "~"):
            value = None
        elif raw_value.lower() == "true":
            value = True
        elif raw_value.lower() == "false":
            value = False
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
        target[keys[-1]] = value
    return cfg


def flatten_training_cfg(cfg: dict) -> dict:
    """Merge training + loss + swa + debug into a single flat dict."""
    flat = {}
    flat.update(cfg.get("training", {}))
    flat.update(cfg.get("loss",     {}))
    flat.update(cfg.get("swa",      {}))
    flat.update(cfg.get("debug",    {}))
    flat["performance"] = cfg.get("performance", {})
    flat["device"] = DEVICE
    return flat


def _root_fingerprint(root: Path) -> dict:
    root = Path(root)
    key = str(root.resolve()) if root.exists() else str(root)
    if key in _ROOT_FINGERPRINT_MEMO:
        return _ROOT_FINGERPRINT_MEMO[key]
    count = 0
    total_size = 0
    max_mtime = 0
    if root.exists():
        for p in root.rglob("*.npz"):
            try:
                st = p.stat()
            except OSError:
                continue
            count += 1
            total_size += st.st_size
            max_mtime = max(max_mtime, st.st_mtime_ns)
    fp = {"root": key, "count": count, "size": total_size, "max_mtime": max_mtime}
    _ROOT_FINGERPRINT_MEMO[key] = fp
    return fp


def _split_cache_key(data_cfg: dict) -> str:
    payload = {
        "version": CACHE_VERSION,
        "data": data_cfg,
        "roots": {
            "polyglot": _root_fingerprint(POLYGLOT_ROOT),
            "hidf": _root_fingerprint(HIDF_ROOT),
            "celebdfv2": _root_fingerprint(CELEBDFV2_ROOT),
            "fakeavceleb": _root_fingerprint(FAKEAVCELEB_ROOT),
            "ffpp": _root_fingerprint(FFPP_ROOT),
        },
    }
    raw = yaml.safe_dump(payload, sort_keys=True).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()


def scan_cached(root: Path, extract_fn, fake_type: str, audio_avail: bool):
    key = (
        str(Path(root).resolve()),
        getattr(extract_fn, "__name__", repr(extract_fn)),
        fake_type,
        bool(audio_avail),
        tuple(sorted(_root_fingerprint(root).items())),
    )
    if key not in _SCAN_MEMO:
        _SCAN_MEMO[key] = scan(root, extract_fn, fake_type=fake_type,
                               audio_avail=audio_avail)
    return _SCAN_MEMO[key]


FFPP_4_TYPES = frozenset(
    {"deepfakes", "face2face", "faceswap", "neuraltextures"}
)
FFPP_5_TYPES = FFPP_4_TYPES | {"faceshifter"}


def _ffpp_vid_id(record):
    subj = record[2]
    if not subj.startswith("ffpp_"):
        return None
    suffix = subj[5:]
    if suffix.startswith("dfd_") or suffix.startswith("unk_"):
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


def _ffpp_type(record):
    stem = record[0][0].stem.rsplit("_seg", 1)[0]
    return stem.split("_")[0]


def _ffpp_official_split(records, allowed_types, label):
    def in_range(record, lo, hi):
        vid_id = _ffpp_vid_id(record)
        if vid_id is None:
            return False
        if record[1] == 0:
            return lo <= vid_id <= hi
        return lo <= vid_id <= hi and _ffpp_type(record) in allowed_types

    train = [r for r in records if in_range(r, 0, 719)]
    val = [r for r in records if in_range(r, 720, 859)]

    real_tr = sum(1 for r in train if r[1] == 0)
    fake_tr = len(train) - real_tr
    real_val = sum(1 for r in val if r[1] == 0)
    fake_val = len(val) - real_val
    types = ", ".join(sorted(allowed_types))
    print(f"FF++ train ({label}, IDs 000-719):"
          f"  real={real_tr}  fake={fake_tr}  total={len(train)}")
    print(f"FF++ val   ({label}, IDs 720-859):"
          f"  real={real_val}  fake={fake_val}  total={len(val)}")
    print(f"  included fake types: {types}")
    print("  excluded fake types: deepfakedetection")
    return train, val


def _subject_split_56_14_30(records, seed):
    rng = random.Random(seed)
    subjects = sorted({r[2] for r in records})
    rng.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(n * 0.30))
    n_val = max(1, int(n * 0.14))
    test_subs = set(subjects[:n_test])
    val_subs = set(subjects[n_test:n_test + n_val])

    train = [r for r in records if r[2] not in test_subs and r[2] not in val_subs]
    val = [r for r in records if r[2] in val_subs]
    test = [r for r in records if r[2] in test_subs]

    train_subs = {r[2] for r in train}
    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(data_cfg: dict, performance: dict | None = None):
    """
    Return (train, val, test, domain_labels, norm_overrides).

    norm_overrides may contain:
        "extra_test_sets" — list of {name, records, stats_records} dicts
                            for evaluating unseen datasets after training
    """
    performance = performance or {}
    use_cache = bool(performance.get("cache_splits", CACHE_SPLITS))
    cache_key = _split_cache_key(data_cfg) if use_cache else None
    if cache_key and cache_key in _SPLIT_MEMO:
        return _SPLIT_MEMO[cache_key]
    cache_path = Path(CACHE_DIR) / "splits" / f"{cache_key}.pt" if cache_key else None
    if cache_path and cache_path.exists():
        try:
            result = torch.load(cache_path, map_location="cpu", weights_only=False)
            _SPLIT_MEMO[cache_key] = result
            print(f"[cache] splits loaded: {cache_path.name}")
            return result
        except Exception as e:
            print(f"[cache] ignoring stale split cache: {e}")

    val_ratio = data_cfg.get("val_ratio", VAL_RATIO)
    seed      = data_cfg.get("seed",      SEED)
    exp_type  = data_cfg.get("experiment", "joint_all_tests")

    # ── Scan training datasets ────────────────────────────────────────────────
    polyglot_all = scan_cached(POLYGLOT_ROOT, _extract_subject_polyglot,
                               fake_type="audio_video", audio_avail=True)
    hidf_all     = scan_cached(HIDF_ROOT, _extract_subject_hidf,
                               fake_type="video_only", audio_avail=True)

    polyglot_train, polyglot_val, polyglot_test = subject_split(
        polyglot_all, val_ratio, seed)
    hidf_train, hidf_val, hidf_test = subject_split(
        hidf_all, val_ratio, seed)

    print(f"PolyglotFake  train={len(polyglot_train)}  "
          f"val={len(polyglot_val)}  test={len(polyglot_test)}")
    print(f"HiDF          train={len(hidf_train)}  "
          f"val={len(hidf_val)}  test={len(hidf_test)}")

    # ── All 5 datasets combined — primary ablation experiment ────────────────
    # Scan FakeAVCeleb, CelebDFv2, FF++ for the combined ablation
    fac_all_ab = []
    if FAKEAVCELEB_ROOT.exists():
        fac_all_ab = scan_cached(FAKEAVCELEB_ROOT, _extract_subject_fakeavceleb,
                                 fake_type="audio_video", audio_avail=True)
    celeb_all_ab = []
    if CELEBDFV2_ROOT.exists():
        celeb_all_ab = scan_cached(CELEBDFV2_ROOT, _extract_subject_celebdfv2,
                                   fake_type="video_only", audio_avail=False)
    ffpp_all_ab = []
    if FFPP_ROOT.exists():
        ffpp_all_ab = scan_cached(FFPP_ROOT, _extract_subject_ffpp,
                                  fake_type="video_only", audio_avail=False)

    fac_tr_ab,   fac_vl_ab,   fac_ts_ab   = ([], [], [])
    celeb_tr_ab, celeb_vl_ab, celeb_ts_ab = ([], [], [])
    ffpp_tr_ab,  ffpp_vl_ab,  ffpp_ts_ab  = ([], [], [])
    if fac_all_ab:
        fac_tr_ab,   fac_vl_ab,   fac_ts_ab   = subject_split(fac_all_ab,   val_ratio, seed)
    if celeb_all_ab:
        celeb_tr_ab, celeb_vl_ab, celeb_ts_ab = subject_split(celeb_all_ab, val_ratio, seed)
    if ffpp_all_ab:
        ffpp_tr_ab,  ffpp_vl_ab,  ffpp_ts_ab  = subject_split(ffpp_all_ab,  val_ratio, seed)

    # ── Experiment-type routing ───────────────────────────────────────────────
    if exp_type == "all_datasets_combined":
        # Subject-split each dataset independently; combine all splits.
        # domain labels: 0=Polyglot, 1=HiDF, 2=FakeAVCeleb, 3=CelebDFv2, 4=FF++
        _ab_datasets = [
            ("PolyglotFake",  polyglot_train, polyglot_val, polyglot_test),
            ("HiDF",          hidf_train,     hidf_val,     hidf_test),
            ("FakeAVCeleb",   fac_tr_ab,      fac_vl_ab,    fac_ts_ab),
            ("CelebDFv2",     celeb_tr_ab,    celeb_vl_ab,  celeb_ts_ab),
            ("FF++",          ffpp_tr_ab,     ffpp_vl_ab,   ffpp_ts_ab),
        ]
        train, val, test = [], [], []
        domain_ids_list  = []
        for d_id, (name, tr, vl, ts) in enumerate(_ab_datasets):
            train += tr;  val += vl;  test += ts
            domain_ids_list.extend([d_id] * len(tr))
            print(f"  {name:<15} train={len(tr)}  val={len(vl)}  test={len(ts)}")
        domains = torch.tensor(domain_ids_list, dtype=torch.long)
        norm_overrides = None

    # ── Proper LODO: train on N-1 datasets, test on 1 unseen (full) ─────────────
    elif exp_type in ("lodo_leave_polyglot", "lodo_leave_hidf",
                      "lodo_leave_fakeavceleb", "lodo_leave_celebdfv2",
                      "lodo_leave_ffpp"):

        # Scan all 5 datasets
        fac_all = []
        if FAKEAVCELEB_ROOT.exists():
            fac_all = scan_cached(FAKEAVCELEB_ROOT, _extract_subject_fakeavceleb,
                                  fake_type="audio_video", audio_avail=True)

        celeb_all = []
        if CELEBDFV2_ROOT.exists():
            celeb_all = scan_cached(CELEBDFV2_ROOT, _extract_subject_celebdfv2,
                                    fake_type="video_only", audio_avail=False)

        ffpp_all = []
        if FFPP_ROOT.exists():
            ffpp_all = scan_cached(FFPP_ROOT, _extract_subject_ffpp,
                                   fake_type="video_only", audio_avail=False)

        # Split each training candidate 80/20 (train/val) — no test portion needed.
        # For LODO, the left-out dataset is the test; training datasets use
        # tr + ts combined for maximum training data (80%), val only for early stopping.
        poly_tr, poly_vl, poly_ts = subject_split(polyglot_all, val_ratio, seed)
        hidf_tr, hidf_vl, hidf_ts = subject_split(hidf_all,     val_ratio, seed)

        fac_tr, fac_vl, fac_ts   = [], [], []
        cel_tr, cel_vl, cel_ts   = [], [], []
        fp_tr,  fp_vl,  fp_ts    = [], [], []

        if fac_all:
            fac_tr, fac_vl, fac_ts = subject_split(fac_all,   val_ratio, seed)
        if celeb_all:
            cel_tr, cel_vl, cel_ts = subject_split(celeb_all, val_ratio, seed)
        if ffpp_all:
            fp_tr,  fp_vl,  fp_ts  = subject_split(ffpp_all,  val_ratio, seed)

        # Map: left-out dataset → (full_records, lodo_train=tr+ts, lodo_val=vl)
        # lodo_train uses tr+ts so 80% of subjects train, 20% for early-stopping val.
        # ts subjects are safe to include in train: they never appear in the left-out test.
        all_datasets = {
            "polyglot":    (polyglot_all, poly_tr + poly_ts, poly_vl),
            "hidf":        (hidf_all,     hidf_tr + hidf_ts, hidf_vl),
            "fakeavceleb": (fac_all,      fac_tr  + fac_ts,  fac_vl),
            "celebdfv2":   (celeb_all,    cel_tr  + cel_ts,  cel_vl),
            "ffpp":        (ffpp_all,     fp_tr   + fp_ts,   fp_vl),
        }

        left_out_key = exp_type.replace("lodo_leave_", "")
        test_all, _, _ = all_datasets[left_out_key]
        if not test_all:
            raise FileNotFoundError(
                f"Left-out dataset '{left_out_key}' not found. "
                "Check embeddings path in configs/config.py.")

        # Training pool = all other 4 datasets (DANN domain labels per source dataset)
        train, val = [], []
        train_domain_ids = []
        domain_counter   = 0
        for key, (_, tr, vl) in all_datasets.items():
            if key != left_out_key:
                train += tr
                val   += vl
                train_domain_ids.extend([domain_counter] * len(tr))
                domain_counter += 1

        # Test = ENTIRE left-out dataset (proper cross-domain OOD test)
        test    = test_all
        domains = (torch.tensor(train_domain_ids, dtype=torch.long)
                   if train_domain_ids else None)

        d_idx = 0
        for key, (ds_all, tr, vl) in all_datasets.items():
            if key == left_out_key:
                print(f"  {key:<15} TEST (left-out)  n={len(ds_all)}")
            else:
                print(f"  {key:<15} train domain={d_idx}  n={len(tr)}")
                d_idx += 1

        norm_overrides = None

    # ── FF++ official split → CelebDFv2 full (fair comparison with FTCN/CADDM/LAA-Net) ──
    elif exp_type in {"ffpp_official_to_celebdfv2", "ffpp5_official_to_celebdfv2"}:
        if not FFPP_ROOT.exists():
            raise FileNotFoundError(f"FF++ embeddings not found at {FFPP_ROOT}")
        if not CELEBDFV2_ROOT.exists():
            raise FileNotFoundError(f"CelebDFv2 embeddings not found at {CELEBDFV2_ROOT}")

        def _ffpp_vid_id(r):
            """Numeric video ID from subject key 'ffpp_042' → 42. None for dfd/unk."""
            subj = r[2]
            if not subj.startswith("ffpp_"):
                return None
            suffix = subj[5:]
            if suffix.startswith("dfd_") or suffix.startswith("unk_"):
                return None
            try:
                return int(suffix)
            except ValueError:
                return None

        def _ffpp_type(r):
            """Manipulation type prefix from first file stem, e.g. 'deepfakes'."""
            stem = r[0][0].stem.rsplit("_seg", 1)[0]   # "deepfakes_000_003"
            return stem.split("_")[0]                   # "deepfakes"

        def _ffpp_in_range(r, lo, hi):
            vid_id = _ffpp_vid_id(r)
            if vid_id is None:
                return False
            if r[1] == 0:   # real
                return lo <= vid_id <= hi
            # fake — also filter by manipulation type
            return lo <= vid_id <= hi and _ffpp_type(r) in FFPP_4_TYPES

        ffpp_all  = scan_cached(FFPP_ROOT, _extract_subject_ffpp,
                                fake_type="video_only", audio_avail=False)
        celeb_all = scan_cached(CELEBDFV2_ROOT, _extract_subject_celebdfv2,
                                fake_type="video_only", audio_avail=False)

        if exp_type == "ffpp5_official_to_celebdfv2":
            allowed_types = FFPP_5_TYPES
            label = "official 5 types"
        else:
            allowed_types = FFPP_4_TYPES
            label = "official 4 types"

        ffpp_train, ffpp_val = _ffpp_official_split(
            ffpp_all, allowed_types, label)

        train   = ffpp_train
        val     = ffpp_val
        test    = celeb_all   # full CelebDFv2 — no split applied
        domains = None
        norm_overrides = None

        print(f"CelebDFv2 test (full dataset):  {len(celeb_all)}")

    # ── FakeAVCeleb in-distribution (fair comparison with Joint AV / AVAD) ─────
    elif exp_type == "fakeavceleb_70_30":
        if not FAKEAVCELEB_ROOT.exists():
            raise FileNotFoundError(
                f"FakeAVCeleb embeddings not found at {FAKEAVCELEB_ROOT}")
        fac_all = scan_cached(FAKEAVCELEB_ROOT, _extract_subject_fakeavceleb,
                              fake_type="audio_video", audio_avail=True)
        train, val, test = _subject_split_56_14_30(fac_all, seed)
        domains = None
        norm_overrides = None
        print(f"FakeAVCeleb 70/30  train={len(train)}  val={len(val)}"
              f"  test={len(test)}")

    elif exp_type == "fakeavceleb_indist":
        if not FAKEAVCELEB_ROOT.exists():
            raise FileNotFoundError(
                f"FakeAVCeleb embeddings not found at {FAKEAVCELEB_ROOT}")
        fac_all = scan_cached(FAKEAVCELEB_ROOT, _extract_subject_fakeavceleb,
                              fake_type="audio_video", audio_avail=True)
        fac_train, fac_val, fac_test = subject_split(fac_all, val_ratio, seed)
        train   = fac_train
        val     = fac_val
        test    = fac_test
        domains = None
        norm_overrides = None
        print(f"FakeAVCeleb  train={len(train)}  val={len(val)}  test={len(test)}")

    else:
        raise ValueError(
            f"Unknown experiment type '{exp_type}'. "
            "Choose: all_datasets_combined | "
            "lodo_leave_polyglot | lodo_leave_hidf | lodo_leave_fakeavceleb | "
            "lodo_leave_celebdfv2 | lodo_leave_ffpp | "
            "ffpp_official_to_celebdfv2 | ffpp5_official_to_celebdfv2 | "
            "fakeavceleb_70_30 | fakeavceleb_indist"
        )

    result = (train, val, test, domains, norm_overrides)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, cache_path)
        _SPLIT_MEMO[cache_key] = result
        print(f"[cache] splits saved: {cache_path.name}")
    return result



# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run a deepfake detection experiment from a YAML config."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--override", nargs="*", default=[], metavar="key.sub=value")
    args = parser.parse_args()

    cfg = apply_overrides(load_yaml(args.config), args.override)

    exp_name  = cfg.get("name", args.config.stem)
    save_dir  = SAVE_BASE / exp_name
    train_cfg = flatten_training_cfg(cfg)

    print(f"\n{'='*60}")
    print(f"  Experiment : {exp_name}")
    print(f"  Model      : type={cfg['model']['type']}  "
          f"backbone={cfg['model'].get('backbone', 'n/a')}")
    print(f"  Data       : {cfg['data']['experiment']}")
    print(f"  Device     : {DEVICE}")
    print(f"{'='*60}\n")

    model    = build_model(cfg["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}\n")

    print("Scanning datasets...")
    train_records, val_records, test_records, domain_labels, norm_overrides = \
        load_splits(cfg["data"], cfg.get("performance", {}))

    result = run_experiment(
        name           = exp_name,
        model          = model,
        train_records  = train_records,
        val_records    = val_records,
        test_records   = test_records,
        save_dir       = save_dir,
        cfg            = train_cfg,
        domain_labels  = domain_labels,
        norm_overrides = norm_overrides,
    )

    summary = {
        "name":         exp_name,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "best_f1":      round(float(result["best_f1"]),   4),
        "best_loss":    round(float(result["best_loss"]), 4),
        "test_results": result.get("test_results", {}),
        "model":        cfg["model"],
        "data":         cfg["data"],
        "training":     cfg.get("training", {}),
        "loss":         cfg.get("loss", {}),
        "swa":          cfg.get("swa", {}),
        "performance":  cfg.get("performance", {}),
    }
    results_path = save_dir / "results.yaml"
    with open(results_path, "w", encoding="utf-8") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)

    print(f"\nResults saved to: {results_path}")
    print(f"  best_f1   = {summary['best_f1']}")
    print(f"  best_loss = {summary['best_loss']}")
    if result.get("test_results"):
        print("\n  Test results by dataset:")
        for ds_name, metrics in result["test_results"].items():
            alpha_str = f"  alpha={metrics['alpha_mean']:.3f}" if metrics.get("alpha_mean") is not None else ""
            print(f"    {ds_name:<20} AP={metrics['ap']:.4f}  "
                  f"AUC={metrics['auc']:.4f}  F1={metrics['f1']:.4f}{alpha_str}")


if __name__ == "__main__":
    main()
