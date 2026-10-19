"""
Dataset utilities for the Modality-Aware Adaptive Fusion project.

Record format (5-tuple)
───────────────────────
    (files, label_id, subject_key, manip_type, audio_avail)

    files       : list[Path]  – segment .npz files in temporal order
    label_id    : int         – 0 = real, 1 = fake
    subject_key : str         – identity key used for subject-level splitting
    manip_type  : str         – "real" | "audio_video" | "video_only" | "unknown"
    audio_avail : bool        – True if audio carries manipulation signal;
                                False for datasets where audio is unrelated
                                (CelebDFv2, FF++) — audio features are zeroed

manipulation_type semantics
────────────────────────────
    "real"        – genuine sample
    "audio_video" – both modalities manipulated (PolyglotFake: TTS + lip-sync)
    "video_only"  – only video manipulated, audio is genuine (HiDF: face-swap)
    "unknown"     – type not tracked for this dataset

audio_avail=False signals
──────────────────────────
When False, audio features are zeroed in __getitem__ so the AudioSyncExpert
cannot learn spurious signals from background noise.  The MoE gate additionally
overrides α=1.0 for those samples to route entirely through the VisualArtifactExpert.
"""
import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

from deepfake_detection.reference.config import AUDIO_DIM, VIDEO_DIM, MODAL_DROP_P, FEAT_NOISE, SEED

LABELS = {"real": 0, "fake": 1}

# Integer codes used inside batches
MANIP_TYPE_MAP = {"real": 0, "audio_video": 1, "video_only": 2, "unknown": -1}


# ── Subject-key extractors ────────────────────────────────────────────────────

def _extract_subject_polyglot(stem: str, label_name: str) -> str:
    """
    PolyglotFake: real_ar_10 / fake_ar_10_to_en_MicroTts
    The numeric speaker ID (parts[2]) maps across all languages.
    """
    parts = stem.split("_")
    if len(parts) >= 3:
        return parts[2]
    return stem


def _extract_subject_hidf(stem: str, label_name: str) -> str:
    """
    HiDF: c00003 (real) / c00003_c11507 (fake).
    First component = target identity; used consistently for both.
    """
    base = stem.rsplit("_seg", 1)[0]
    return base.split("_")[0]


def _extract_subject_celebdfv2(stem: str, label_name: str) -> str:
    """
    CelebDFv2 embeddings:
        real/  id0_0000_seg*.npz          → subject "celeb_id0"
        real/  00000_seg*.npz (YouTube)   → subject "celeb_yt_00000"
        fake/  id0_id16_0000_seg*.npz     → subject "celeb_id0" (target face)

    Prefix "celeb_" prevents collisions with other dataset subject keys.
    """
    base  = stem.rsplit("_seg", 1)[0]
    parts = base.split("_")
    if parts[0].startswith("id"):
        return f"celeb_{parts[0]}"          # Celeb-real or Celeb-synthesis target
    else:
        return f"celeb_yt_{base}"           # YouTube-real


def _extract_subject_fakeavceleb(stem: str, label_name: str) -> str:
    """
    FakeAVCeleb — subject key = source clip number (parts[0]).

    Ensures real clip and ALL its fake derivatives share the same subject key,
    so no original/fake pair is ever split across train/val/test splits.

    Real (embeddings/real/):
        00001_seg000          → parts[0] = "00001" → "fac_00001"

    FakeVideo (embeddings/fake/):
        00001_0_id06152_wavtolip_seg000  → parts[0] = "00001" → "fac_00001"

    RealVideo-FakeAudio (embeddings/fake/):
        00001_0_seg000        → parts[0] = "00001" → "fac_00001"

    All three map to the same subject key → always land in the same split.
    Prefix "fac_" prevents cross-dataset subject key collisions.
    """
    base  = stem.rsplit("_seg", 1)[0]
    parts = base.split("_")
    return f"fac_{parts[0]}"   # source clip number, consistent across all variants


def _extract_subject_ffpp(stem: str, label_name: str) -> str:
    """
    FF++ embeddings — source-identity grouping so real and fake from the same
    source video always land in the same train/val/test split.

    Real:
        000_seg*.npz               → "ffpp_000"
    Standard fakes (3-digit IDs):
        deepfakes_000_003_seg*.npz → "ffpp_000"  (source = first numeric field)
        face2face_000_003_seg*.npz → "ffpp_000"
        faceswap_000_003_seg*.npz  → "ffpp_000"
        neuraltextures_000_003_seg*.npz → "ffpp_000"
    DeepFakeDetection (2-digit actor IDs, separate namespace):
        deepfakedetection_01_02__..._seg*.npz → "ffpp_dfd_01"
    """
    base  = stem.rsplit("_seg", 1)[0]
    parts = base.split("_")
    if parts[0].isdigit():
        return f"ffpp_{parts[0]}"
    nums = [p for p in parts[1:] if p.isdigit()]
    if nums:
        src = nums[0]
        if len(src) <= 2:
            return f"ffpp_dfd_{src}"
        return f"ffpp_{src}"
    return f"ffpp_unk_{base[:10]}"


# ── Dataset scanning ──────────────────────────────────────────────────────────

def scan(root, extract_fn, fake_type: str = "unknown", audio_avail: bool = True):
    """
    Walk *root* (expecting root/real/ and root/fake/ subdirs).

    Returns a list of 5-tuple records:
        (segment_file_list, label_id, subject_key, manip_type, audio_avail)

    Parameters
    ----------
    root        : Path  dataset embeddings root (contains real/ and fake/)
    extract_fn  : callable(stem, label_name) → subject_key
    fake_type   : str   manipulation type tag for fake samples
                        ("audio_video" | "video_only" | "unknown")
    audio_avail : bool  whether audio is a meaningful signal in this dataset
                        Set False for CelebDFv2 and FF++
    """
    records = []
    for label_name, label_id in LABELS.items():
        manip_type = "real" if label_name == "real" else fake_type
        folder     = root / label_name
        if not folder.exists():
            continue
        grouped = defaultdict(list)
        for f in sorted(folder.glob("*.npz")):
            base = f.stem.rsplit("_seg", 1)[0]
            grouped[base].append(f)
        for base, files in sorted(grouped.items()):
            files   = sorted(files, key=lambda p: int(p.stem.rsplit("_seg", 1)[-1]))
            subject = extract_fn(base, label_name)
            records.append((files, label_id, subject, manip_type, audio_avail))
    return records


def subject_split(records, val_ratio, seed):
    """
    Split records into train / val / test by subject identity.
    No subject ever appears in more than one split.
    Compatible with both 4-tuple and 5-tuple record formats.
    """
    rng      = random.Random(seed)
    subjects = sorted({r[2] for r in records})
    rng.shuffle(subjects)
    n         = max(1, int(len(subjects) * val_ratio))
    val_subs  = set(subjects[:n])
    test_subs = set(subjects[n : 2 * n])
    train = [r for r in records if r[2] not in val_subs and r[2] not in test_subs]
    val   = [r for r in records if r[2] in val_subs]
    test  = [r for r in records if r[2] in test_subs]
    return train, val, test


# ── Normalisation ─────────────────────────────────────────────────────────────

def compute_norm_stats(records):
    """
    Compute per-feature mean and std from records (call on training split only).
    Compatible with 4-tuple and 5-tuple record formats.
    """
    vf, af = [], []
    for row in tqdm(records, desc="Norm stats", unit="video", leave=False):
        for f in row[0]:
            try:
                d = np.load(f, allow_pickle=True)
                vf.append(d["video_feat"].astype(np.float32))
                af.append(d["audio_feat"].astype(np.float32))
                d.close()
            except Exception:
                pass
    v = np.stack(vf)
    a = np.stack(af)
    return {
        "video_mean": v.mean(0),
        "video_std":  v.std(0).clip(1e-6),
        "audio_mean": a.mean(0),
        "audio_std":  a.std(0).clip(1e-6),
    }


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoSequenceDataset(Dataset):
    """
    Variable-length sequences of (video + audio) feature vectors.

    Each item: tensor [T, VIDEO_DIM + AUDIO_DIM].

    Exposes:
        self.manip_type_labels  : list[int]   MANIP_TYPE_MAP codes per sample
        self.audio_avail_flags  : list[bool]  per-sample audio availability

    When audio_avail=False for a record, audio features are zeroed in __getitem__
    so the AudioSyncExpert cannot pick up spurious background-noise signals.

    Augmentations (training only):
        modal_drop_p  – randomly zero-out one modality with this probability
        mixup_alpha   – beta-distribution alpha for temporal mixup (0 = off)
        feat_noise    – std of Gaussian noise added to features (0 = off)
    """

    def __init__(self, records, stats, modal_drop_p=0.0, mixup_alpha=0.0, feat_noise=0.0,
                 domain_ids=None):
        self.records      = records
        self.stats        = stats
        self.modal_drop_p = modal_drop_p
        self.mixup_alpha  = mixup_alpha
        self.feat_noise   = feat_noise
        self.labels       = [r[1] for r in records]
        # Backward-compatible: 4-tuple records have no audio_avail (default True)
        self.audio_avail_flags = [
            bool(r[4]) if len(r) > 4 else True for r in records
        ]
        self.manip_type_labels = [
            MANIP_TYPE_MAP.get(r[3] if len(r) > 3 else "unknown", -1)
            for r in records
        ]
        # Per-sample domain IDs for DANN (−1 = unknown / not set)
        self.domain_id_labels = domain_ids if domain_ids is not None else [-1] * len(records)

    def _load(self, idx):
        files, label_id   = self.records[idx][0], self.records[idx][1]
        audio_avail       = self.audio_avail_flags[idx]
        segs = []
        for f in files:
            try:
                d     = np.load(f, allow_pickle=True)
                vfeat = (d["video_feat"].astype(np.float32)
                         - self.stats["video_mean"]) / self.stats["video_std"]
                if audio_avail:
                    afeat = (d["audio_feat"].astype(np.float32)
                             - self.stats["audio_mean"]) / self.stats["audio_std"]
                else:
                    afeat = np.zeros(AUDIO_DIM, dtype=np.float32)
                d.close()
                segs.append(np.concatenate([vfeat, afeat]))
            except Exception:
                pass
        if not segs:
            segs = [np.zeros(VIDEO_DIM + AUDIO_DIM, dtype=np.float32)]
        return np.stack(segs), label_id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        x, y          = self._load(idx)
        audio_avail   = self.audio_avail_flags[idx]

        if self.feat_noise > 0.0:
            x = x + np.random.randn(*x.shape).astype(np.float32) * self.feat_noise

        if self.modal_drop_p > 0.0:
            r = random.random()
            if r < self.modal_drop_p:
                x[:, :VIDEO_DIM] = 0.0
            elif r < self.modal_drop_p * 2:
                x[:, VIDEO_DIM:] = 0.0

        if self.mixup_alpha > 0.0 and random.random() < 0.5:
            j       = random.randrange(len(self.records))
            x2, y2  = self._load(j)
            lam     = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            t       = min(x.shape[0], x2.shape[0])
            x       = lam * x[:t] + (1 - lam) * x2[:t]
            y       = y if lam >= 0.5 else y2

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(audio_avail, dtype=torch.bool),
                torch.tensor(self.domain_id_labels[idx],  dtype=torch.long),
                torch.tensor(self.manip_type_labels[idx], dtype=torch.long))


def collate_fn(batch):
    """Pad variable-length sequences.

    Returns: (padded, labels, lengths, audio_avail, domain_ids, manip_type_ids)
    """
    seqs, labels, avails, dom_ids, manip_ids = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)
    return (padded, torch.stack(labels), lengths,
            torch.stack(avails), torch.stack(dom_ids), torch.stack(manip_ids))
