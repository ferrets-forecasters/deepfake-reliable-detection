"""
Training and evaluation engine.

Compatible with any model whose forward signature is:
    model(x, mask, lengths, lambda_=0.0, audio_avail=None)
        → (logit [B,2], domain_logit [B,2] | None)

MoEFusionClassifier additionally supports:
    model(..., return_gate=True) → (logit, domain_logit, alpha [B])

Gate supervision
─────────────────
audio_video fakes (PolyglotFake) → alpha target = 0.15  (sync expert)
video_only  fakes (HiDF)         → alpha target = 0.85  (visual expert)
real samples                     → skipped

Returns
───────
run_epoch returns a dict:
    loss        : float
    f1          : float  macro-F1 at threshold 0.5
    f1_opt      : float  macro-F1 at Youden-optimal threshold (useful for OOD)
    thresh_opt  : float  the optimal decision threshold
    ap          : float  average precision (area under PR curve)
    auc         : float  area under ROC curve
    preds       : list[int]
    labels      : list[int]
    probs       : list[float]  P(fake) for each sample
    per_type_f1 : dict  {"audio_video": float, "video_only": float}
                        Each type evaluated vs. ALL real samples so both classes present.
    alpha_mean  : float | None   mean gate α across all samples (MoE only)
    dann_acc    : float | None   DANN discriminator accuracy (training only)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, roc_curve
from tqdm import tqdm

from deepfake_detection.reference.config import DEVICE, DANN_LAMBDA
from deepfake_detection.reference.data.dataset import MANIP_TYPE_MAP

# ── Gate supervision constants ────────────────────────────────────────────────
GATE_TARGET_AUDIO_VIDEO = 0.15
GATE_TARGET_VIDEO_ONLY  = 0.85


def _gate_supervision_loss(alpha, manip_type_batch, weight: float) -> torch.Tensor:
    """
    Soft BCE between gate α and manipulation-type targets.
    Real samples are skipped (no supervision signal for gate on real).
    """
    CODE_AV = MANIP_TYPE_MAP["audio_video"]
    CODE_VO = MANIP_TYPE_MAP["video_only"]

    targets = torch.full_like(alpha, fill_value=-1.0)
    targets[manip_type_batch == CODE_AV] = GATE_TARGET_AUDIO_VIDEO
    targets[manip_type_batch == CODE_VO] = GATE_TARGET_VIDEO_ONLY

    mask = targets >= 0.0
    if not mask.any():
        return torch.tensor(0.0, device=alpha.device)

    with torch.autocast(device_type=alpha.device.type, enabled=False):
        return weight * F.binary_cross_entropy(
            alpha[mask].float(), targets[mask].float())


def run_epoch(model, loader, criterion, optimizer, is_training: bool, desc: str,
              lambda_: float = 0.0, gate_supervision_weight: float = 0.0,
              amp_enabled: bool = False, amp_dtype: torch.dtype = torch.float16,
              scaler=None, non_blocking: bool = False):
    """
    Run one full epoch (train or eval).

    Parameters
    ----------
    model                   : nn.Module
    loader                  : DataLoader — yields (x, labels, lengths, audio_avail, domain_ids)
    criterion               : loss callable(logits, labels) → scalar
    optimizer               : Optimizer or None (eval mode)
    is_training             : bool
    desc                    : tqdm label
    lambda_                 : DANN gradient reversal coefficient
    gate_supervision_weight : weight on gate supervision BCE loss
                              (manip_type_ids now come directly from each batch)

    Returns
    -------
    dict with keys: loss, f1, ap, auc, preds, labels, probs, per_type_f1, alpha_mean
    """
    model.train() if is_training else model.eval()

    total_loss       = 0.0
    all_preds        = []
    all_labels       = []
    all_probs        = []   # P(fake) for AP / AUC
    all_manip        = []
    all_alpha        = []   # gate α values (MoE only)
    all_dann_preds   = []   # DANN discriminator predictions
    all_dann_labels  = []   # DANN ground-truth domain IDs
    domain_crit      = nn.CrossEntropyLoss()
    ctx              = torch.enable_grad() if is_training else torch.no_grad()

    has_gate = gate_supervision_weight > 0.0 and hasattr(model, "gate")

    with ctx:
        for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False)):
            x, labels, lengths, audio_avail, domain_ids_batch, manip_batch = batch
            x = x.to(DEVICE, non_blocking=non_blocking)
            labels = labels.to(DEVICE, non_blocking=non_blocking)
            audio_avail = audio_avail.to(DEVICE, non_blocking=non_blocking)
            mask = (
                torch.arange(x.shape[1], device=DEVICE).unsqueeze(0)
                < lengths.unsqueeze(1).to(DEVICE, non_blocking=non_blocking)
            )

            lam = lambda_ if is_training else 0.0

            use_amp = bool(amp_enabled and DEVICE.type == "cuda")
            with torch.autocast(device_type=DEVICE.type, dtype=amp_dtype, enabled=use_amp):
                # ── Forward ───────────────────────────────────────────────────
                if has_gate and is_training:
                    logits, domain_logits, alpha = model(
                        x, mask, lengths, lambda_=lam,
                        return_gate=True, audio_avail=audio_avail)
                else:
                    logits, domain_logits = model(
                        x, mask, lengths, lambda_=lam, audio_avail=audio_avail)
                    alpha = None

                # Collect gate α for reporting (eval pass too)
                if alpha is None and not is_training and hasattr(model, "gate"):
                    _, _, alpha = model(
                        x, mask, lengths, lambda_=0.0,
                        return_gate=True, audio_avail=audio_avail)

                # ── Losses ────────────────────────────────────────────────────
                loss = criterion(logits, labels)

                if (is_training and lam > 0.0 and domain_logits is not None):
                    d_labels = domain_ids_batch.to(
                        DEVICE, non_blocking=non_blocking)
                    valid_mask = d_labels >= 0
                    if valid_mask.any():
                        loss = loss + DANN_LAMBDA * domain_crit(
                            domain_logits[valid_mask], d_labels[valid_mask])
                    if valid_mask.any():
                        all_dann_preds.extend(
                            domain_logits[valid_mask].argmax(1).cpu().tolist())
                        all_dann_labels.extend(
                            d_labels[valid_mask].cpu().tolist())

                if alpha is not None and is_training and gate_supervision_weight > 0.0:
                    loss = loss + _gate_supervision_loss(
                        alpha, manip_batch.to(DEVICE, non_blocking=non_blocking),
                        gate_supervision_weight)

            # ── Backward ──────────────────────────────────────────────────────
            if is_training and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            probs = F.softmax(logits, dim=1)[:, 1]   # P(fake)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probs.extend(probs.detach().cpu().tolist())

            if alpha is not None:
                all_alpha.extend(alpha.detach().cpu().tolist())

            all_manip.extend(manip_batch.tolist())

    n            = len(all_labels)
    macro_f1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_prec     = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    try:
        roc_auc  = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    except Exception:
        roc_auc  = 0.0

    # ── Per-manipulation-type F1 ──────────────────────────────────────────────
    # Each type is evaluated against ALL real samples so both classes are present.
    per_type_f1: dict = {}
    arr_labels = np.array(all_labels)
    arr_preds  = np.array(all_preds)
    arr_manip  = np.array(all_manip)
    REAL_CODE  = MANIP_TYPE_MAP["real"]
    for type_name, code in [("audio_video", MANIP_TYPE_MAP["audio_video"]),
                              ("video_only",  MANIP_TYPE_MAP["video_only"])]:
        mask_t = (arr_manip == code) | (arr_manip == REAL_CODE)
        if mask_t.sum() >= 4 and len(np.unique(arr_labels[mask_t])) > 1:
            per_type_f1[type_name] = round(float(
                f1_score(arr_labels[mask_t], arr_preds[mask_t],
                         average="macro", zero_division=0)
            ), 4)

    # ── Optimal-threshold F1 (Youden's J) ────────────────────────────────────
    # Useful for OOD sets where the 0.5 default is miscalibrated.
    opt_f1, opt_thresh = macro_f1, 0.5
    if len(set(all_labels)) > 1:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        best_idx   = int(np.argmax(tpr - fpr))
        opt_thresh = float(thresholds[best_idx])
        opt_preds  = (np.array(all_probs) >= opt_thresh).astype(int).tolist()
        opt_f1     = float(f1_score(all_labels, opt_preds, average="macro", zero_division=0))

    # ── DANN discriminator accuracy ───────────────────────────────────────────
    dann_acc = (
        float(np.mean(np.array(all_dann_preds) == np.array(all_dann_labels)))
        if all_dann_preds else None
    )

    alpha_mean = float(np.mean(all_alpha)) if all_alpha else None

    return {
        "loss":        total_loss / n,
        "f1":          macro_f1,
        "f1_opt":      opt_f1,
        "thresh_opt":  opt_thresh,
        "ap":          float(avg_prec),
        "auc":         float(roc_auc),
        "preds":       all_preds,
        "labels":      all_labels,
        "probs":       all_probs,
        "per_type_f1": per_type_f1,
        "alpha_mean":  alpha_mean,
        "dann_acc":    dann_acc,
        # ── Per-sample arrays for interpretability analysis ───────────────────
        # alpha_arr  : gate α per sample (MoE only; None if not applicable)
        # labels_arr : true label per sample  (0=real, 1=fake)
        # probs_arr  : P(fake) score per sample
        # manip_arr  : manipulation type code per sample (MANIP_TYPE_MAP)
        "alpha_arr":  np.array(all_alpha, dtype=np.float32) if all_alpha else None,
        "labels_arr": np.array(all_labels, dtype=np.int32),
        "probs_arr":  np.array(all_probs,  dtype=np.float32),
        "manip_arr":  np.array(all_manip,  dtype=np.int32),
    }
