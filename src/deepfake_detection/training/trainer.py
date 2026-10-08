from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detection.config import Config
from deepfake_detection.evaluation import compute_metrics
from deepfake_detection.modeling import MoEFusionClassifier
from deepfake_detection.training.losses import DomainAdversarialLoss, FocalLoss, GateRoutingAuxLoss
from deepfake_detection.utils.checkpoint import save_checkpoint
from deepfake_detection.utils.logging import build_logger
from deepfake_detection.utils.seed import set_seed


class Trainer:
    """Trains MoEFusionClassifier or ConcatFusionClassifier on cached, frozen-encoder
    features. AdamW + cosine annealing until swa_start_epoch, then SWALR; early
    stopping on validation F1; best and SWA checkpoints written to models/checkpoints/<run_name>/."""

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        run_name: str,
        device: torch.device,
        use_domain_adversarial: bool = False,
    ):
        set_seed(config.seed)
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.run_name = run_name
        self.is_moe = isinstance(model, MoEFusionClassifier)
        self.use_domain_adversarial = use_domain_adversarial

        self.optimizer = AdamW(self.model.parameters(), lr=config.train.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.train.max_epochs)
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=config.train.learning_rate)

        self.main_loss = FocalLoss(config.train.focal_gamma, config.train.focal_alpha, config.train.label_smoothing)
        self.aux_loss = (
            GateRoutingAuxLoss(config.train.focal_gamma, config.train.focal_alpha, config.train.label_smoothing)
            if self.is_moe
            else None
        )
        self.domain_loss = DomainAdversarialLoss() if use_domain_adversarial else None

        self.checkpoint_dir = config.paths.checkpoints_dir(run_name)
        self.logger = build_logger(run_name, config.paths.logs_dir(run_name))

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        targets = batch["label"].to(self.device).float()
        concat_repr = batch["concat_repr"].to(self.device)

        if self.is_moe:
            visual_seq = batch["visual_seq"].to(self.device)
            audio_seq = batch["audio_seq"].to(self.device)
            domain_input = concat_repr if self.use_domain_adversarial else None
            out = self.model(visual_seq, audio_seq, domain_input)
            loss = self.main_loss(out.final_logit, targets)
            loss = loss + self.config.train.gate_routing_aux_weight * self.aux_loss(
                out.visual_logit, out.audio_logit, targets
            )
        else:
            out = self.model(concat_repr)
            loss = self.main_loss(out.final_logit, targets)

        if self.use_domain_adversarial:
            domain_labels = batch["domain_label"].to(self.device).long()
            loss = loss + self.config.train.domain_adversarial_weight * self.domain_loss(
                out.domain_logits, domain_labels
            )

        probs = torch.sigmoid(out.final_logit)
        return loss, probs

    def _run_epoch(self, loader: DataLoader, train: bool, epoch: int) -> tuple[float, np.ndarray, np.ndarray]:
        self.model.train(train)
        all_probs, all_targets = [], []
        total_loss = 0.0
        desc = f"epoch {epoch} [{'train' if train else 'val'}]"
        for batch in tqdm(loader, desc=desc, leave=False):
            with torch.set_grad_enabled(train):
                loss, probs = self._forward(batch)
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * len(batch["label"])
            all_probs.append(probs.detach().cpu().numpy())
            all_targets.append(batch["label"].numpy())
        avg_loss = total_loss / len(loader.dataset)
        return avg_loss, np.concatenate(all_probs), np.concatenate(all_targets)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Path:
        best_f1 = -1.0
        epochs_without_improvement = 0
        best_checkpoint_path = self.checkpoint_dir / "best.pt"

        for epoch in range(1, self.config.train.max_epochs + 1):
            train_loss, _, _ = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_loss, val_probs, val_targets = self._run_epoch(val_loader, train=False, epoch=epoch)
            val_metrics = compute_metrics(val_probs, val_targets)

            if epoch >= self.config.train.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.scheduler.step()

            self.logger.info(
                "epoch %d train_loss=%.4f val_loss=%.4f val_f1=%.4f val_ap=%.4f val_auc=%.4f",
                epoch,
                train_loss,
                val_loss,
                val_metrics.f1,
                val_metrics.ap,
                val_metrics.auc,
            )

            if val_metrics.f1 > best_f1:
                best_f1 = val_metrics.f1
                epochs_without_improvement = 0
                save_checkpoint(
                    {"epoch": epoch, "model_state": self.model.state_dict(), "val_f1": best_f1},
                    self.checkpoint_dir,
                    "best.pt",
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.train.early_stopping_patience:
                    self.logger.info("early stopping at epoch %d, best val_f1=%.4f", epoch, best_f1)
                    break

        save_checkpoint({"model_state": self.swa_model.module.state_dict()}, self.checkpoint_dir, "swa.pt")
        return best_checkpoint_path
