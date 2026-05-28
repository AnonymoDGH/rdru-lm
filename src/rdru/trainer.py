"""Training loop for RDRUv2."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .config import ModelConfig, TrainingConfig
from .model import RDRUv2

logger = logging.getLogger(__name__)


class Trainer:
    """Handles the training loop, checkpointing, and logging.

    Args:
        model_config: Model architecture configuration.
        training_config: Training hyperparameters.
    """

    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        self.device = training_config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = RDRUv2(model_config).to(self.device)
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("model parameters: %d", n_params)
        logger.info("device: %s", self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=training_config.n_epochs
        )
        self.criterion = nn.CrossEntropyLoss()

    def train(self, dataloader: DataLoader) -> dict[str, float]:
        """Run the full training loop.

        Args:
            dataloader: DataLoader providing ``(input_ids, target_ids)`` batches.

        Returns:
            Dictionary with final metrics (``avg_loss`` per epoch).
        """
        metrics = {}

        for epoch in range(1, self.training_config.n_epochs + 1):
            epoch_loss = self._run_epoch(dataloader, epoch)
            self.scheduler.step()
            metrics[f"epoch_{epoch}_avg_loss"] = epoch_loss
            logger.info("epoch %d average loss: %.4f", epoch, epoch_loss)

        self._save_checkpoint()
        return metrics

    def _run_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            logits, denoise_loss = self.model(x, return_denoising_loss=True)

            ce_loss = self.criterion(
                logits.view(-1, self.model_config.vocab_size), y.view(-1)
            )
            loss = ce_loss + self.training_config.denoising_loss_weight * denoise_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_config.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if batch_idx % self.training_config.log_interval == 0:
                logger.info(
                    "epoch %d  batch %d/%d  loss=%.4f  ce=%.4f  denoise=%.4f",
                    epoch,
                    batch_idx,
                    len(dataloader),
                    loss.item(),
                    ce_loss.item(),
                    denoise_loss.item(),
                )

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self) -> None:
        path = self.training_config.checkpoint_path
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "training_config": self.training_config,
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        logger.info("checkpoint saved to %s", path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        training_config: Optional[TrainingConfig] = None,
    ) -> Trainer:
        """Load a saved model and resume training.

        Args:
            checkpoint_path: Path to a ``.pth`` checkpoint file.
            training_config: New training config (uses saved one if None).

        Returns:
            Trainer instance with loaded model weights.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_config = checkpoint["model_config"]
        if training_config is None:
            training_config = checkpoint.get("training_config", TrainingConfig())

        trainer = cls(model_config, training_config)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("loaded checkpoint from %s", checkpoint_path)
        return trainer
