"""Configuration dataclasses for model architecture and training hyperparameters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional


@dataclass
class ModelConfig:
    """Transformer model configuration.

    Attributes:
        vocab_size: Number of characters in the vocabulary.
        d_model: Embedding and hidden state dimension.
        n_query_heads: Number of query heads in GQA.
        n_kv_heads: Number of key/value heads in GQA.
        n_reasoning_steps: Maximum number of iterative RRU applications.
        n_experts: Number of MoE feed-forward experts.
        top_k_experts: Number of active experts per token.
        d_ff_multiplier: Hidden dimension multiplier for feed-forward layers.
        max_seq_len: Maximum supported sequence length.
        denoising_noise_std: Standard deviation of Gaussian noise for denoising loss.
        weight_init_std: Standard deviation for embedding weight initialization.
    """

    vocab_size: int = 91
    d_model: int = 256
    n_query_heads: int = 8
    n_kv_heads: int = 4
    n_reasoning_steps: int = 8
    n_experts: int = 4
    top_k_experts: int = 2
    d_ff_multiplier: int = 3
    max_seq_len: int = 4096
    denoising_noise_std: float = 0.05
    weight_init_std: float = 0.02

    def save(self, path: str | Path) -> None:
        """Serialize configuration to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        batch_size: Number of sequences per training step.
        seq_len: Number of tokens per sequence.
        learning_rate: Peak learning rate for AdamW.
        weight_decay: Weight decay coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        denoising_loss_weight: Coefficient for the auxiliary denoising loss.
        n_epochs: Number of full passes over the training data.
        target_chars: Target corpus size in characters (for automatic dataset building).
        checkpoint_path: Path for saving model checkpoints.
        log_interval: Log training metrics every N steps.
        device: Training device (auto-detected if None).
    """

    batch_size: int = 16
    seq_len: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    denoising_loss_weight: float = 0.02
    n_epochs: int = 3
    target_chars: int = 100_000_000
    checkpoint_path: str = "rdru_checkpoint.pth"
    log_interval: int = 1000
    device: Optional[str] = None

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
