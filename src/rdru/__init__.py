"""RDRU — Recursive Denoising Reasoning Unit.

A character-level language model that applies the same transformer block
iteratively with an auxiliary denoising objective, combining GQA, RoPE,
and sparse MoE feed-forward.
"""

from .config import ModelConfig, TrainingConfig
from .model import RDRUv2
from .data import CharDataset, build_gsm8k_corpus, build_large_corpus
from .trainer import Trainer

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "RDRUv2",
    "CharDataset",
    "build_gsm8k_corpus",
    "build_large_corpus",
    "Trainer",
]
