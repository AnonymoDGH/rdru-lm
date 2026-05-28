"""Tests for configuration dataclasses."""

import json
import tempfile
from pathlib import Path

from src.rdru.config import ModelConfig, TrainingConfig


def test_model_config_defaults() -> None:
    cfg = ModelConfig()
    assert cfg.d_model == 256
    assert cfg.n_query_heads == 8
    assert cfg.n_kv_heads == 4


def test_model_config_serialization() -> None:
    cfg = ModelConfig(vocab_size=100, d_model=128)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        cfg.save(f.name)
        loaded = ModelConfig.load(f.name)
    assert loaded.vocab_size == 100
    assert loaded.d_model == 128


def test_training_config_serialization() -> None:
    cfg = TrainingConfig(batch_size=32, n_epochs=5)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        cfg.save(f.name)
        loaded = TrainingConfig.load(f.name)
    assert loaded.batch_size == 32
    assert loaded.n_epochs == 5
