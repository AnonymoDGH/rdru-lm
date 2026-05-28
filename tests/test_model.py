"""Tests for the RDRUv2 model definition."""

import torch
import pytest

from src.rdru import ModelConfig, RDRUv2


@pytest.fixture
def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=16,
        d_model=32,
        n_query_heads=4,
        n_kv_heads=2,
        n_reasoning_steps=2,
        n_experts=2,
        top_k_experts=1,
        max_seq_len=128,
    )


def test_model_creation(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    assert model is not None
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0


def test_forward_shape(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    B, T = 2, 16
    x = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits = model(x)
    assert logits.shape == (B, T, tiny_config.vocab_size)


def test_forward_with_denoising_loss(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    model.train()
    B, T = 2, 16
    x = torch.randint(0, tiny_config.vocab_size, (B, T))
    logits, denoise_loss = model(x, return_denoising_loss=True)
    assert logits.shape == (B, T, tiny_config.vocab_size)
    assert denoise_loss.item() >= 0.0


def test_generate_shape(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    model.eval()
    prompt = torch.zeros((1, 4), dtype=torch.long)
    output = model.generate(prompt, max_new_tokens=8, temperature=1.0, top_k=4)
    assert output.shape == (1, 12)


def test_generate_deterministic_with_temp_zero(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    model.eval()
    prompt = torch.zeros((1, 4), dtype=torch.long)
    out1 = model.generate(prompt, max_new_tokens=4, temperature=0.0, top_k=None)
    out2 = model.generate(prompt, max_new_tokens=4, temperature=0.0, top_k=None)
    assert torch.equal(out1, out2)


def test_model_train_eval_modes(tiny_config: ModelConfig) -> None:
    model = RDRUv2(tiny_config)
    assert model.training
    model.eval()
    assert not model.training
    model.train()
    assert model.training
