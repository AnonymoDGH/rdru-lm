"""Configuration dataclasses for RDRU-Nyx model architecture."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

@dataclass
class ModelConfig:
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
    use_weight_tying: bool = True
    use_act: bool = False
    use_load_balancing: bool = True
    load_balancing_coef: float = 0.01
    use_gradient_checkpointing: bool = True
    use_qk_norm: bool = True
    rope_ntk_alpha: float = 1.0
    n_think_tokens: int = 0
    def save(self, path): json.dump(asdict(self), open(path, "w"), indent=2)
    @classmethod
    def load(cls, path): return cls(**json.load(open(path)))

@dataclass
class TrainingConfig:
    batch_size: int = 16
    seq_len: int = 512
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    denoising_loss_weight: float = 0.02
    n_epochs: int = 3
    target_chars: int = 50_000_000
    checkpoint_path: str = "rdru_nyx_checkpoint.pth"
    log_interval: int = 1000
    device: Optional[str] = None
    gradient_accumulation_steps: int = 1
    use_amp: bool = True
    warmup_steps: int = 100
    swa_start_epoch: int = 999
    ema_decay: float = 0.995
    curriculum_steps_start: int = 1
    curriculum_steps_increment: int = 1
    repetition_penalty: float = 1.0
    def save(self, path): json.dump(asdict(self), open(path, "w"), indent=2)
    @classmethod
    def load(cls, path): return cls(**json.load(open(path)))
