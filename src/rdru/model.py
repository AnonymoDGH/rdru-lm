"""RDRUv2 model definition.

References:
    - RoPE: https://arxiv.org/abs/2104.09864
    - GQA: https://arxiv.org/abs/2305.13245
    - SwiGLU: https://arxiv.org/abs/2002.05202
    - MoE: https://arxiv.org/abs/1701.06538
"""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Swaps the two halves of the last dimension and negates the first half.

    This is the core rotation operation used by RoPE:
        rotate_half([x1, x2]) = [-x2, x1]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes relative position information by applying a rotation to query
    and key tensors. Unlike learned position embeddings, RoPE is translation-
    invariant and supports arbitrary sequence lengths at inference time.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.size(2)
        cos = self.cos_cached[None, None, offset : offset + seq_len, :].to(q.device)
        sin = self.sin_cached[None, None, offset : offset + seq_len, :].to(q.device)

        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)

        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward block.

    Applies a gated activation: SiLU(x @ W_g) * (x @ W_v), projected back to
    the model dimension via W_p. Empirically outperforms standard ReLU FFNs
    at equivalent parameter counts.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.output_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))


class MoEFFN(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer.

    Routes each token to the top-k experts via a learned gating network.
    Only the selected experts are evaluated, making the total compute
    independent of the total number of experts.
    """

    def __init__(self, d_model: int, d_ff: int, n_experts: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        routing_logits = self.gate(x)
        top_k_weights, top_k_indices = torch.topk(routing_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        out = torch.zeros_like(x)
        flat_x = x.reshape(-1, D)
        flat_out = out.reshape(-1, D)
        flat_weights = top_k_weights.reshape(-1, self.top_k)
        flat_indices = top_k_indices.reshape(-1, self.top_k)

        for k in range(self.top_k):
            expert_idx = flat_indices[:, k]
            weight = flat_weights[:, k : k + 1]
            for expert_id, expert in enumerate(self.experts):
                mask = expert_idx == expert_id
                if mask.any():
                    flat_out[mask] += weight[mask] * expert(flat_x[mask])

        return out


class GQA(nn.Module):
    """Grouped Query Attention with Rotary Position Embeddings.

    Projects to fewer key/value heads than query heads, reducing memory
    and compute at inference time. Each key/value head is shared by
    ``n_query_heads // n_kv_heads`` query heads.
    """

    def __init__(self, d_model: int, n_query_heads: int, n_kv_heads: int, max_seq_len: int):
        super().__init__()
        assert d_model % n_query_heads == 0, "d_model must be divisible by n_query_heads"

        self.n_query_heads = n_query_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_query_heads
        self.n_query_groups = n_query_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)

        q, k = self.rope(q, k)

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        if causal_mask is not None:
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out)


class RRU(nn.Module):
    """Recurrent Reasoning Unit.

    A single transformer block applied iteratively by the RDRU model.
    Pre-norm architecture: LayerNorm → GQA → residual → LayerNorm → MoE-FFN → residual.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = GQA(config.d_model, config.n_query_heads, config.n_kv_heads, config.max_seq_len)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = MoEFFN(
            config.d_model,
            config.d_model * config.d_ff_multiplier,
            config.n_experts,
            config.top_k_experts,
        )

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class RDRUv2(nn.Module):
    """Recursive Denoising Reasoning Unit v2.

    Embeds character indices into a dense representation, then applies a shared
    RRU block for a variable number of reasoning steps. An auxiliary denoising
    objective trains the hidden state to be robust to perturbations, encouraging
    the model to converge toward a fixed point across iterations.

    During training, the number of steps is sampled uniformly from
    ``[n_reasoning_steps // 2, n_reasoning_steps]`` per forward pass.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rru = RRU(config)
        self.denoise_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=self.config.weight_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_denoising_loss: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        B, T = input_ids.shape

        causal_mask = torch.tril(torch.ones(T, T, device=input_ids.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        hidden = self.input_projection(self.token_embedding(input_ids))

        denoising_loss = torch.tensor(0.0, device=input_ids.device)
        n_steps = (
            random.randint(self.config.n_reasoning_steps // 2, self.config.n_reasoning_steps)
            if self.training
            else self.config.n_reasoning_steps
        )

        for _ in range(n_steps):
            if self.training and return_denoising_loss:
                clean = hidden.detach()
                noise = torch.randn_like(hidden) * self.config.denoising_noise_std
                denoising_loss = denoising_loss + F.mse_loss(
                    self.denoise_projection(hidden + noise), clean
                )
            hidden = self.rru(hidden, causal_mask)

        logits = self.output_projection(hidden)
        if return_denoising_loss:
            return logits, denoising_loss / n_steps
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            prompt_ids: Tensor of shape ``(1, prompt_len)`` with the initial token IDs.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from the top-k highest probability tokens.

        Returns:
            Tensor of shape ``(1, prompt_len + max_new_tokens)`` with the full sequence.
        """
        self.eval()
        device = prompt_ids.device
        ids = prompt_ids.clone()
        top_k = min(top_k or self.config.vocab_size, self.config.vocab_size)

        for _ in range(max_new_tokens):
            logits = self(ids)[0, -1, :] / temperature

            if top_k < self.config.vocab_size:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[-1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).unsqueeze(0)
            ids = torch.cat([ids, next_id], dim=1)

        return ids
