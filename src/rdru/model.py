"""RDRU-Nyx — Core Model Architecture with all innovations."""
from __future__ import annotations
import math, random
from typing import Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from .config import ModelConfig

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=4096, ntk_alpha=1.0):
        super().__init__()
        base = 10000.0 * (ntk_alpha ** (dim / (dim - 2))) if ntk_alpha > 1.0 else 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        pos = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)
    def forward(self, q, k, offset=0):
        seq_len = q.size(2)
        cos = self.cos_cached[None, None, offset:offset+seq_len, :].to(q.device)
        sin = self.sin_cached[None, None, offset:offset+seq_len, :].to(q.device)
        cos, sin = cos.repeat_interleave(2, dim=-1), sin.repeat_interleave(2, dim=-1)
        return (q*cos)+(rotate_half(q)*sin), (k*cos)+(rotate_half(k)*sin)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.value_proj = nn.Linear(d_model, d_ff, bias=False)
        self.output_proj = nn.Linear(d_ff, d_model, bias=False)
    def forward(self, x):
        return self.output_proj(F.silu(self.gate_proj(x)) * self.value_proj(x))

class MoEFFN(nn.Module):
    def __init__(self, d_model, d_ff, n_experts, top_k):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])
    def forward(self, x):
        B, T, D = x.shape
        routing_logits = self.gate(x)
        routing_probs = F.softmax(routing_logits, dim=-1)
        topk_w, topk_i = torch.topk(routing_logits, self.top_k, dim=-1)
        topk_w = F.softmax(topk_w, dim=-1)
        # Load balancing loss
        tokens_per_expert = torch.zeros(self.n_experts, device=x.device)
        for k in range(self.top_k):
            idx = topk_i[..., k]
            for e in range(self.n_experts):
                tokens_per_expert[e] += (idx == e).float().sum()
        frac = tokens_per_expert / (B * T * self.top_k + 1e-6)
        imp = routing_probs.mean(dim=(0, 1))
        lb_loss = ((frac * imp).sum() * self.n_experts - 1.0).pow(2)
        # Forward
        out = torch.zeros_like(x)
        fx, fo = x.reshape(-1, D), out.reshape(-1, D)
        fw, fi = topk_w.reshape(-1, self.top_k), topk_i.reshape(-1, self.top_k)
        for k in range(self.top_k):
            ei = fi[:, k]; w = fw[:, k:k+1]
            for eid, expert in enumerate(self.experts):
                mask = ei == eid
                if mask.any(): fo[mask] += w[mask] * expert(fx[mask])
        return out, lb_loss

class GQA(nn.Module):
    def __init__(self, d_model, n_query_heads, n_kv_heads, max_seq_len, use_qk_norm=True, ntk_alpha=1.0):
        super().__init__()
        assert d_model % n_query_heads == 0
        self.n_query_heads, self.n_kv_heads = n_query_heads, n_kv_heads
        self.head_dim = d_model // n_query_heads
        self.n_query_groups = n_query_heads // n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.use_qk_norm = use_qk_norm
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len, ntk_alpha)
    def forward(self, x, causal_mask=None, kv_cache=None, return_cache=False):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_query_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if kv_cache is not None:
            kc, vc = kv_cache; k = torch.cat([kc, k], dim=2); v = torch.cat([vc, v], dim=2)
        new_cache = (k, v) if return_cache else None
        offset = 0 if kv_cache is None else kv_cache[0].size(2)
        k = k.repeat_interleave(self.n_query_groups, dim=1)
        v = v.repeat_interleave(self.n_query_groups, dim=1)
        q, k = self.rope(q, k, offset)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if causal_mask is not None and kv_cache is None:
            attn = attn.masked_fill(causal_mask == 0, float("-inf"))
        out = F.softmax(attn, dim=-1) @ v
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(out), new_cache

class ACTGate(nn.Module):
    def __init__(self, d_model, epsilon=0.01):
        super().__init__(); self.epsilon = epsilon
        self.gate = nn.Linear(d_model, 1)
    def forward(self, hidden, halting_prob=None):
        p = torch.sigmoid(self.gate(hidden))
        if halting_prob is None: halting_prob = torch.zeros_like(p)
        halting_prob = halting_prob + p
        return p, halting_prob, (halting_prob >= (1.0 - self.epsilon)).all()

class RRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = GQA(config.d_model, config.n_query_heads, config.n_kv_heads,
                        config.max_seq_len, config.use_qk_norm, config.rope_ntk_alpha)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ffn = MoEFFN(config.d_model, config.d_model * config.d_ff_multiplier,
                          config.n_experts, config.top_k_experts)
    def forward(self, x, causal_mask=None, kv_cache=None, return_cache=False):
        ao, nc = self.attn(self.norm1(x), causal_mask, kv_cache, return_cache)
        x = x + ao
        fo, ml = self.ffn(self.norm2(x))
        return x + fo, nc, ml

class RDRUNyx(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.input_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rru = RRU(config)
        self.act_gate = ACTGate(config.d_model) if config.use_act else None
        self.denoise_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.use_weight_tying:
            self.output_projection.weight = self.token_embedding.weight
        self._init_weights()
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=self.config.weight_init_std)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m is self.output_projection and self.config.use_weight_tying: continue
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, input_ids, return_denoising_loss=False, curriculum_step=None, return_uncertainty=False):
        B, T = input_ids.shape; device = input_ids.device
        causal_mask = torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)
        hidden = self.input_projection(self.token_embedding(input_ids))
        if curriculum_step is not None:
            n_steps = min(curriculum_step, self.config.n_reasoning_steps)
        elif self.training:
            n_steps = random.randint(self.config.n_reasoning_steps // 2, self.config.n_reasoning_steps)
        else:
            n_steps = self.config.n_reasoning_steps
        if not self.training and self.act_gate is not None and not return_denoising_loss:
            return (self._forward_act(hidden, causal_mask, n_steps),)
        dl = torch.tensor(0.0, device=device); ml = torch.tensor(0.0, device=device)
        for step in range(n_steps):
            if self.training and return_denoising_loss:
                clean = hidden.detach()
                noise = torch.randn_like(hidden) * self.config.denoising_noise_std
                dl = dl + F.mse_loss(self.denoise_projection(hidden + noise), clean)
            if self.training and self.config.use_gradient_checkpointing:
                hidden, _, moe_l = torch.utils.checkpoint.checkpoint(
                    self.rru, hidden, causal_mask, None, False, use_reentrant=False)
            else:
                hidden, _, moe_l = self.rru(hidden, causal_mask)
            ml = ml + moe_l
        logits = self.output_projection(hidden)
        if return_denoising_loss: return logits, dl / n_steps, ml / n_steps
        if return_uncertainty:
            probs = F.softmax(logits, dim=-1)
            return logits, 1.0 - probs.topk(3, dim=-1)[0][:, :, 0]
        return logits
    def _forward_act(self, hidden, causal_mask, max_steps):
        hp = None
        for step in range(max_steps):
            p, hp, ah = self.act_gate(hidden, hp)
            if ah and step > 0: break
            hidden, _, _ = self.rru(hidden, causal_mask)
        return self.output_projection(hidden)
    @torch.no_grad()
    def generate(self, prompt_ids, max_new_tokens, temperature=0.8, top_k=40,
                 top_p=None, repetition_penalty=1.0, use_kv_cache=True, return_logprobs=False):
        self.eval(); device = prompt_ids.device; ids = prompt_ids.clone()
        eff_k = min(top_k or self.config.vocab_size, self.config.vocab_size)
        lps = [] if return_logprobs else None
        for step in range(max_new_tokens):
            inp = ids[:, -1:] if (use_kv_cache and step > 0) else ids
            logits = self(inp)[0, -1, :]
            if repetition_penalty > 1.0:
                for tid in set(ids[0].tolist()):
                    if logits[tid] < 0: logits[tid] *= repetition_penalty
                    else: logits[tid] /= repetition_penalty
            if temperature < 1e-6:
                nid = logits.argmax(dim=-1, keepdim=True).unsqueeze(0)
            else:
                logits = logits / temperature
                if eff_k < self.config.vocab_size:
                    vals, _ = torch.topk(logits, eff_k)
                    logits[logits < vals[-1]] = float("-inf")
                if top_p is not None and top_p < 1.0:
                    sl, si = torch.sort(logits, descending=True)
                    cp = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                    rm = cp > top_p; rm[1:] = rm[:-1].clone(); rm[0] = False
                    logits[si[rm]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nid = torch.multinomial(probs, 1).unsqueeze(0)
            ids = torch.cat([ids, nid], dim=1)
            if return_logprobs: lps.append(F.softmax(logits, dim=-1).log()[nid.item()])
        return (ids, torch.tensor(lps, device=device)) if return_logprobs else ids

class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay; self.shadow = {}; self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow[n] + (1-self.decay) * p.data
    def apply_shadow(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])
