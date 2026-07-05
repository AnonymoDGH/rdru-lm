<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/params-2.7M-orange" alt="2.7M parameters">
  <img src="https://img.shields.io/badge/weight_tying-✓-brightgreen" alt="Weight Tying">
  <img src="https://img.shields.io/badge/QK_norm-✓-brightgreen" alt="QK Norm">
</p>

<h1 align="center">RDRU — Recursive Denoising Reasoning Unit</h1>

<p align="center">
  A character-level language model that applies the <b>same transformer block iteratively</b>,
  paired with an <b>auxiliary denoising objective</b>. Designed for mathematical reasoning
  at small scales (~2.7M parameters).
</p>

---

## ✨ Key Ideas

- **Recursive reasoning** — a single RRU block is applied 4–8 times per token, allowing the
  hidden state to refine its representation iteratively, instead of stacking many distinct layers.
- **Denoising auxiliary loss** — MSE between the hidden state and a perturbed version of itself
  encourages convergence toward a stable fixed point.
- **Grouped Query Attention (GQA)** — 8 query heads, 4 key/value heads, with Rotary Position
  Embeddings (RoPE).
- **Sparse MoE-FFN** — 4 SwiGLU experts, top-2 routing, increasing capacity without a
  proportional increase in compute.

---

## 🆕 What's New (v2.1)

| Improvement | Description | Benefit |
|---|---|---|
| **Weight Tying** | Embedding and output projection share weights | -23K params, better representations |
| **QK-Norm** | LayerNorm on Q/K before attention (PaLM-style) | Stable training, no loss spikes |
| **NTK-Aware RoPE** | Frequency scaling for longer sequences | Extrapolates beyond `max_seq_len` |
| **Load-Balancing MoE** | Auxiliary loss penalizes expert imbalance | Prevents expert collapse |
| **Gradient Checkpointing** | Trades compute for memory in the RRU loop | ~8× less VRAM during training |
| **AMP Support** | Automatic Mixed Precision (FP16/BF16) | ~2× faster on modern CPUs/GPUs |
| **Gradient Accumulation** | Accumulates gradients over N steps | Large effective batch on limited RAM |
| **EMA** | Exponential Moving Average of weights | Better generalization at inference |
| **Curriculum Learning** | Progressive reasoning depth | Starts at 1 step, grows gradually |
| **Top-p / Repetition Penalty** | Nucleus sampling + repeat penalty | Higher-quality generations |
| **KV-Cache Ready** | Structure prepared for cached generation | O(T) instead of O(T²) generation |
| **Temperature = 0 Safe** | Argmax fallback instead of division by zero | No more crashes at `temp=0` |

---

## 🏗️ Architecture

| Component | Description | Reference |
|---|---|---|
| `RotaryEmbedding` | Rotation-based relative position encoding | [RoFormer](https://arxiv.org/abs/2104.09864) |
| `GroupedQueryAttention` | Grouped Query Attention (8 Q-heads, 4 KV-heads) | [GQA](https://arxiv.org/abs/2305.13245) |
| `MoEFeedForward` | Sparse Mixture-of-Experts (4 experts, top-2) | [MoE](https://arxiv.org/abs/1701.06538) |
| `SwiGLU` | Gated feed-forward activation | [GLU Variants](https://arxiv.org/abs/2002.05202) |
| `RRUBlock` | Single shared transformer block, applied iteratively | — |
| `RDRUModel` | Top-level model: embedding + RRU loop + denoising head | — |

### Parameter count (default config)

| Component | Parameters |
|---|---|
| Token embedding | 23,296 |
| Attention (QKV + output) | 139,264 |
| MoE-FFN (4 SwiGLU experts) | 1,575,424 |
| Layer norms & projections | 1,005,149 |
| **Total** | **≈ 2,735,616** |

---

## 📦 Installation

```bash
git clone https://github.com/<your-user>/rdru.git
cd rdru
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, 🤗 `datasets` 2.14+.

---

## 🚀 Usage

### Training on GSM8K

```bash
python train.py \
  --config configs/base.json \
  --dataset gsm8k \
  --output_dir checkpoints/base
```

### Configurations

| Config | Params | d_model | Experts | RRU steps | RAM | Time/epoch (CPU) |
|---|---|---|---|---|---|---|
| `configs/tiny.json` | ~7K | 64 | 2 | 4 | ~256 MB | ~2 min |
| `configs/small.json` | ~176K | 128 | 4 | 6 | ~512 MB | ~15 min |
| `configs/base.json` | ~2.7M | 256 | 4 | 8 | ~1.5 GB | ~45 min |

### Generation

```bash
python generate.py \
  --checkpoint checkpoints/base/best.pt \
  --prompt "Janet has 3 apples. She buys 5 more. How many apples does she have?" \
  --max_new_tokens 200 \
  --top_p 0.9 \
  --repetition_penalty 1.2
```

### Evaluation

```bash
python evaluate.py \
  --checkpoint checkpoints/base/best.pt \
  --dataset gsm8k \
  --split test
```

### Running tests

```bash
pytest tests/ -v
```

---

## ⚙️ Configuration

Model and training hyperparameters are defined as `RDRUConfig` objects and
serialized to JSON, following the same pattern as
[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3).

```python
from rdru.config import RDRUConfig

config = RDRUConfig(
    vocab_size=128,
    d_model=256,
    n_heads=8,
    n_kv_heads=4,
    n_experts=4,
    top_k=2,
    n_recursions=8,
    max_seq_len=512,
)
config.save("configs/base.json")
```

Pre-built configs live in [`configs/`](configs/):

| Config | Description |
|---|---|
| `tiny.json` | Ultra-light (~7K params), runs on any machine |
| `small.json` | Lightweight (~176K params), fast training |
| `base.json` | Default (~2.7M params), best quality |

---

## 📈 Training Curves

Trained on the GSM8K corpus (~4M characters of real math reasoning, augmented
with synthetic arithmetic problems up to 100M characters):

![Training curves](assets/training_curves.png)

---

## 📁 Project Structure

```
rdru/
├── configs/
│   ├── tiny.json
│   ├── small.json
│   └── base.json
├── rdru/
│   ├── __init__.py
│   ├── config.py
│   ├── model.py
│   ├── layers.py
│   └── tokenizer.py
├── scripts/
│   ├── train.py
│   ├── generate.py
│   └── evaluate.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 📚 References

- Su et al., 2021 — [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- Ainslie et al., 2023 — [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- Shazeer et al., 2017 — [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- Shazeer, 2020 — [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- Chowdhery et al., 2022 — [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) (QK-Norm)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
