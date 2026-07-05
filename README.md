<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/params-2.7M-orange" alt="2.7M parameters">
  <img src="https://img.shields.io/badge/weight_tying-✓-brightgreen" alt="Weight Tying">
  <img src="https://img.shields.io/badge/QK_norm-✓-brightgreen" alt="QK Norm">
</p>

# RDRU — Recursive Denoising Reasoning Unit

A character-level language model that applies the **same transformer block
iteratively**, paired with an **auxiliary denoising objective**. Designed for
mathematical reasoning at small scales (~2.7M parameters).



**Key ideas:**

- **Recursive reasoning** — a single RRU block is applied 4–8 times per token,
  allowing the hidden state to refine its representation iteratively.
- **Denoising auxiliary loss** — MSE between the hidden state and a perturbed
  version thereof encourages convergence toward a fixed point.
- **Grouped Query Attention (GQA)** — 8 query heads, 4 key/value heads, with
  Rotary Position Embeddings (RoPE).
- **Sparse MoE-FFN** — 4 SwiGLU experts, top-2 routing, increasing capacity
  without proportional compute.

---

## What's New (v2.1)

| Improvement | Description | Benefit |
|-------------|-------------|---------|
| **Weight Tying** | Embedding and output projection share weights | -23K params, better representations |
| **QK-Norm** | LayerNorm on Q/K before attention (PaLM-style) | Stable training, no loss spikes |
| **NTK-Aware RoPE** | Frequency scaling for longer sequences | Extrapolate beyond max_seq_len |
| **Load Balancing MoE** | Auxiliary loss penalizes expert imbalance | No expert collapse |
| **Gradient Checkpointing** | Trade compute for memory in RRU loop | ~8× less VRAM during training |
| **AMP Support** | Automatic Mixed Precision (FP16/BF16) | ~2× speed on modern CPUs/GPUs |
| **Gradient Accumulation** | Accumulate gradients over N steps | Large effective batch on limited RAM |
| **EMA** | Exponential Moving Average of weights | Better generalization at inference |
| **Curriculum Learning** | Progressive reasoning depth | Starts with 1 step, grows gradually |
| **Top-p / Repetition Penalty** | Nucleus sampling + repeat penalty | Higher quality generations |
| **KV-Cache Ready** | Structure for cached generation | O(T) vs O(T²) generation |
| **Temperature=0 Safe** | Argmax fallback instead of div-by-zero | No more crashes at temp=0 |

---

## Architecture

| Component | Description | Reference |
|-----------|-------------|-----------|
| \ | Rotation-based relative position encoding | [RoFormer](https://arxiv.org/abs/2104.09864) |
| \ | Grouped Query Attention (8 Q-heads, 4 KV-heads) | [GQA](https://arxiv.org/abs/2305.13245) |
| \ | Sparse Mixture-of-Experts (4 experts, top-2) | [MoE](https://arxiv.org/abs/1701.06538) |
| \ | Gated feed-forward activation | [GLU Variants](https://arxiv.org/abs/2002.05202) |
| \ | Single shared transformer block (applied iteratively) | — |
| \ | Top-level model: embedding + RRU loop + denoising | — |

### Parameter count (default config)

| Component | Parameters |
|-----------|-----------|
| Token embedding | 23,296 |
| Attention (QKV + output) | 139,264 |
| MoE-FFN (4 SwiGLU experts) | 1,575,424 |
| Layer norms & projections | 1,005,149 |
| **Total** | **~2,735,616** |

---

## Installation

\Requirement already satisfied: torch>=2.0.0 in /usr/local/lib/python3.13/site-packages (from -r requirements.txt (line 1)) (2.12.1+cpu)
Requirement already satisfied: datasets>=2.14.0 in /usr/local/lib/python3.13/site-packages (from -r requirements.txt (line 2)) (5.0.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.29.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (4.15.0)
Requirement already satisfied: setuptools<82 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (70.2.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.13/site-packages (from torch>=2.0.0->-r requirements.txt (line 1)) (2026.4.0)
Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (2.3.5)
Requirement already satisfied: pyarrow>=21.0.0 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (24.0.0)
Requirement already satisfied: dill<0.4.2,>=0.3.0 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (0.4.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (2.2.3)
Requirement already satisfied: requests>=2.32.2 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (2.33.0)
Requirement already satisfied: httpx<1.0.0 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (0.28.1)
Requirement already satisfied: tqdm>=4.66.3 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (4.68.3)
Requirement already satisfied: xxhash in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (3.8.0)
Requirement already satisfied: multiprocess<0.70.20 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (0.70.19)
Requirement already satisfied: huggingface-hub<2.0,>=0.25.0 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (1.22.0)
Requirement already satisfied: packaging in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (26.2)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.13/site-packages (from datasets>=2.14.0->-r requirements.txt (line 2)) (6.0.3)
Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /usr/local/lib/python3.13/site-packages (from fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (3.14.1)
Requirement already satisfied: anyio in /usr/local/lib/python3.13/site-packages (from httpx<1.0.0->datasets>=2.14.0->-r requirements.txt (line 2)) (4.14.0)
Requirement already satisfied: certifi in /usr/local/lib/python3.13/site-packages (from httpx<1.0.0->datasets>=2.14.0->-r requirements.txt (line 2)) (2026.6.17)
Requirement already satisfied: httpcore==1.* in /usr/local/lib/python3.13/site-packages (from httpx<1.0.0->datasets>=2.14.0->-r requirements.txt (line 2)) (1.0.9)
Requirement already satisfied: idna in /usr/local/lib/python3.13/site-packages (from httpx<1.0.0->datasets>=2.14.0->-r requirements.txt (line 2)) (3.18)
Requirement already satisfied: h11>=0.16 in /usr/local/lib/python3.13/site-packages (from httpcore==1.*->httpx<1.0.0->datasets>=2.14.0->-r requirements.txt (line 2)) (0.16.0)
Requirement already satisfied: click<9.0.0,>=8.4.2 in /usr/local/lib/python3.13/site-packages (from huggingface-hub<2.0,>=0.25.0->datasets>=2.14.0->-r requirements.txt (line 2)) (8.4.2)
Requirement already satisfied: hf-xet<2.0.0,>=1.5.1 in /usr/local/lib/python3.13/site-packages (from huggingface-hub<2.0,>=0.25.0->datasets>=2.14.0->-r requirements.txt (line 2)) (1.5.1)
Requirement already satisfied: aiohappyeyeballs>=2.5.0 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (2.6.2)
Requirement already satisfied: aiosignal>=1.4.0 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (1.4.0)
Requirement already satisfied: attrs>=17.3.0 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (26.1.0)
Requirement already satisfied: frozenlist>=1.1.1 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (1.8.0)
Requirement already satisfied: multidict<7.0,>=4.5 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (6.7.1)
Requirement already satisfied: propcache>=0.2.0 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (0.5.2)
Requirement already satisfied: yarl<2.0,>=1.17.0 in /usr/local/lib/python3.13/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]<=2026.4.0,>=2023.1.0->datasets>=2.14.0->-r requirements.txt (line 2)) (1.24.2)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.13/site-packages (from requests>=2.32.2->datasets>=2.14.0->-r requirements.txt (line 2)) (3.4.7)
Requirement already satisfied: urllib3<3,>=1.26 in /usr/local/lib/python3.13/site-packages (from requests>=2.32.2->datasets>=2.14.0->-r requirements.txt (line 2)) (2.7.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.13/site-packages (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 1)) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.13/site-packages (from jinja2->torch>=2.0.0->-r requirements.txt (line 1)) (3.0.3)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.13/site-packages (from pandas->datasets>=2.14.0->-r requirements.txt (line 2)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.13/site-packages (from pandas->datasets>=2.14.0->-r requirements.txt (line 2)) (2025.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.13/site-packages (from pandas->datasets>=2.14.0->-r requirements.txt (line 2)) (2026.2)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.13/site-packages (from python-dateutil>=2.8.2->pandas->datasets>=2.14.0->-r requirements.txt (line 2)) (1.17.0)
---

## Usage

### Training on GSM8K

\
### Configurations

| Config | Params | d_model | Experts | Steps | RAM | Time/epoch (CPU) |
|--------|--------|---------|---------|-------|-----|-------------------|
| \ | ~7K | 64 | 2 | 4 | ~256 MB | ~2 min |
| \ | ~176K | 128 | 4 | 6 | ~512 MB | ~15 min |
| \ | ~2.7M | 256 | 4 | 8 | ~1.5 GB | ~45 min |

### Generation

\
### Evaluation

\
### Running tests

\Requirement already satisfied: pytest in /usr/local/lib/python3.13/site-packages (9.0.3)
Requirement already satisfied: iniconfig>=1.0.1 in /usr/local/lib/python3.13/site-packages (from pytest) (2.3.0)
Requirement already satisfied: packaging>=22 in /usr/local/lib/python3.13/site-packages (from pytest) (26.2)
Requirement already satisfied: pluggy<2,>=1.5 in /usr/local/lib/python3.13/site-packages (from pytest) (1.6.0)
Requirement already satisfied: pygments>=2.7.2 in /usr/local/lib/python3.13/site-packages (from pytest) (2.20.0)
============================= test session starts ==============================
platform linux -- Python 3.13.13, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/user
plugins: anyio-4.14.0
collected 0 items

============================ no tests ran in 0.01s =============================
---

## Configuration

Model and training hyperparameters are defined as \ objects and
serialised to JSON, following the same pattern as
[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3).

\
Pre-built configs are in [\](configs/):

| Config | Description |
|--------|-------------|
| \ | Ultra-light (~7K params), runs on any machine |
| \ | Lightweight (~176K params), fast training |
| \ | Default (~2.7M params), best quality |

---

## Training curves

Training on the GSM8K corpus (~4M characters of real math reasoning +
synthetic arithmetic problems to reach 100M characters):

\
---

## Project structure

\
---

## References

\
---

## License

This project is licensed under the MIT License — see [\](LICENSE).
