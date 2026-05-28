<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/params-2.7M-orange" alt="2.7M parameters">
</p>

# RDRU — Recursive Denoising Reasoning Unit

A character-level language model that applies the **same transformer block
iteratively**, paired with an **auxiliary denoising objective**. Designed for
mathematical reasoning at small scales (~2.7M parameters).

```
Embedding → Projection → [RRU × steps] ⋯ → Output
                            │
                   ┌────────┴────────┐
               GQA (RoPE)      MoE-FFN (top-2/4)
```

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

## Architecture

| Component | Description | Reference |
|-----------|-------------|-----------|
| `RotaryEmbedding` | Rotation-based relative position encoding | [RoFormer](https://arxiv.org/abs/2104.09864) |
| `GQA` | Grouped Query Attention (8 Q-heads, 4 KV-heads) | [GQA](https://arxiv.org/abs/2305.13245) |
| `MoEFFN` | Sparse Mixture-of-Experts (4 experts, top-2) | [MoE](https://arxiv.org/abs/1701.06538) |
| `SwiGLU` | Gated feed-forward activation | [GLU Variants](https://arxiv.org/abs/2002.05202) |
| `RRU` | Single shared transformer block | — |
| `RDRUv2` | Top-level model: embedding + RRU loop + denoising | — |

### Parameter count

| Component | Parameters |
|-----------|-----------|
| Token embedding | 23,296 |
| Attention (QKV + output) | 139,264 |
| MoE-FFN (4 SwiGLU experts) | 1,575,424 |
| Layer norms & projections | 1,005,149 |
| **Total** | **~2,743,133** |

---

## Installation

```bash
git clone https://github.com/USUARIO/rdru-lm
cd rdru-lm
pip install -r requirements.txt
```

---

## Usage

### Training on GSM8K

```bash
python train.py
```

```bash
# With a JSON config file:
python train.py configs/gsm8k.json

# Override any parameter from the CLI:
python train.py configs/gsm8k.json --batch_size=32 --n_epochs=5 --target_chars=200000000
```

### Generation

```bash
python generate.py rdru_checkpoint.pth --prompt "what is 2 + 2"
```

```bash
python generate.py rdru_checkpoint.pth \
    --prompt "q: natalia sold clips to 48 of her friends\na:" \
    --temperature 0.4 \
    --max_new 200
```

### Running tests

```bash
pip install pytest
pytest tests/
```

---

## Configuration

Model and training hyperparameters are defined as `@dataclass` objects and
serialised to JSON, following the same pattern as
[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3).

```json
{
  "d_model": 256,
  "n_query_heads": 8,
  "n_kv_heads": 4,
  "n_reasoning_steps": 8,
  "n_experts": 4
}
```

Pre-built configs are in [`configs/`](configs/):

| Config | Description |
|--------|-------------|
| `default.json` | Default model architecture |
| `gsm8k.json` | Training setup for GSM8K (100M chars, 3 epochs) |

---

## Training curves

Training on the GSM8K corpus (~4M characters of real math reasoning +
synthetic arithmetic problems to reach 100M characters):

```
epoch 1  loss: 5.37 → 1.42
epoch 2  loss: 1.21
epoch 3  loss: 1.00
epoch 4  loss: 0.87
epoch 5  loss: 0.78
```

---

## Project structure

```
rdru-lm/
├── src/
│   └── rdru/
│       ├── __init__.py     # Public API
│       ├── config.py       # ModelConfig & TrainingConfig dataclasses
│       ├── model.py        # RDRUv2, RRU, GQA, MoE, RoPE
│       ├── data.py         # CharDataset, corpus builders
│       └── trainer.py      # Training loop
├── tests/
│   ├── test_model.py
│   ├── test_data.py
│   └── test_config.py
├── configs/
│   ├── default.json
│   └── gsm8k.json
├── .github/ISSUE_TEMPLATE/
│   ├── bug_report.md
│   └── feature_request.md
├── train.py                # Entry point
├── generate.py             # Entry point
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## References

```bibtex
@inproceedings{su2024roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Ahmed and Liu, Yunfeng and others},
  booktitle={Neurocomputing},
  year={2024}
}

@article{ainslie2023gqa,
  title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
  author={Ainslie, Joshua and Lee-Thorp, James and de Jong, Michiel and Zemlyanskiy, Yury and Lebrón, Federico and Sanghai, Sumit},
  journal={arXiv preprint arXiv:2305.13245},
  year={2023}
}

@article{shazeer2020glu,
  title={GLU Variants Improve Transformer},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:2002.05202},
  year={2020}
}

@article{shazeer2017moe,
  title={Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  journal={arXiv preprint arXiv:1701.06538},
  year={2017}
}

@article{deepseek2025r1,
  title={DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning},
  author={DeepSeek-AI},
  journal={arXiv preprint arXiv:2501.12948},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License — see [`LICENSE`](LICENSE).
