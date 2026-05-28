#!/usr/bin/env python3
"""Generate text from a trained RDRUv2 checkpoint.

Usage:
    python generate.py rdru_checkpoint.pth --prompt "what is 2 + 2"
    python generate.py rdru_checkpoint.pth --prompt "q: 2+2?\na:" --max_new 100 --temperature 0.4
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from src.rdru import ModelConfig, RDRUv2
from src.rdru.data import decode


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with RDRUv2")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--prompt", type=str, default="what is the sum of 5 and 3")
    parser.add_argument("--max_new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
        legacy_keys = {"d_model": 256, "n_query_heads": 8, "n_kv_heads": 4,
                       "n_reasoning_steps": 8, "n_experts": 4, "vocab_size": 91}
        model_config = ModelConfig(**{k: checkpoint.get(k, v) for k, v in legacy_keys.items()})

    model = RDRUv2(model_config).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load vocab
    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})
    if not itos and "itos" not in checkpoint:
        logging.warning("no vocabulary found in checkpoint, using default char mapping")
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .:?$/-+*#'\n,;!%()"
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}

    # Tokenize prompt
    prompt = args.prompt.lower()
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)

    # Generate
    output_ids = model.generate(ids, args.max_new, args.temperature, args.top_k)
    generated = decode(output_ids[0], itos)

    print(f"Prompt: {prompt}")
    print(f"Output:{generated[len(prompt):]}")


if __name__ == "__main__":
    main()
