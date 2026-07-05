#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         RDRU-Nyx — Text Generation                           ║
║  Advanced sampling with KV-cache and uncertainty estimation ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python generate.py nyx_runs/best_model.pth --prompt "what is 2 + 2"
    python generate.py checkpoint.pth --prompt "q: natalia sold clips\\na:" --temp 0.4 --max_new 200
    python generate.py checkpoint.pth --interactive  # chat mode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from src.rdru import ModelConfig, RDRUNyx
from src.rdru.data import decode


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with RDRU-Nyx")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new", type=int, default=200)
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--show_uncertainty", action="store_true", help="Show confidence scores")
    parser.add_argument("--vocab", type=str, default=None, help="Path to vocab.json")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str) -> tuple:
    """Load model, config, and vocabulary from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = RDRUNyx(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logging.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load vocabulary
    stoi, itos = checkpoint.get("stoi", {}), checkpoint.get("itos", {})
    if not itos:
        # Default character mapping
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .:?$/-+*#'\n,;!%()"
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}

    return model, model_config, stoi, itos


def generate(model, prompt: str, stoi, itos, args) -> str:
    """Generate text from a prompt."""
    prompt = prompt.lower()
    ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=args.device)

    output_ids = model.generate(
        ids,
        max_new_tokens=args.max_new,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_kv_cache=True,
    )

    generated = decode(output_ids[0], itos)
    return generated


def interactive_mode(model, stoi, itos, args) -> None:
    """Interactive chat-like generation."""
    print("\n" + "=" * 60)
    print("  RDRU-Nyx Interactive Mode (Ctrl+C to exit)")
    print("=" * 60)
    print()

    history = ""
    try:
        while True:
            user_input = input(">> ").strip()
            if not user_input:
                continue

            prompt = f"q: {user_input.lower()}\na:"
            full = generate(model, prompt, stoi, itos, args)
            answer = full[len(prompt):].strip()
            print(f"   {answer}\n")

            history += prompt + answer + "\n"
    except KeyboardInterrupt:
        print("\nBye!")


def main() -> None:
    setup_logging()
    args = parse_args()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab if provided separately
    if args.vocab:
        vocab = json.load(open(args.vocab))
        stoi = vocab["stoi"]
        itos = {int(k): v for k, v in vocab["itos"].items()}

    model, model_config, stoi, itos = load_model(args.checkpoint, args.device)

    if args.interactive:
        interactive_mode(model, stoi, itos, args)
        return

    if args.prompt:
        result = generate(model, args.prompt, stoi, itos, args)
        print(f"Prompt: {args.prompt}")
        print(f"Output: {result[len(args.prompt):]}")

        if args.show_uncertainty:
            # Show uncertainty per token
            ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt.lower()]], device=args.device)
            _, uncertainty = model(ids, return_uncertainty=True)
            print(f"\nAvg uncertainty: {uncertainty.mean().item():.4f}")
    else:
        print("No prompt provided. Use --prompt or --interactive")


if __name__ == "__main__":
    main()
