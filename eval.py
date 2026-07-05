#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         RDRU-Nyx — Evaluation on GSM8K                      ║
║  Measures exact-match accuracy on math word problems        ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python eval.py nyx_runs/best_model.pth --n_samples=100
    python eval.py checkpoint.pth --n_samples=50 --temp=0.2
"""

from __future__ import annotations

import argparse
import logging
import re
import sys

import torch

from src.rdru import ModelConfig, RDRUNyx
from src.rdru.data import decode

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def extract_answer(text: str) -> str:
    """Extract the final numeric answer from GSM8K format."""
    # Look for #### answer pattern
    match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    # Look for "is X" pattern at end
    match = re.search(r"is\s*(-?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    return ""


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = RDRUNyx(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    stoi = checkpoint.get("stoi", {})
    itos = checkpoint.get("itos", {})
    if not itos:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 .:?$/-+*#'\n,;!%()"
        stoi = {c: i for i, c in enumerate(chars)}
        itos = {i: c for c, i in stoi.items()}
    return model, model_config, stoi, itos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RDRU-Nyx on GSM8K")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of test examples")
    parser.add_argument("--max_new", type=int, default=256, help="Max generation tokens")
    parser.add_argument("--temp", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, model_config, stoi, itos = load_model(args.checkpoint, device)
    logger.info(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # Load GSM8K test set
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    except ImportError:
        logger.error("datasets not installed. pip install datasets")
        return

    correct = 0
    total = 0
    results = []

    for i, example in enumerate(ds):
        if i >= args.n_samples:
            break

        question = example["question"].lower()
        true_answer = extract_answer(example["answer"].lower())

        prompt = f"q: {question}\na:"
        ids = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)

        output_ids = model.generate(
            ids, max_new_tokens=args.max_new, temperature=args.temp,
            top_k=40, top_p=None, repetition_penalty=1.0,
        )
        generated = decode(output_ids[0], itos)
        model_answer = extract_answer(generated[len(prompt):])

        is_correct = (model_answer == true_answer)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question[:80],
            "true": true_answer,
            "model": model_answer,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            logger.info(f"[{i+1}/{args.n_samples}] Acc: {correct}/{total} = {correct/total*100:.1f}%")

    accuracy = correct / max(total, 1) * 100
    logger.info(f"\n{'='*50}")
    logger.info(f"GSM8K Evaluation: {correct}/{total} = {accuracy:.1f}%")
    logger.info(f"{'='*50}")

    # Show some results
    print("\nSample results:")
    for r in results[:5]:
        mark = "✅" if r["correct"] else "❌"
        print(f"  {mark} Q: {r['question'][:60]}...")
        print(f"     True: {r['true']:>6s}  |  Model: {r['model']:>6s}")


if __name__ == "__main__":
    main()
