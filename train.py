#!/usr/bin/env python3
"""Entry point for training RDRUv2.

Usage:
    python train.py
    python train.py configs/gsm8k.json
    python train.py configs/gsm8k.json --batch_size=32 --n_epochs=5
"""

from __future__ import annotations

import argparse
import logging
import sys

from torch.utils.data import DataLoader

from src.rdru import Trainer, ModelConfig, TrainingConfig, CharDataset, build_gsm8k_corpus


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RDRUv2 language model")
    parser.add_argument(
        "config",
        nargs="?",
        default=None,
        help="Path to a JSON training config (optional)",
    )
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_query_heads", type=int, default=None)
    parser.add_argument("--n_kv_heads", type=int, default=None)
    parser.add_argument("--n_reasoning_steps", type=int, default=None)
    parser.add_argument("--n_experts", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--n_epochs", type=int, default=None)
    parser.add_argument("--target_chars", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()

    # Load configs (with JSON file as base, then CLI overrides)
    model_config = ModelConfig()
    training_config = TrainingConfig()

    if args.config:
        model_config = ModelConfig.load(args.config)
        logging.info("loaded model config from %s", args.config)

    # CLI overrides for model config
    for key in ("d_model", "n_query_heads", "n_kv_heads", "n_reasoning_steps", "n_experts"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(model_config, key, value)

    # CLI overrides for training config
    mapping = {
        "batch_size": "batch_size",
        "seq_len": "seq_len",
        "learning_rate": "learning_rate",
        "n_epochs": "n_epochs",
        "target_chars": "target_chars",
        "checkpoint": "checkpoint_path",
        "device": "device",
    }
    for arg_key, config_key in mapping.items():
        value = getattr(args, arg_key, None)
        if value is not None:
            setattr(training_config, config_key, value)

    # Build dataset
    logging.info("building corpus (target=%d chars)...", training_config.target_chars)
    corpus = build_gsm8k_corpus(target_chars=training_config.target_chars)
    dataset = CharDataset(corpus, training_config.seq_len)
    model_config.vocab_size = dataset.vocab_size

    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Train
    trainer = Trainer(model_config, training_config)
    trainer.train(dataloader)
    logging.info("training complete")


if __name__ == "__main__":
    main()
