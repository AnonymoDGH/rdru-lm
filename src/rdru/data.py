"""Data loading and corpus construction utilities."""

from __future__ import annotations

import logging
import random
from typing import List, Optional

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CharDataset(Dataset):
    """Character-level language modelling dataset.

    Maps a raw text corpus to fixed-length sequences for next-token prediction.
    The vocabulary is built from the set of characters appearing in the text.

    Each sample is a tuple ``(input_ids, target_ids)`` where ``target_ids``
    is ``input_ids`` shifted right by one position.
    """

    def __init__(self, text: str, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

        chars = sorted(set(text))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for c, i in self.stoi.items()}
        self.vocab_size: int = len(chars)

        data = [self.stoi[c] for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.n_chunks = (len(self.data) - 1) // seq_len

        logger.info(
            "vocab=%d chars, tokens=%d, chunks=%d",
            self.vocab_size,
            len(self.data),
            self.n_chunks,
        )

    def __len__(self) -> int:
        return self.n_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start + 1 : start + self.seq_len + 1]
        return x, y


def build_gsm8k_corpus(target_chars: int = 50_000_000) -> str:
    """Build a text corpus from the GSM8K math reasoning dataset.

    Each sample is formatted as::

        q: <question>
        a: <chain-of-thought answer>

    Args:
        target_chars: Maximum number of characters to include.

    Returns:
        Full corpus as a single string.
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    samples: List[str] = []
    total = 0

    for example in ds:
        text = f"q: {example['question'].lower()}\na: {example['answer'].lower()}"
        samples.append(text)
        total += len(text) + 1
        if total >= target_chars:
            break

    corpus = "\n".join(samples)
    logger.info("gsm8k: %d samples, %d chars", len(samples), len(corpus))
    return corpus


def build_large_corpus(target_chars: int = 1_000_000_000) -> str:
    """Build a large corpus from GSM8K and synthetic arithmetic problems.

    After exhausting GSM8K (train + test splits), the remaining capacity is
    filled with procedurally-generated arithmetic word problems. This is
    intended for fast bootstrapping; production use should replace synthetic
    data with diverse sources (MetaMathQA, NuminaMath, code corpora, etc.).

    Args:
        target_chars: Target corpus size in characters.

    Returns:
        Full corpus as a single string.
    """
    from datasets import load_dataset

    blocks: List[str] = []
    total = 0

    for split in ("train", "test"):
        for example in load_dataset("openai/gsm8k", "main", split=split, streaming=True):
            text = f"q: {example['question'].lower()}\na: {example['answer'].lower()}"
            blocks.append(text)
            total += len(text) + 1

    gsm_block = "\n".join(blocks) + "\n"
    logger.info("gsm8k: %d chars", len(gsm_block))

    operators = ("+", "-", "*")
    op_names = {"+": "plus", "-": "minus", "*": "times"}
    batch_num = 0

    while total < target_chars:
        if total + len(gsm_block) <= target_chars:
            blocks.append(gsm_block)
            total += len(gsm_block)
        else:
            blocks.append(gsm_block[: target_chars - total])
            total = target_chars
            break

        batch = []
        for _ in range(100_000):
            a = random.randint(1, 99)
            b = random.randint(1, 99)
            op = random.choice(operators)
            result = {"+": a + b, "-": a - b, "*": a * b}[op]
            batch.append(
                f"q: what is {a} {op} {b}\n"
                f"a: {a} {op_names[op]} {b} is {result}. #### {result}"
            )

        block = "\n".join(batch) + "\n"
        remaining = target_chars - total
        blocks.append(block[:remaining])
        total += min(len(block), remaining)
        batch_num += 1

        if batch_num % 5 == 0:
            logger.info("synthetic batch %d: %d / %d chars", batch_num, total, target_chars)

    return "".join(blocks)


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    """Convert a token ID tensor back to a string."""
    return "".join(itos[i.item()] for i in ids)
