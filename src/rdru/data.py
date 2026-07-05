"""Data loading utilities for RDRU-Nyx."""
from __future__ import annotations
import logging
import random
from typing import Dict, List, Optional
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int, stoi: Optional[Dict[str, int]] = None):
        super().__init__()
        self.seq_len = seq_len
        if stoi is None:
            chars = sorted(set(text))
            self.stoi = {c: i for i, c in enumerate(chars)}
            self.itos = {i: c for c, i in self.stoi.items()}
            self.vocab_size = len(chars)
        else:
            self.stoi = stoi
            self.itos = {i: c for c, i in stoi.items()}
            self.vocab_size = len(stoi)
        data = [self.stoi.get(c, 0) for c in text]
        self.data = torch.tensor(data, dtype=torch.long)
        self.n_chunks = max(0, (len(self.data) - 1) // seq_len)
        logger.info(f"vocab={self.vocab_size} tokens={len(self.data):,} chunks={self.n_chunks:,}")
    def __len__(self): return self.n_chunks
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return (self.data[start: start + self.seq_len],
                self.data[start + 1: start + self.seq_len + 1])

def build_gsm8k_corpus(target_chars: int = 50_000_000) -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed"); return ""
    ds = load_dataset("openai/gsm8k", "main", split="train", streaming=True)
    samples, total = [], 0
    for ex in ds:
        text = f"q: {ex['question'].lower()}\na: {ex['answer'].lower()}"
        samples.append(text); total += len(text) + 1
        if total >= target_chars: break
    corpus = "\n".join(samples)
    logger.info(f"GSM8K: {len(samples)} samples, {len(corpus):,} chars")
    return corpus

def build_large_corpus(target_chars: int = 100_000_000) -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        return _generate_synthetic(target_chars)
    blocks, total = [], 0
    for split in ("train", "test"):
        try:
            for ex in load_dataset("openai/gsm8k", "main", split=split, streaming=True):
                text = f"q: {ex['question'].lower()}\na: {ex['answer'].lower()}"
                blocks.append(text); total += len(text) + 1
        except: pass
    if total < target_chars:
        blocks.append(_generate_synthetic(target_chars - total))
    corpus = "".join(blocks)
    return corpus[:target_chars]

def _generate_synthetic(target_chars: int) -> str:
    ops = ("+", "-", "*"); opn = {"+": "plus", "-": "minus", "*": "times"}
    samples, total = [], 0
    while total < target_chars:
        a, b = random.randint(1, 99), random.randint(1, 99)
        op = random.choice(ops)
        r = {"+": a + b, "-": a - b, "*": a * b}[op]
        t = [f"q: what is {a} {op} {b}", f"q: calculate {a} {op} {b}",
             f"q: solve: {a} {op} {b}", f"q: {a} {op} {b} eq"]
        text = f"{random.choice(t)}\na: {a} {opn[op]} {b} is {r}. #### {r}"
        samples.append(text); total += len(text) + 1
    logger.info(f"Synthetic: {len(samples):,} samples, {total:,} chars")
    return "\n".join(samples)

def decode(ids, itos, default="�"):
    return "".join(itos.get(i.item(), default) for i in ids)
