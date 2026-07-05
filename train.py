#!/usr/bin/env python3
"""RDRU-Nyx Training Entry Point — optimized for CPU/low-resource."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from torch.utils.data import DataLoader
from src.rdru import NyxTrainer, ModelConfig, TrainingConfig, CharDataset, build_gsm8k_corpus

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)

def parse_args():
    p = argparse.ArgumentParser(description="Train RDRU-Nyx")
    p.add_argument("config", nargs="?", default=None)
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_query_heads", type=int, default=None)
    p.add_argument("--n_kv_heads", type=int, default=None)
    p.add_argument("--n_reasoning_steps", type=int, default=None)
    p.add_argument("--n_experts", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=None)
    p.add_argument("--seq_len", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--n_epochs", type=int, default=None)
    p.add_argument("--target_chars", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--run_dir", type=str, default="nyx_runs")
    return p.parse_args()

def main():
    setup_logging(); args = parse_args()
    mc = ModelConfig.load(args.config) if args.config else ModelConfig()
    for k in ("d_model","n_query_heads","n_kv_heads","n_reasoning_steps","n_experts"):
        v = getattr(args,k,None)
        if v is not None: setattr(mc,k,v)
    tc = TrainingConfig()
    mapping={"batch_size":"batch_size","gradient_accumulation_steps":"gradient_accumulation_steps",
             "seq_len":"seq_len","learning_rate":"learning_rate","n_epochs":"n_epochs",
             "target_chars":"target_chars","warmup_steps":"warmup_steps","checkpoint":"checkpoint_path","device":"device"}
    for a,c in mapping.items():
        v = getattr(args,a,None)
        if v is not None: setattr(tc,c,v)
    if args.no_amp: tc.use_amp = False
    logging.info(f"Building corpus ({tc.target_chars:,} chars)...")
    corpus = build_gsm8k_corpus(target_chars=tc.target_chars)
    ds = CharDataset(corpus, tc.seq_len); mc.vocab_size = ds.vocab_size
    Path(args.run_dir).mkdir(parents=True, exist_ok=True); json.dump({"stoi":ds.stoi,"itos":{str(k):v for k,v in ds.itos.items()}}, open(Path(args.run_dir)/"vocab.json","w"))
    dl = DataLoader(ds, batch_size=tc.batch_size, shuffle=True, num_workers=0, drop_last=True)
    logging.info(f"Model: d_model={mc.d_model} heads={mc.n_query_heads}/{mc.n_kv_heads} experts={mc.n_experts} steps={mc.n_reasoning_steps} wt={mc.use_weight_tying} qkn={mc.use_qk_norm}")
    tr = NyxTrainer(mc, tc, run_dir=args.run_dir)
    metrics = tr.train(dl)
    logging.info(f"Done! Best loss: {metrics['best_loss']:.4f}")
if __name__ == "__main__": main()
