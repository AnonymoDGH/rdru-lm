"""NyxTrainer — Advanced training loop with AMP, gradient accumulation, EMA, curriculum."""
from __future__ import annotations
import json, logging, math, time
from pathlib import Path
from typing import Optional
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from .config import ModelConfig, TrainingConfig
from .model import RDRUNyx, EMA

logger = logging.getLogger(__name__)

class NyxTrainer:
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig, run_dir="nyx_runs"):
        self.model_config = model_config; self.training_config = training_config
        self.run_dir = Path(run_dir); self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = training_config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device: {self.device}")
        self.model = RDRUNyx(model_config).to(self.device)
        self.n_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parameters: {self.n_params:,}")
        no_decay = ["bias", "layernorm", "norm"]
        grps = [
            {"params": [p for n,p in self.model.named_parameters() if not any(nd in n.lower() for nd in no_decay)], "weight_decay": training_config.weight_decay},
            {"params": [p for n,p in self.model.named_parameters() if any(nd in n.lower() for nd in no_decay)], "weight_decay": 0.0},
        ]
        self.optimizer = optim.AdamW(grps, lr=training_config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
        total_steps = training_config.n_epochs * 100
        ws = training_config.warmup_steps
        def lr_lambda(s): return s/max(1,ws) if s<ws else 0.5*(1+math.cos(math.pi*(s-ws)/max(1,total_steps-ws)))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.criterion = nn.CrossEntropyLoss()
        self.use_amp = training_config.use_amp and self.device in ("cuda", "cpu") and self._supports_bf16()
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        logger.info(f"AMP: {'enabled' if self.use_amp else 'disabled'}")
        self.ema = EMA(self.model, training_config.ema_decay) if training_config.ema_decay > 0 else None
        self.grad_accum_steps = training_config.gradient_accumulation_steps
        self.best_loss = float("inf"); self.metrics_history = []
    def _supports_bf16(self):
        try: return torch.ones(1, dtype=torch.bfloat16).device.type == "cpu"
        except: return False
    def train(self, dataloader):
        cfg = self.training_config
        for epoch in range(1, cfg.n_epochs + 1):
            cs = min(cfg.curriculum_steps_start + (epoch-1)*cfg.curriculum_steps_increment, self.model_config.n_reasoning_steps)
            logger.info(f"Epoch {epoch}/{cfg.n_epochs} — curriculum_steps={cs}")
            el = self._run_epoch(dataloader, epoch, cs)
            self.scheduler.step()
            if self.ema: self.ema.update(self.model)
            if el < self.best_loss: self.best_loss = el; self._save_checkpoint(epoch, is_best=True)
            self._save_checkpoint(epoch)
            logger.info(f"Epoch {epoch} avg_loss: {el:.4f} (best: {self.best_loss:.4f})")
            self.metrics_history.append({"epoch": epoch, "loss": el})
        json.dump(self.metrics_history, open(self.run_dir/"metrics.json", "w"), indent=2)
        if self.ema: self.ema.apply_shadow(self.model); self._save_checkpoint(cfg.n_epochs, suffix="ema"); self.ema.restore(self.model)
        return {"best_loss": self.best_loss, "final_loss": el}
    def _run_epoch(self, dataloader, epoch, curriculum_step):
        self.model.train(); total_loss=0.0; total_ce=0.0; total_dl=0.0; total_ml=0.0
        n_batches=0; acc=0; st=time.time()
        self.optimizer.zero_grad()
        for bidx, (x, y) in enumerate(dataloader):
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                logits, dl, ml = self.model(x, return_denoising_loss=True, curriculum_step=curriculum_step)
                ce = self.criterion(logits.view(-1, self.model_config.vocab_size), y.view(-1))
                loss = (ce + self.training_config.denoising_loss_weight * dl + self.model_config.load_balancing_coef * ml) / self.grad_accum_steps
            if self.use_amp: self.scaler.scale(loss).backward()
            else: loss.backward()
            acc += 1
            if acc >= self.grad_accum_steps:
                if self.use_amp: self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
                if self.use_amp: self.scaler.step(self.optimizer); self.scaler.update()
                else: self.optimizer.step()
                self.optimizer.zero_grad(); acc = 0
            total_loss += loss.item() * self.grad_accum_steps; total_ce += ce.item()
            total_dl += dl.item(); total_ml += ml.item(); n_batches += 1
            if bidx % self.training_config.log_interval == 0:
                logger.info(f"epoch {epoch:2d} batch {bidx:5d}/{len(dataloader):5d} loss={loss.item()*self.grad_accum_steps:.4f} ce={ce.item():.4f} dl={dl.item():.6f} ml={ml.item():.6f} lr={self.optimizer.param_groups[0]['lr']:.6f}")
        if acc > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
            if self.use_amp: self.scaler.step(self.optimizer); self.scaler.update()
            else: self.optimizer.step()
            self.optimizer.zero_grad()
        return total_loss / max(n_batches, 1)
    def _save_checkpoint(self, epoch, is_best=False, is_swa=False, suffix=""):
        name = "best_model" if is_best else (f"swa_epoch_{epoch}" if is_swa else (f"checkpoint_{suffix}" if suffix else f"checkpoint_epoch_{epoch}"))
        path = self.run_dir / f"{name}.pth"
        torch.save({"model_state_dict": self.model.state_dict(), "model_config": self.model_config,
                     "training_config": self.training_config, "optimizer_state_dict": self.optimizer.state_dict(),
                     "scheduler_state_dict": self.scheduler.state_dict(), "epoch": epoch, "best_loss": self.best_loss}, path)
        logger.info(f"Checkpoint: {path}")
    @classmethod
    def from_checkpoint(cls, path, training_config=None):
        ckpt = torch.load(path, map_location="cpu")
        mc = ckpt["model_config"]; tc = training_config or ckpt.get("training_config", TrainingConfig())
        self = cls(mc, tc, run_dir=str(Path(path).parent))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_loss = ckpt.get("best_loss", float("inf"))
        return self
