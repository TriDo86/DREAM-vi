"""
Trainer for the DREAM model.

Features
--------
- Clean train / eval loop separated into private methods.
- EarlyStopping with configurable patience and min_delta.
- Checkpoint save/resume: stores model + optimizer + epoch + best val loss.
- Rolling checkpoint pruning (keep only last K).
- WandB integration: un-comment the three wandb lines to activate.
- Structured logging via Python's standard `logging` module.

Usage
-----
    from dream.trainer import Trainer, TrainerConfig

    cfg = TrainerConfig(epochs=100, patience=15)
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.fit()

    # Resume from a checkpoint:
    trainer.resume("checkpoints/dream_epoch_0005.pt")
    trainer.fit()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .loss import LossComponents, compute_loss
from .model import DREAMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    # ── optimisation ─────────────────────────────────────────────
    epochs:        int   = 100
    learning_rate: float = 1e-4
    weight_decay:  float = 0.0    # paper uses plain Adam with no weight decay


    # ── reproducibility ──────────────────────────────────────────
    seed: int = 86

    # ── early stopping ───────────────────────────────────────────
    patience:   int   = 15
    min_delta:  float = 1e-4

    # ── checkpointing ────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints"
    save_every:     int = 5   # save a rolling checkpoint every N epochs
    keep_last_k:    int = 3   # keep only the K most recent rolling checkpoints

    # ── hardware ─────────────────────────────────────────────────
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Monitors a validation metric and signals when to stop.

    Returns True from `step()` when training should stop.
    """

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best: float = float("inf")
        self._counter: int = 0

    @property
    def best(self) -> float:
        return self._best

    def step(self, val_loss: float) -> bool:
        if val_loss < self._best - self.min_delta:
            self._best    = val_loss
            self._counter = 0
            return False
        self._counter += 1

        logger.debug(
            "EarlyStopping: no improvement for %d/%d epochs.",
            self._counter,
            self.patience,
        )
        return self._counter >= self.patience


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model:        DREAMModel,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        cfg:          TrainerConfig,
    ) -> None:
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.cfg          = cfg
        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        self.early_stopping = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

        self.ckpt_dir = Path(cfg.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._start_epoch:   int   = 1
        self._best_val_loss: float = float("inf")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run training from `_start_epoch` to `cfg.epochs`."""
        logger.info(
            "Training on %s | epochs=%d | lr=%g | patience=%d",
            self.device,
            self.cfg.epochs,
            self.cfg.learning_rate,
            self.cfg.patience,
        )

        for epoch in range(self._start_epoch, self.cfg.epochs + 1):
            # Regenerate random negatives before each epoch
            if hasattr(self.train_loader.dataset, "shuffle"):
                self.train_loader.dataset.shuffle()

            train_lc = self._train_epoch()
            val_lc   = self._val_epoch()

            is_best = val_lc.total.item() < self._best_val_loss
            self._log(epoch, train_lc, val_lc, is_best=is_best)

            # ── WandB (un-comment to activate) ──────────────────────────────
            import wandb
            wandb.log({"epoch": epoch,
                       **train_lc.as_log_dict("train/"),
                       **val_lc.as_log_dict("val/")})

            if is_best:
                self._best_val_loss = val_lc.total.item()
                self._save(epoch, tag="best")

            # Checkpoint: rolling save every N epochs
            if epoch % self.cfg.save_every == 0:
                self._save(epoch, tag=f"epoch_{epoch:04d}")
                self._prune_rolling()

            if self.early_stopping.step(val_lc.total.item()):
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    def resume(self, checkpoint_path: str | Path) -> None:
        """
        Load a checkpoint and resume training from the saved epoch.

        The checkpoint must have been created by this Trainer (i.e. contain
        the keys 'model_state', 'optimizer_state', 'epoch', 'best_val_loss').
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._start_epoch   = ckpt["epoch"] + 1
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))

        logger.info(
            "Resumed from '%s' (epoch=%d, best_val=%.4f)",
            checkpoint_path,
            ckpt["epoch"],
            self._best_val_loss,
        )

    # ------------------------------------------------------------------
    # Private: training / evaluation
    # ------------------------------------------------------------------

    def _train_epoch(self) -> LossComponents:
        self.model.train()
        accum = _Accumulator()

        for batch in self.train_loader:
            e_src, e_tgt, rand_src, rand_tgt, src_lid, tgt_lid = [
                t.to(self.device) for t in batch
            ]

            self.optimizer.zero_grad()

            em_src,  el_src,  logits_src  = self.model(e_src)
            em_tgt,  el_tgt,  logits_tgt  = self.model(e_tgt)
            em_rand_src, el_rand_src, _   = self.model(rand_src)
            em_rand_tgt, el_rand_tgt, _   = self.model(rand_tgt)

            lc = compute_loss(
                e_src, e_tgt,
                em_src, em_tgt, em_rand_src, em_rand_tgt,
                el_src, el_tgt, el_rand_src, el_rand_tgt,
                logits_src, logits_tgt, src_lid, tgt_lid,
            )

            lc.total.backward()
            self.optimizer.step()
            accum.update(lc)

        return accum.mean(len(self.train_loader))

    @torch.no_grad()
    def _val_epoch(self) -> LossComponents:
        self.model.eval()
        accum = _Accumulator()

        for batch in self.val_loader:
            e_src, e_tgt, rand_src, rand_tgt, src_lid, tgt_lid = [
                t.to(self.device) for t in batch
            ]

            em_src,  el_src,  logits_src  = self.model(e_src)
            em_tgt,  el_tgt,  logits_tgt  = self.model(e_tgt)
            em_rand_src, el_rand_src, _   = self.model(rand_src)
            em_rand_tgt, el_rand_tgt, _   = self.model(rand_tgt)

            lc = compute_loss(
                e_src, e_tgt,
                em_src, em_tgt, em_rand_src, em_rand_tgt,
                el_src, el_tgt, el_rand_src, el_rand_tgt,
                logits_src, logits_tgt, src_lid, tgt_lid,
            )
            accum.update(lc)

        return accum.mean(len(self.val_loader))

    # ------------------------------------------------------------------
    # Private: logging / checkpointing
    # ------------------------------------------------------------------

    def _log(
        self,
        epoch: int,
        train: LossComponents,
        val:   LossComponents,
        *,
        is_best: bool = False,
    ) -> None:
        
        logger.info(
            "Epoch %4d | train=%.4f (R=%.3f M=%.3f L=%.3f) | val=%.4f%s",
            epoch,
            train.total.item(),
            train.reconstruction.item(),
            train.meaning.item(),
            train.language.item(),
            val.total.item(),
            " ★" if is_best else "",
        )

    def _save(self, epoch: int, tag: str) -> Path:
        path = self.ckpt_dir / f"dream_{tag}.pt"
        torch.save(
            {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss":   self._best_val_loss,
                # store arch params so the checkpoint is self-contained
                "embedding_dim":   self.model.embedding_dim,
                "num_languages":   self.model.num_languages,
            },
            path,
        )

        logger.info("  Checkpoint saved → %s", path)
        return path

    def _prune_rolling(self) -> None:
        """Delete rolling epoch_* checkpoints beyond the last K."""
        epoch_ckpts = sorted(self.ckpt_dir.glob("dream_epoch_*.pt"))
        for old in epoch_ckpts[: -self.cfg.keep_last_k]:
            old.unlink(missing_ok=True)
            logger.debug("  Pruned old checkpoint: %s", old.name)


# ---------------------------------------------------------------------------
# Internal accumulator
# ---------------------------------------------------------------------------

class _Accumulator:
    """Accumulates LossComponents over batches, returns per-batch mean."""

    def __init__(self) -> None:
        self._total:          float = 0.0
        self._reconstruction: float = 0.0
        self._meaning:        float = 0.0
        self._language:       float = 0.0

    def update(self, lc: LossComponents) -> None:
        self._total          += lc.total.item()
        self._reconstruction += lc.reconstruction.item()
        self._meaning        += lc.meaning.item()
        self._language       += lc.language.item()

    def mean(self, n: int) -> LossComponents:
        def t(v: float) -> torch.Tensor:
            return torch.tensor(v / n)
        return LossComponents(
            total=t(self._total),
            reconstruction=t(self._reconstruction),
            meaning=t(self._meaning),
            language=t(self._language),
        )