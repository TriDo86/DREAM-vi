"""
Training entry point.

Usage
-----
    # Basic run
    python scripts/train.py

    # Custom config
    python scripts/train.py --config configs/train.yaml

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/dream_epoch_0010.pt

    # Override a config value without editing the YAML
    python scripts/train.py --epochs 50 --lr 5e-5
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dream.dataset import MultilingualDataset, free_backbone
from dream.model   import DREAMModel
from dream.trainer import Trainer, TrainerConfig
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed all RNGs for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the DREAM disentanglement model.")
    p.add_argument("--config",  default="configs/train.yaml", help="Path to YAML config.")
    p.add_argument("--resume",  default=None, help="Path to checkpoint to resume from.")
    # Optional CLI overrides (take priority over YAML)
    p.add_argument("--epochs",  type=int,   default=None)
    p.add_argument("--lr",      type=float, default=None)
    p.add_argument("--device",  type=str,   default=None)
    p.add_argument("--seed",    type=int,   default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg  = load_yaml(args.config)

    # ── Merge CLI overrides ──────────────────────────────────────────────────
    trainer_cfg = cfg.get("trainer", {})
    if args.epochs: trainer_cfg["epochs"]        = args.epochs
    if args.lr:     trainer_cfg["learning_rate"] = args.lr
    if args.device: trainer_cfg["device"]        = args.device
    if args.seed:   trainer_cfg["seed"]          = args.seed

    model_cfg = cfg.get("model", {})
    data_cfg  = cfg.get("data",  {})

    tcfg = TrainerConfig(**trainer_cfg)
    set_seed(tcfg.seed)

    logger.info("=" * 60)
    logger.info(f"Config: {tcfg}")
    logger.info("=" * 60)

    # ── WandB init (un-comment to activate) ─────────────────────────────────
    # import wandb
    # wandb.init(project="dream-embed", config={**trainer_cfg, **model_cfg, **data_cfg})

    # ── Backbone (used ONLY for pre-computing embeddings) ────────────────────
    backbone_name = model_cfg.get("backbone", "sentence-transformers/LaBSE")
    logger.info(f"Loading backbone: {backbone_name}")
    backbone = SentenceTransformer(backbone_name)

    # ── Datasets ─────────────────────────────────────────────────────────────
    logger.info("Pre-computing training embeddings …")
    train_ds = MultilingualDataset(
        data_dir=data_cfg["train_dir"],
        backbone=backbone,
        encode_batch=data_cfg.get("encode_batch", 64),
    )

    logger.info("Pre-computing validation embeddings …")
    val_ds = MultilingualDataset(
        data_dir=data_cfg["val_dir"],
        backbone=backbone,
        encode_batch=data_cfg.get("encode_batch", 64),
    )

    # Free backbone GPU memory — training uses only the MLP heads
    logger.info("Releasing backbone from memory …")
    free_backbone(backbone)

    # ── DataLoaders ──────────────────────────────────────────────────────────
    batch_size = data_cfg.get("batch_size", 512)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,       # shuffling is done by dataset.shuffle() each epoch
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    embedding_dim = backbone.get_sentence_embedding_dimension() if hasattr(backbone, "get_sentence_embedding_dimension") else model_cfg.get("embedding_dim", 768)
    # backbone was deleted — fall back to config value
    embedding_dim = model_cfg.get("embedding_dim", 768)
    num_languages = len(train_ds._datasets) + 1  # +1 for English

    model = DREAMModel(
        embedding_dim=embedding_dim,
        num_languages=num_languages,
    )
    logger.info(f"Model: {model}")

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, train_loader, val_loader, tcfg)

    if args.resume:
        trainer.resume(args.resume)

    trainer.fit()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
