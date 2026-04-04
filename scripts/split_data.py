"""
scripts/split_data.py — Split Tatoeba TSV files into train / val sets.

Usage
-----
    # Default: 90/10 split, reads from data/Tatoeba, writes to data/Tatoeba_Train & data/Tatoeba_Val
    python scripts/split_data.py

    # Custom split ratio and paths
    python scripts/split_data.py --val-ratio 0.15 --src data/Tatoeba --train data/Tatoeba_Train --val data/Tatoeba_Val

    # Fix random seed for reproducibility
    python scripts/split_data.py --seed 42
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split Tatoeba TSV files into train/val sets.")
    p.add_argument(
        "--src",
        default="data/Tatoeba",
        help="Directory containing raw *.tsv files (default: data/Tatoeba).",
    )
    p.add_argument(
        "--train",
        default="data/Tatoeba_Train",
        help="Output directory for training split (default: data/Tatoeba_Train).",
    )
    p.add_argument(
        "--val",
        default="data/Tatoeba_Val",
        help="Output directory for validation split (default: data/Tatoeba_Val).",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1 = 10%%).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=86,
        help="Random seed for reproducibility (default: 86).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def split_file(
    tsv_path: Path,
    train_dir: Path,
    val_dir: Path,
    val_ratio: float,
    seed: int,
) -> dict:
    """
    Split a single TSV file into train and val sets.

    Returns a summary dict with file name and split sizes.
    """
    df = pd.read_csv(
        tsv_path,
        sep="\t",
        header=None,
        names=["src_id", "src", "tgt_id", "tgt"],
        on_bad_lines="skip",
        dtype={"src_id": str, "tgt_id": str},
    ).dropna(subset=["src_id", "src", "tgt_id", "tgt"])

    if len(df) == 0:
        logger.warning("  Skipping %s — no valid rows found.", tsv_path.name)
        return {}

    # Shuffle and split
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    val_size   = max(1, int(len(df) * val_ratio))
    train_size = len(df) - val_size

    df_train = df.iloc[:train_size]
    df_val   = df.iloc[train_size:]

    # Write — preserve original filename in both output dirs
    out_train = train_dir / tsv_path.name
    out_val   = val_dir   / tsv_path.name

    df_train.to_csv(out_train, sep="\t", header=False, index=False)
    df_val.to_csv(out_val,   sep="\t", header=False, index=False)

    return {
        "file":  tsv_path.name,
        "total": len(df),
        "train": train_size,
        "val":   val_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    src_dir   = Path(args.src)
    train_dir = Path(args.train)
    val_dir   = Path(args.val)

    # Validate source
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    tsv_files = sorted(src_dir.glob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"No .tsv files found in: {src_dir}")

    # Create output dirs
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Source      : %s  (%d files)", src_dir, len(tsv_files))
    logger.info("Train output: %s", train_dir)
    logger.info("Val output  : %s", val_dir)
    logger.info("Val ratio   : %.0f%%", args.val_ratio * 100)
    logger.info("Seed        : %d", args.seed)
    logger.info("-" * 60)

    summaries = []
    for tsv_path in tsv_files:
        summary = split_file(tsv_path, train_dir, val_dir, args.val_ratio, args.seed)
        if summary:
            summaries.append(summary)
            logger.info(
                "  %-55s  total=%6d  train=%6d  val=%5d",
                summary["file"],
                summary["total"],
                summary["train"],
                summary["val"],
            )

    logger.info("-" * 60)
    total_rows  = sum(s["total"] for s in summaries)
    total_train = sum(s["train"] for s in summaries)
    total_val   = sum(s["val"]   for s in summaries)
    logger.info(
        "Total  %-49s  total=%6d  train=%6d  val=%5d",
        f"({len(summaries)} files)",
        total_rows,
        total_train,
        total_val,
    )


if __name__ == "__main__":
    main()