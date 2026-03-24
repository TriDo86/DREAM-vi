"""
Evaluate Pearson correlation on the SemEval-2017 cross-lingual STS task.

This reproduces Table 4 in the paper (Tiyajamorn et al., EMNLP 2021).

Dataset
-------
Download STS2017-extended.zip from:
https://public.ukp.informatik.tu-darmstadt.de/reimers/sentence-transformers/datasets/STS2017-extended.zip

Unzip to data/STS2017-extended/

Usage
-----
    python scripts/evaluate.py --checkpoint checkpoints/dream_best.pt
    python scripts/evaluate.py --checkpoint checkpoints/dream_best.pt --sts_dir data/STS2017-extended
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dream.pipeline import DREAMPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sts_file(path: Path) -> tuple[list[str], list[str], list[float]]:
    """
    Load a STS file.  Expected columns: score \\t sentence1 \\t sentence2
    """
    all_txt = glob.glob(f'{path}/*.txt')
    dfs = [
        pd.read_csv(
        txt_path,
        sep="\t",
        header=None,
        names=["sent1", "sent2", "score"],
        on_bad_lines="skip",
        dtype=str).dropna()
        for txt_path in all_txt]
    
    df = pd.concat(dfs)

    scores = df["score"].astype(float).tolist()
    sent1  = df["sent1"].tolist()
    sent2  = df["sent2"].tolist()
    return sent1, sent2, scores


def evaluate_pair(
    pipeline: DREAMPipeline,
    sent1: list[str],
    sent2: list[str],
    scores: list[float],
) -> float:
    """Compute Pearson correlation between predicted cosine similarity and human scores."""
    embs1 = pipeline.encode(sent1)   # (N, D), normalised
    embs2 = pipeline.encode(sent2)

    cosines = (embs1 * embs2).sum(axis=1).tolist()  # dot of normalised = cosine
    r, _ = pearsonr(cosines, scores)
    return float(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to dream_best.pt")
    p.add_argument("--backbone",   default="sentence-transformers/LaBSE")
    p.add_argument("--sts_dir",    default="data/STS2017-extended")
    p.add_argument("--device",     default=None)
    args = p.parse_args()

    print(f"Loading pipeline from '{args.checkpoint}' …")
    pipe = DREAMPipeline.from_pretrained(
        args.checkpoint,
        backbone_name=args.backbone,
        device=args.device,
    )

    sts_dir = Path(args.sts_dir)
    results: dict[str, float] = {}

    for sts_file in sorted(sts_dir.glob("STS.input.*.txt")):
        # Extract language pair from filename, e.g. "STS.input.track5.en-ar.txt" → "en-ar"
        pair = sts_file.stem.split(".")[-1]
        gs_file = sts_file.parent / sts_file.name.replace("input", "gs") 

        if not gs_file.exists():
            print(f"  Skipping {sts_file.name} — gold standard not found.")
            continue

        sent1, sent2, scores = load_sts_file(sts_file)
        # Gold-standard file may have only scores
        _, _, gs_scores = load_sts_file(gs_file) if "\t" in gs_file.read_text()[:200] else (None, None, [float(l.strip()) for l in gs_file.read_text().splitlines() if l.strip()])

        if not gs_scores:
            gs_scores = scores  # fallback: scores column in input file
            sent1_eval, sent2_eval, _ = load_sts_file(sts_file)
        else:
            sent1_eval, sent2_eval = sent1, sent2

        r = evaluate_pair(pipe, sent1_eval, sent2_eval, gs_scores)
        results[pair] = r
        print(f"  {pair:12s}  Pearson = {r:.4f}")

    if results:
        avg = np.mean(list(results.values()))
        print(f"\n{'Average':12s}  Pearson = {avg:.4f}")
        print("\nFull results:")
        for pair, r in sorted(results.items()):
            print(f"  {pair}: {r:.4f}")


if __name__ == "__main__":
    main()
