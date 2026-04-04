"""
Loss functions for DREAM (Tiyajamorn et al., EMNLP 2021).

Total loss:  L = L_R + L_M + L_L

  L_R  — Reconstruction  (eq. 2):
         MSE between the original embedding and ê_M + ê_L.

  L_M  — Meaning         (eqs. 5-7):
         L_x: parallel sentences    → push meaning embeddings together  (+1).
         L_m: random sentences      → push meaning embeddings apart     (-1).

  L_L  — Language        (eqs. 8-11):
         L_m^L: same-language pairs    → push language embeddings together (+1).
                cross-language pairs   → push language embeddings apart    (-1).
         L_i^L: cross-entropy language identification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossComponents:
    """Structured container for the three DREAM loss terms and their total."""

    total:          torch.Tensor
    reconstruction: torch.Tensor
    meaning:        torch.Tensor
    language:       torch.Tensor

    def as_log_dict(self, prefix: str = "") -> dict[str, float]:
        """Return a flat dict suitable for logging (WandB, TensorBoard, etc.)."""
        return {
            f"{prefix}loss/total":          self.total.item(),
            f"{prefix}loss/reconstruction": self.reconstruction.item(),
            f"{prefix}loss/meaning":        self.meaning.item(),
            f"{prefix}loss/language":       self.language.item(),
        }


# ---------------------------------------------------------------------------
# Individual loss terms
# ---------------------------------------------------------------------------

def reconstruction_loss(
    original:  torch.Tensor,
    meaning:   torch.Tensor,
    language:  torch.Tensor,
) -> torch.Tensor:
    """
    L_R (eq. 2) — autoencoder reconstruction.

    Ensures that meaning + language embeddings sum back to the original
    backbone embedding.

    Args:
        original:  (B, D) original backbone embedding.
        meaning:   (B, D) meaning embedding  ê_M.
        language:  (B, D) language embedding ê_L.
    """
    return F.mse_loss(meaning + language, original)


def meaning_loss(
    src_meaning:      torch.Tensor,
    tgt_meaning:      torch.Tensor,
    rand_src_meaning: torch.Tensor,
    rand_tgt_meaning: torch.Tensor,
) -> torch.Tensor:
    """
    L_M (eqs. 5-7) — disentangle meaning from language.

    L_x (eq. 6): parallel pair  → push meaning embeddings together (+1).
    L_m (eq. 7): random pair    → push meaning embeddings apart    (-1).

    Args:
        src_meaning:      (B, D) meaning embedding of source sentences (parallel pair).
        tgt_meaning:      (B, D) meaning embedding of target sentences (parallel pair).
        rand_src_meaning: (B, D) meaning embedding of random source sentences.
        rand_tgt_meaning: (B, D) meaning embedding of random target sentences.
    """
    cos   = torch.nn.CosineEmbeddingLoss()
    B     = src_meaning.size(0)
    pos   = torch.ones(B, device=src_meaning.device)
    neg   = torch.full((B,), -1.0, device=src_meaning.device)

    # L_x: parallel pair → together
    L_x = cos(src_meaning, tgt_meaning, pos)

    # L_m: random pair → apart
    L_m = (
        cos(src_meaning, rand_src_meaning, neg)
      + cos(tgt_meaning, rand_tgt_meaning, neg)
    )

    return L_x + L_m


def language_loss(
    src_language:      torch.Tensor,
    tgt_language:      torch.Tensor,
    rand_src_language: torch.Tensor,
    rand_tgt_language: torch.Tensor,
    src_logits:        torch.Tensor,
    tgt_logits:        torch.Tensor,
    src_lang_ids:      torch.Tensor,
    tgt_lang_ids:      torch.Tensor,
) -> torch.Tensor:
    """
    L_L (eqs. 8-11) — preserve language-specific information.

    L_m^L (eq. 9):
        same-language pairs  → push language embeddings together (+1).
        cross-language pairs → push language embeddings apart    (-1).
    L_i^L (eq. 11): cross-entropy language identification.

    Args:
        src_language:      (B, D) language embedding of source sentences.
        tgt_language:      (B, D) language embedding of target sentences.
        rand_src_language: (B, D) language embedding of random same-language source sentences.
        rand_tgt_language: (B, D) language embedding of random same-language target sentences.
        src_logits:        (B, num_languages) raw logits for source language identification.
        tgt_logits:        (B, num_languages) raw logits for target language identification.
        src_lang_ids:      (B,) ground-truth language IDs for source sentences.
        tgt_lang_ids:      (B,) ground-truth language IDs for target sentences.
    """
    cos   = torch.nn.CosineEmbeddingLoss()
    B     = src_language.size(0)
    pos   = torch.ones(B, device=src_language.device)
    neg   = torch.full((B,), -1.0, device=src_language.device)

    # L_m^L — Language Embedding Loss (PullOnly)
    # Pull same-language embeddings closer together.
    # No push force between embeddings of different languages.
    L_m = (
        cos(src_language, rand_src_language, pos)   # en vs en → closer
      + cos(tgt_language, rand_tgt_language, pos)   # de vs de → closer
    )

    # L_m^L — Language Embedding Loss (PullNPush)
    # Pull same-language embeddings closer together (Pull)
    # and push different-language embeddings further apart (Push).
    # L_m = (
    #     cos(src_language, rand_src_language, pos)   # en vs en → closer  (Pull)
    #   + cos(tgt_language, rand_tgt_language, pos)   # de vs de → closer  (Pull)
    #   + cos(src_language, tgt_language,      neg)   # en vs de → apart   (Push)
    # )

    # L_i^L: language identification
    L_i = (
        F.cross_entropy(src_logits, src_lang_ids)
      + F.cross_entropy(tgt_logits, tgt_lang_ids)
    )

    return L_m + L_i


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_loss(
    # Original backbone embeddings
    src_original:      torch.Tensor,
    tgt_original:      torch.Tensor,
    # Meaning branch
    src_meaning:       torch.Tensor,
    tgt_meaning:       torch.Tensor,
    rand_src_meaning:  torch.Tensor,
    rand_tgt_meaning:  torch.Tensor,
    # Language branch
    src_language:      torch.Tensor,
    tgt_language:      torch.Tensor,
    rand_src_language: torch.Tensor,
    rand_tgt_language: torch.Tensor,
    # Language-id logits and ground-truth labels
    src_logits:        torch.Tensor,
    tgt_logits:        torch.Tensor,
    src_lang_ids:      torch.Tensor,
    tgt_lang_ids:      torch.Tensor,
) -> LossComponents:
    """
    Compute L = L_R + L_M + L_L and return all components.

    Returns a :class:`LossComponents` dataclass whose ``.total`` should be
    passed to ``.backward()``.
    """
    L_R = (
        reconstruction_loss(src_original, src_meaning, src_language)
      + reconstruction_loss(tgt_original, tgt_meaning, tgt_language)
    )
    L_M = meaning_loss(
        src_meaning, tgt_meaning,
        rand_src_meaning, rand_tgt_meaning,
    )
    L_L = language_loss(
        src_language, tgt_language,
        rand_src_language, rand_tgt_language,
        src_logits, tgt_logits,
        src_lang_ids, tgt_lang_ids,
    )
    return LossComponents(
        total=L_R + L_M + L_L,
        reconstruction=L_R,
        meaning=L_M,
        language=L_L,
    )