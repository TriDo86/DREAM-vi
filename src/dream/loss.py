"""
Loss functions for DREAM (Tiyajamorn et al., EMNLP 2021).

Total loss:  L = L_R  +  L_M  +  L_L

  L_R  — Reconstruction  (eq. 2):
         MSE between the original embedding and ê_M + ê_L.

  L_M  — Meaning         (eqs. 5-7):
         L_x: parallel sentences → push meaning embeddings together.
         L_m: random same-language sentences → push meaning embeddings apart.

  L_L  — Language        (eqs. 8-11):
         L_m^L: same-language embeddings → push language embeddings together.
         L_i^L: cross-entropy language identification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossComponents:
    """Structured container for the three DREAM loss terms + total."""
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
    e:   torch.Tensor,
    e_m: torch.Tensor,
    e_l: torch.Tensor,
) -> torch.Tensor:
    """
    L_R  (eq. 2) — autoencoder reconstruction.

    Ensures that meaning + language embeddings sum back to the input.

    Args:
        e:   (B, D) original backbone embedding.
        e_m: (B, D) meaning embedding.
        e_l: (B, D) language embedding.
    """
    return F.mse_loss(e_m + e_l, e)


def meaning_loss(
    s_m:      torch.Tensor,
    t_m:      torch.Tensor,
    rand_s_m: torch.Tensor,
    rand_t_m: torch.Tensor,
) -> torch.Tensor:
    """
    L_M  (eqs. 5-7) — disentangle meaning from language.

    L_x (eq. 6): parallel pair → minimise cosine distance between s_m and t_m.
    L_m (eq. 7): random pair   → hinge loss, push apart when too similar.

    Args:
        s_m:      (B, D) meaning embedding of source sentences (parallel).
        t_m:      (B, D) meaning embedding of target sentences (parallel).
        rand_s_m: (B, D) meaning embedding of random source sentences.
        rand_t_m: (B, D) meaning embedding of random target sentences.
    """
    # L_x: push parallel meanings together
    L_x = (1.0 - F.cosine_similarity(s_m, t_m, dim=-1)).mean()

    # L_m: push random meanings apart (hinge at 0)
    L_m = (
        F.relu(F.cosine_similarity(s_m, rand_s_m, dim=-1))
        + F.relu(F.cosine_similarity(t_m, rand_t_m, dim=-1))
    ).mean()

    return L_x + L_m


def language_loss(
    s_l:        torch.Tensor,
    t_l:        torch.Tensor,
    rand_s_l:   torch.Tensor,
    rand_t_l:   torch.Tensor,
    s_logits:   torch.Tensor,
    t_logits:   torch.Tensor,
    src_lang_ids: torch.Tensor,
    tgt_lang_ids: torch.Tensor,
) -> torch.Tensor:
    """
    L_L  (eqs. 8-11) — preserve language-specific information.

    L_m^L (eq. 9): same-language embeddings → push together (cosine distance).
    L_i^L (eq. 11): cross-entropy for language identification.

    Args:
        s_l, t_l:           (B, D) language embeddings of src / tgt sentences.
        rand_s_l, rand_t_l: (B, D) language embeddings of random sentences.
        s_logits, t_logits: (B, num_languages) raw lang-id logits.
        src_lang_ids:       (B,) ground-truth language IDs for source.
        tgt_lang_ids:       (B,) ground-truth language IDs for target.
    """
    # L_m^L: push same-language embeddings together
    L_m = (
        2.0
        - F.cosine_similarity(s_l, rand_s_l, dim=-1)
        - F.cosine_similarity(t_l, rand_t_l, dim=-1)
    ).mean()

    # L_i^L: language identification (classification)
    L_i = (
        F.cross_entropy(s_logits, src_lang_ids)
        + F.cross_entropy(t_logits, tgt_lang_ids)
    )

    return L_m + L_i


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def compute_loss(
    # original backbone embeddings
    e_src: torch.Tensor,
    e_tgt: torch.Tensor,
    # meaning branch
    em_src: torch.Tensor,
    em_tgt: torch.Tensor,
    em_rand_src: torch.Tensor,
    em_rand_tgt: torch.Tensor,
    # language branch
    el_src: torch.Tensor,
    el_tgt: torch.Tensor,
    el_rand_src: torch.Tensor,
    el_rand_tgt: torch.Tensor,
    # language-id logits + ground-truth labels
    logits_src: torch.Tensor,
    logits_tgt: torch.Tensor,
    src_lang_ids: torch.Tensor,
    tgt_lang_ids: torch.Tensor,
) -> LossComponents:
    """
    Compute L = L_R + L_M + L_L and return all components.

    Returns a :class:`LossComponents` dataclass whose ``.total`` should be
    passed to ``.backward()``.
    """
    L_R = reconstruction_loss(e_src, em_src, el_src) + reconstruction_loss(e_tgt, em_tgt, el_tgt)
    L_M = meaning_loss(em_src, em_tgt, em_rand_src, em_rand_tgt)
    L_L = language_loss(
        el_src, el_tgt, el_rand_src, el_rand_tgt,
        logits_src, logits_tgt, src_lang_ids, tgt_lang_ids,
    )
    return LossComponents(total=L_R + L_M + L_L, reconstruction=L_R, meaning=L_M, language=L_L)
