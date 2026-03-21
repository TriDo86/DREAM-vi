"""
DREAM model: Disentangled Representation for Meaning.

Design note
-----------
The backbone (LaBSE / mBERT / XLM-R) is FROZEN and used only to pre-compute
sentence embeddings once before training.  DREAMModel therefore operates on
*pre-computed float tensors*, not on raw text / token ids.

This makes the training loop extremely fast: the MLP heads are tiny compared
to any transformer backbone.

At inference time, DREAMPipeline (see pipeline.py) chains backbone → DREAMModel
transparently, so end-users never need to think about this split.

Reference: Tiyajamorn et al., EMNLP 2021 – Section 3.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DREAMModel(nn.Module):
    """
    Two-headed single-layer MLP autoencoder.

    Splits a frozen multilingual sentence embedding  e ∈ R^d  into:
      • meaning embedding   ê_M  — language-agnostic
      • language embedding  ê_L  — language-specific

    Reconstruction constraint:  ê_M + ê_L ≈ e  (autoencoder).

    Architecture follows the paper exactly: *single-layer feedforward* for both
    heads.  Do NOT add LayerNorm / ReLU / Dropout here — that would be a
    modification, not a faithful implementation.

    Args:
        embedding_dim:  Dimensionality of the backbone output.
                        LaBSE / mBERT → 768, XLM-R-large → 1024.
        num_languages:  Number of language classes for the identification head.
    """

    def __init__(self, embedding_dim: int, num_languages: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_languages = num_languages

        self.meaning_encoder     = nn.Linear(embedding_dim, embedding_dim)
        self.language_encoder    = nn.Linear(embedding_dim, embedding_dim)
        self.language_identifier = nn.Linear(embedding_dim, num_languages)

    # ------------------------------------------------------------------

    def forward(
        self, sentence_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            sentence_embedding: (B, embedding_dim) — output of a frozen backbone.

        Returns:
            meaning_embedding:   (B, embedding_dim)
            language_embedding:  (B, embedding_dim)
            language_logits:     (B, num_languages) — raw logits for lang-id loss
        """
        meaning  = self.meaning_encoder(sentence_embedding)
        language = self.language_encoder(sentence_embedding)
        logits   = self.language_identifier(language)
        return meaning, language, logits

    @torch.no_grad()
    def encode_meaning(
        self, sentence_embedding: torch.Tensor, normalize: bool = True
    ) -> torch.Tensor:
        """
        Convenience method for inference: backbone embedding → L2-normalised
        meaning embedding, no gradient tracking.

        Args:
            sentence_embedding: (B, embedding_dim)
            normalize: L2-normalise output (recommended for cosine similarity).

        Returns:
            (B, embedding_dim)
        """
        meaning, _, _ = self.forward(sentence_embedding)
        if normalize:
            meaning = F.normalize(meaning, p=2, dim=-1)
        return meaning

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}, num_languages={self.num_languages}"
