"""
DREAMPipeline — end-to-end inference interface.

This is the class that:
  • your FastAPI app loads once at startup,
  • your Gradio demo calls,
  • a user installs and uses in two lines:

      from dream import DREAMPipeline
      pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt")
      print(pipe.similarity("Hello world", "Hola mundo"))   # ~0.92

Design
------
The pipeline owns both the frozen backbone (SentenceTransformer) and the
trained DREAMModel.  Backbone is configurable: pass any model available on
HuggingFace (LaBSE, mBERT, XLM-R-large, etc.) as long as its output
dimensionality matches the checkpoint.

The pipeline is thread-safe for concurrent read-only inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .backbone import BackboneBase
from .backbone_factory import BackboneInput, create_backbone
from .model import DREAMModel
from .dataset import DEFAULT_LANGUAGE_MAP

_NUM_LANGUAGES = len(DEFAULT_LANGUAGE_MAP)
_DEFAULT_BACKBONE = "sentence-transformers/LaBSE"


class DREAMPipeline:
    """
    Frozen backbone + trained DREAMModel, packaged for inference.

    Args:
        backbone:         A model id string (auto-detected) or a BackboneBase
                          instance for full control over quantization, pooling, etc.
        checkpoint_path:  Path to a ``dream_*.pt`` checkpoint.
                          If None, the DREAMModel is randomly initialised
                          (useful for testing the pipeline structure).
        device:           ``"cuda"`` | ``"cpu"`` | ``"mps"``.
                          Defaults to the best available device.
        backbone_type:    ``"st"`` | ``"bge"`` | ``"hf"`` | ``"auto"``.
                          Only used when backbone is a string.
        backbone_kwargs:  Extra kwargs forwarded to the backbone adapter constructor.
    """

    def __init__(
        self,
        backbone: BackboneInput = _DEFAULT_BACKBONE,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        backbone_type: str = "auto",
        backbone_kwargs: Optional[dict] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # ── Backbone (frozen) ─────────────────────────────────────────────
        self.backbone: BackboneBase = create_backbone(
            backbone,
            device=self.device,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
        )
        self.backbone.eval()
        embedding_dim = self.backbone.embedding_dim

        # ── DREAM head ────────────────────────────────────────────────────
        num_languages = _NUM_LANGUAGES
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            num_languages = ckpt.get(
                "num_languages",
                ckpt["model_state"]["language_identifier.weight"].shape[0],
            )
            # Validate embedding_dim matches
            ckpt_dim = ckpt.get(
                "embedding_dim",
                ckpt["model_state"]["meaning_encoder.weight"].shape[0],
            )
            if ckpt_dim != embedding_dim:
                raise ValueError(
                    f"Backbone '{self.backbone.model_id}' has embedding_dim={embedding_dim}, "
                    f"but checkpoint expects {ckpt_dim}. Use the same backbone as during training."
                )

        self.dream = DREAMModel(embedding_dim, num_languages).to(self.device)

        if checkpoint_path is not None:
            self.dream.load_state_dict(ckpt["model_state"])
        self.dream.eval()

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        backbone: BackboneInput = _DEFAULT_BACKBONE,
        device: Optional[str] = None,
        backbone_type: str = "auto",
        backbone_kwargs: Optional[dict] = None,
    ) -> "DREAMPipeline":
        """
        Load a fully trained pipeline from a checkpoint file.

        Example::

            # Default backbone
            pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt")

            # Different backbone by string
            pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt",
                                                 backbone="BAAI/bge-m3")

            # Custom backbone instance (quantized, etc.)
            pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt",
                                                 backbone=my_backbone_instance)
        """
        return cls(
            backbone=backbone,
            checkpoint_path=checkpoint_path,
            device=device,
            backbone_type=backbone_type,
            backbone_kwargs=backbone_kwargs,
        )

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[str, list[str]],
        batch_size: int = 64,
        normalize: bool = True,
        only_backbone = False
    ) -> np.ndarray:
        """
        Encode sentences into language-agnostic meaning embeddings.

        Args:
            sentences:  A single string or a list of strings (any language).
            batch_size: Batch size for the backbone encoder.
            normalize:  L2-normalise the output.  Recommended for cosine similarity.

        Returns:
            numpy array of shape ``(N, embedding_dim)`` with dtype float32.
        """
        if isinstance(sentences, str):
            sentences = [sentences]

        if only_backbone:
            raw = self.backbone.encode(sentences, batch_size=batch_size, normalize=normalize)
            return raw.cpu().float().numpy()

        # Step 1: frozen backbone → raw sentence embeddings (CPU float32)
        raw: torch.Tensor = self.backbone.encode(
            sentences, batch_size=batch_size, normalize=False
        ).to(self.device)  # move to training device for DREAM head

        # Step 2: DREAM head → meaning sub-space
        meaning = self.dream.encode_meaning(raw, normalize=normalize)  # (N, D)

        return meaning.cpu().float().numpy()
    
    @torch.no_grad()
    def dream_forward(self, 
                    sentences: Union[str, list[str]],
                    batch_size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        # Step 1: frozen backbone → raw sentence embeddings (CPU float32)
        raw: torch.Tensor = self.backbone.encode(
            sentences, batch_size=batch_size, normalize=False
        ).to(self.device)  # move to training device for DREAM head

        # Step 2: full DREAM forward pass
        meaning, language, logits = self.dream.forward(raw)

        to_cpu_ndarray = lambda tensor: tensor.cpu().float().numpy()
        return to_cpu_ndarray(meaning), to_cpu_ndarray(language), to_cpu_ndarray(logits)
        

    @torch.no_grad()
    def similarity(self, sentence_a: str, sentence_b: str, only_backbone=False) -> float:
        """
        Cosine similarity between two sentences (cross-lingual aware).

        Returns a float in [-1, 1].  Values near 1.0 indicate high semantic overlap.

        Example::

            score = pipe.similarity("The cat is on the mat.", "Le chat est sur le tapis.")
        """
        embs = self.encode([sentence_a, sentence_b], only_backbone=only_backbone)   # (2, D), normalised
        return float(np.dot(embs[0], embs[1]))

    @torch.no_grad()
    def similarity_matrix(self, sentences: list[str], only_backbone=False) -> np.ndarray:
        """
        Compute an N×N cosine similarity matrix.

        Useful for the Gradio demo heatmap and batch cross-lingual comparisons.

        Returns:
            numpy array of shape ``(N, N)`` with values in [-1, 1].
        """
        embs = self.encode(sentences, only_backbone=only_backbone)       # (N, D), normalised
        return (embs @ embs.T).astype(np.float32)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"DREAMPipeline("
            f"backbone='{self.backbone.model_id}', "
            f"dream={self.dream}, "
            f"device={self.device})"
        )