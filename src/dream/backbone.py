"""
backbone.py — Unified backbone abstraction for DREAM.

---------------
DREAM's training and inference pipeline needs sentence embeddings, but
different backbone libraries have incompatible APIs:

    SentenceTransformer : .encode(..., convert_to_tensor=True, device=...)
    BGEM3FlagModel      : .encode(...)["dense_vecs"]   → numpy array
    HuggingFace AutoModel: tokenizer + model + manual mean-pooling

BackboneBase defines a single contract that every adapter must satisfy,
so the rest of the codebase (dataset.py, pipeline.py, train.py) never
needs to know which library is underneath.

Contract
--------
Every BackboneBase.encode() call MUST return:
    torch.Tensor, shape (N, D), dtype float32, on CPU

The caller (dataset for pre-computation, pipeline for inference) is
responsible for moving tensors to the training device when needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn.functional as F


class BackboneBase(ABC):
    """
    Abstract base class for all backbone adapters.

    Subclass this to add support for any new embedding library or API
    without touching dataset.py, pipeline.py, or train.py.

    Required overrides
    ------------------
    encode()        : encode a list of sentences → CPU float32 Tensor
    embedding_dim   : output dimensionality (property)
    model_id        : human-readable model identifier (property)

    Optional overrides
    ------------------
    eval()          : switch to inference mode (default: no-op)

    Example — custom API backbone::

        class MyAPIBackbone(BackboneBase):
            def encode(self, sentences, batch_size=64, normalize=False):
                vecs = my_api.embed(sentences)
                t = torch.tensor(vecs, dtype=torch.float32)
                return F.normalize(t, dim=-1) if normalize else t

            @property
            def embedding_dim(self): return 1024

            @property
            def model_id(self): return "my-api-v1"
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        Encode a list of sentences into dense vectors.

        Args:
            sentences:  List of strings in any language.
            batch_size: Internal batch size for the underlying model.
            normalize:  If True, L2-normalise each vector before returning.
                        Pass normalize=False during pre-computation (dataset)
                        so the raw embedding is stored; the DREAM head and
                        pipeline normalise themselves as needed.

        Returns:
            torch.Tensor of shape (N, embedding_dim), dtype float32, on CPU.
        """
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Output dimensionality of this backbone."""
        ...

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Human-readable identifier, e.g. 'sentence-transformers/LaBSE'."""
        ...

    # ------------------------------------------------------------------
    # Optional hooks — override if the underlying model needs them
    # ------------------------------------------------------------------

    def eval(self) -> "BackboneBase":
        """Switch to inference mode. No-op by default."""
        return self

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}', dim={self.embedding_dim})"


# ──────────────────────────────────────────────────────────────────────────────
# Adapter: sentence-transformers
# ──────────────────────────────────────────────────────────────────────────────

class SentenceTransformerBackbone(BackboneBase):
    """
    Adapter for any model loadable via sentence-transformers.

    Covers: LaBSE, paraphrase-multilingual-*, all-MiniLM-*, mBERT,
    and most community multilingual models on HuggingFace that are
    packaged as SentenceTransformer checkpoints.

    Args:
        model_name: HuggingFace model id or local path.
        device:     Target device for encoding. Embeddings are always
                    returned on CPU regardless of this setting.
    """

    def __init__(self, model_name: str, device: Optional[torch.device] = None) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._model = SentenceTransformer(model_name, device=str(self._device))
        self._dim: int = self._model.get_sentence_embedding_dimension()

    def encode(self, sentences, batch_size=64, normalize=False) -> torch.Tensor:
        vectors = self._model.encode(
            sentences,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=normalize,
            show_progress_bar=len(sentences) > 256,
            device=str(self._device),
        )
        # Contract: always CPU float32
        return vectors.cpu().float()

    def eval(self) -> "SentenceTransformerBackbone":
        self._model.eval()
        return self

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_name


# ──────────────────────────────────────────────────────────────────────────────
# Adapter 2: FlagEmbedding (BGE-M3, BGE-large-en-v1.5, …)
# ──────────────────────────────────────────────────────────────────────────────

class BGEM3Backbone(BackboneBase):
    """
    Adapter for BAAI/bge-* models via the FlagEmbedding library.

    Only dense_vecs are used here. If you need lexical or multi-vector
    retrieval, subclass this and override encode() accordingly.

    Args:
        model_name: HuggingFace model id, e.g. ``"BAAI/bge-m3"``.
        device:     Used to store the output tensor. FlagEmbedding
                    manages its own CUDA placement internally.
        use_fp16:   Pass True to halve memory usage with minor accuracy
                    trade-off (FlagEmbedding default).
        max_length: Token budget per sentence. BGE-M3 supports up to 8192;
                    lower values speed up encoding.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        use_fp16: bool = True,
        max_length: int = 512,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self._model_name = model_name
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._max_length = max_length
        self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

        # Probe once to get the true output dimension
        _probe = self._model.encode(["probe"], batch_size=1)["dense_vecs"]
        self._dim: int = int(_probe.shape[-1])

    def encode(self, sentences, batch_size=64, normalize=False) -> torch.Tensor:
        # FlagEmbedding returns numpy ndarray
        dense = self._model.encode(
            sentences,
            batch_size=batch_size,
            max_length=self._max_length,
        )["dense_vecs"]

        t = torch.from_numpy(dense).float()   # CPU float32 already
        if normalize:
            t = F.normalize(t, dim=-1)
        return t

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_name


# ──────────────────────────────────────────────────────────────────────────────
# Adapter 3: HuggingFace AutoModel (XLM-R, mBERT raw, KaLM, …)
# ──────────────────────────────────────────────────────────────────────────────

class HuggingFaceBackbone(BackboneBase):
    """
    Generic adapter for any HuggingFace AutoModel.

    Uses attention-mask-weighted mean pooling over the last hidden state.
    This works well for encoder-only models (XLM-R, mBERT, BERT variants).

    Supports quantization, flash_attention_2, and device_map="auto" via
    model_kwargs — pass anything that AutoModel.from_pretrained() accepts.

    Args:
        model_name:       HuggingFace model id or local path.
        device:           Target device. Ignored if model_kwargs contains
                          ``device_map`` (let HuggingFace handle placement).
        tokenizer_kwargs: Extra kwargs forwarded to AutoTokenizer.from_pretrained.
        model_kwargs:     Extra kwargs forwarded to AutoModel.from_pretrained.
                          Common uses:
                            {"quantization_config": BitsAndBytesConfig(...)}
                            {"attn_implementation": "flash_attention_2"}
                            {"torch_dtype": torch.float16}

    Example — XLM-R-large with 4-bit quantization::

        from transformers import BitsAndBytesConfig
        backbone = HuggingFaceBackbone(
            "facebook/xlm-roberta-large",
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                "attn_implementation": "flash_attention_2",
            },
        )
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._model_name = model_name
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        tok_kw  = tokenizer_kwargs or {}
        mod_kw  = model_kwargs or {}

        self._tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kw)
        self._model     = AutoModel.from_pretrained(model_name, **mod_kw)

        # Only move manually when device_map is not set by the caller
        if "device_map" not in mod_kw:
            self._model = self._model.to(self._device)

        self._dim: int = self._model.config.hidden_size

    def eval(self) -> "HuggingFaceBackbone":
        self._model.eval()
        return self

    @torch.no_grad()
    def encode(self, sentences, batch_size=64, normalize=False) -> torch.Tensor:
        chunks: list[torch.Tensor] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            output = self._model(**encoded)
            pooled = self._mean_pool(output.last_hidden_state, encoded["attention_mask"])
            chunks.append(pooled.cpu().float())   # Contract: CPU float32

        vectors = torch.cat(chunks, dim=0)
        if normalize:
            vectors = F.normalize(vectors, dim=-1)
        return vectors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask-weighted mean pooling over token dimension."""
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(1) / mask_expanded.sum(1).clamp(min=1e-9)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_name