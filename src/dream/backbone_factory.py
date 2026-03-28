"""
backbone_factory.py — Create backbone adapters from config or instance.

This module is the single entry point for constructing a BackboneBase.
Neither train.py, dataset.py, nor pipeline.py should import adapter
classes directly — they call create_backbone() and work with the
abstract BackboneBase interface.

backbone_type values
--------------------
  "st"   → SentenceTransformerBackbone   (sentence-transformers library)
  "bge"  → BGEM3Backbone                 (FlagEmbedding library)
  "hf"   → HuggingFaceBackbone           (transformers AutoModel)
  "auto" → heuristic detection from model_name prefix (default)

Passing a BackboneBase instance directly always bypasses the factory —
the instance is returned unchanged. This is the escape hatch for custom
backbones that don't fit any of the three adapters above.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from .backbone import (
    BGEM3Backbone,
    BackboneBase,
    HuggingFaceBackbone,
    SentenceTransformerBackbone,
)

# Type alias used throughout the codebase
BackboneInput = Union[str, BackboneBase]

# ── Prefix heuristics for "auto" mode ────────────────────────────────────────
# Checked in order; first match wins.
_BGE_PREFIXES = ("BAAI/bge", "bge-")
_ST_PREFIXES  = (
    "sentence-transformers/",
    "paraphrase-",
    "all-minilm",
    "all-mpnet",
    "multi-qa-",
    "msmarco-",
)


def create_backbone(
    backbone: BackboneInput,
    device: Optional[torch.device] = None,
    backbone_type: str = "auto",
    backbone_kwargs: Optional[dict] = None,
) -> BackboneBase:
    """
    Construct and return a BackboneBase from a model name or existing instance.

    Args:
        backbone:        Either a model id string (e.g. ``"BAAI/bge-m3"``) or
                         an already-constructed BackboneBase instance.
                         When an instance is passed, all other arguments
                         are ignored and the instance is returned as-is.
        device:          Target device.  Defaults to CUDA if available.
        backbone_type:   One of ``"st"``, ``"bge"``, ``"hf"``, ``"auto"``.
                         ``"auto"`` uses prefix heuristics on the model name.
        backbone_kwargs: Extra keyword arguments forwarded to the adapter
                         constructor (e.g. ``{"use_fp16": True}`` for BGE,
                         ``{"model_kwargs": {"load_in_4bit": True}}`` for HF).

    Returns:
        A BackboneBase instance ready to call .encode() on.

    Raises:
        TypeError:  If backbone is not a str or BackboneBase.
        ValueError: If backbone_type is unrecognised.

    Examples::

        # From string — auto-detect
        backbone = create_backbone("sentence-transformers/LaBSE")

        # From string — explicit type
        backbone = create_backbone("BAAI/bge-m3", backbone_type="bge")

        # From string — HuggingFace with quantization
        from transformers import BitsAndBytesConfig
        backbone = create_backbone(
            "FacebookAI/xlm-roberta-large",
            backbone_type="hf",
            backbone_kwargs={
                "model_kwargs": {
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
                    "attn_implementation": "flash_attention_2",
                }
            },
        )

        # From instance — bypasses factory entirely
        my_backbone = MyCustomBackbone(...)
        backbone = create_backbone(my_backbone)
    """
    # ── Instance passthrough ──────────────────────────────────────────────
    if isinstance(backbone, BackboneBase):
        return backbone

    if not isinstance(backbone, str):
        raise TypeError(
            f"backbone must be a model id string or a BackboneBase instance, "
            f"got {type(backbone).__name__}."
        )

    device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    kw = backbone_kwargs or {}

    # ── Explicit type ─────────────────────────────────────────────────────
    if backbone_type == "st":
        return SentenceTransformerBackbone(backbone, device=device, **kw)

    if backbone_type == "bge":
        return BGEM3Backbone(backbone, device=device, **kw)

    if backbone_type == "hf":
        return HuggingFaceBackbone(backbone, device=device, **kw)

    if backbone_type != "auto":
        raise ValueError(
            f"Unknown backbone_type={backbone_type!r}. "
            f"Choose from 'st', 'bge', 'hf', or 'auto'."
        )

    # ── Auto-detect from model name prefix ───────────────────────────────
    name_lower = backbone.lower()

    if any(name_lower.startswith(p.lower()) for p in _BGE_PREFIXES):
        return BGEM3Backbone(backbone, device=device, **kw)

    if any(name_lower.startswith(p.lower()) for p in _ST_PREFIXES):
        return SentenceTransformerBackbone(backbone, device=device, **kw)

    # Final fallback: try SentenceTransformer first (it supports many HF
    # models transparently), then fall back to raw HuggingFace AutoModel.
    try:
        return SentenceTransformerBackbone(backbone, device=device, **kw)
    except Exception:
        return HuggingFaceBackbone(backbone, device=device, **kw)