"""
DREAM — Disentangled Representation for Cross-lingual Meaning.

Reimplementation of Tiyajamorn et al., EMNLP 2021.
https://aclanthology.org/2021.emnlp-main.612

Public API
----------
    from dream import DREAMPipeline

    pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt")
    score = pipe.similarity("Hello world", "Hola mundo")
"""

from .pipeline import DREAMPipeline
from .model import DREAMModel
from .backbone_factory import create_backbone
from .dataset import MultilingualDataset, DEFAULT_LANGUAGE_MAP

__all__ = [
    "DREAMPipeline",
    "DREAMModel",
    "create_backbone",
    "MultilingualDataset",
    "DEFAULT_LANGUAGE_MAP",
]
