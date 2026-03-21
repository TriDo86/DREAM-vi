"""
dream — Language-agnostic sentence embeddings via DREAM disentanglement.

Typical usage::

    from dream import DREAMPipeline

    pipe = DREAMPipeline.from_pretrained("checkpoints/dream_best.pt")
    score = pipe.similarity("The cat sat on the mat.", "Le chat était assis sur le tapis.")
"""

from .pipeline import DREAMPipeline
from .model    import DREAMModel

__all__ = ["DREAMPipeline", "DREAMModel"]
__version__ = "0.1.0"
