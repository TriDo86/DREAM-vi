"""
FastAPI REST API for DREAM meaning embeddings.

Endpoints
---------
POST /embed          → meaning embeddings for a list of sentences
POST /similarity     → cosine similarity between two sentences
POST /similarity_matrix → N×N similarity matrix
GET  /health         → liveness check (for Docker / load balancer)

Run locally
-----------
    uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

Environment variables
---------------------
DREAM_CHECKPOINT   Path to the checkpoint file  (default: checkpoints/dream_best.pt)
DREAM_BACKBONE     HuggingFace model id         (default: sentence-transformers/LaBSE)
DREAM_DEVICE       "cuda" | "cpu" | "mps"       (default: auto-detect)
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dream.pipeline import DREAMPipeline

# ---------------------------------------------------------------------------
# Application lifespan — load model once at startup
# ---------------------------------------------------------------------------

_pipeline: Optional[DREAMPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    checkpoint = os.getenv("DREAM_CHECKPOINT", "checkpoints/dream_best.pt")
    backbone   = os.getenv("DREAM_BACKBONE",   "sentence-transformers/LaBSE")
    device     = os.getenv("DREAM_DEVICE",     None)
    _pipeline  = DREAMPipeline.from_pretrained(
        checkpoint, backbone_name=backbone, device=device
    )
    yield
    _pipeline = None


app = FastAPI(
    title="DREAM Embedding API",
    description=(
        "Language-agnostic sentence embeddings via DREAM disentanglement.\n"
        "Reference: Tiyajamorn et al., EMNLP 2021."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class EmbedRequest(BaseModel):
    sentences: list[str] = Field(..., description="Sentences to embed (any language).")
    normalize: bool      = Field(True, description="L2-normalise the output vectors.")
    batch_size: int      = Field(64,   description="Backbone batch size.")

    @field_validator("sentences")
    @classmethod
    def non_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("sentences must not be empty.")
        if len(v) > 512:
            raise ValueError("Maximum 512 sentences per request.")
        return v


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    num_sentences: int


class SimilarityRequest(BaseModel):
    sentence_a: str = Field(..., max_length=1024)
    sentence_b: str = Field(..., max_length=1024)


class SimilarityResponse(BaseModel):
    similarity:  float
    sentence_a:  str
    sentence_b:  str


class MatrixRequest(BaseModel):
    sentences: list[str] = Field(..., min_length=2)

    @field_validator("sentences")
    @classmethod
    def at_least_two(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("Need at least 2 sentences.")
        if len(v) > 64:
            raise ValueError("Maximum 64 sentences for the similarity matrix.")
        return v


class MatrixResponse(BaseModel):
    matrix:    list[list[float]]
    sentences: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _require_pipeline() -> DREAMPipeline:
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )
    return _pipeline


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


@app.post("/embed", response_model=EmbedResponse, tags=["inference"])
def embed(req: EmbedRequest):
    """Return language-agnostic meaning embeddings for each sentence."""
    pipe = _require_pipeline()
    vectors = pipe.encode(req.sentences, batch_size=req.batch_size, normalize=req.normalize)
    return EmbedResponse(
        embeddings=vectors.tolist(),
        dim=vectors.shape[1],
        num_sentences=len(vectors),
    )


@app.post("/similarity", response_model=SimilarityResponse, tags=["inference"])
def similarity(req: SimilarityRequest):
    """Cosine similarity between two sentences (cross-lingual aware)."""
    pipe  = _require_pipeline()
    score = pipe.similarity(req.sentence_a, req.sentence_b)
    return SimilarityResponse(
        similarity=round(score, 6),
        sentence_a=req.sentence_a,
        sentence_b=req.sentence_b,
    )


@app.post("/similarity_matrix", response_model=MatrixResponse, tags=["inference"])
def similarity_matrix(req: MatrixRequest):
    """Return an N×N cosine similarity matrix for the provided sentences."""
    pipe   = _require_pipeline()
    matrix = pipe.similarity_matrix(req.sentences)
    return MatrixResponse(
        matrix=matrix.tolist(),
        sentences=req.sentences,
    )
