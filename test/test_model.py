"""Unit tests for DREAMModel."""

import pytest
import torch
from dream.model import DREAMModel


EMBEDDING_DIM = 768
NUM_LANGUAGES = 8
BATCH_SIZE    = 4


@pytest.fixture
def model() -> DREAMModel:
    return DREAMModel(embedding_dim=EMBEDDING_DIM, num_languages=NUM_LANGUAGES)


@pytest.fixture
def dummy_input() -> torch.Tensor:
    return torch.randn(BATCH_SIZE, EMBEDDING_DIM)


# ---------------------------------------------------------------------------

def test_output_shapes(model, dummy_input):
    meaning, language, logits = model(dummy_input)
    assert meaning.shape  == (BATCH_SIZE, EMBEDDING_DIM)
    assert language.shape == (BATCH_SIZE, EMBEDDING_DIM)
    assert logits.shape   == (BATCH_SIZE, NUM_LANGUAGES)


def test_reconstruction_possible(model, dummy_input):
    """meaning + language should have same shape as input (autoencoder constraint)."""
    meaning, language, _ = model(dummy_input)
    reconstruction = meaning + language
    assert reconstruction.shape == dummy_input.shape


def test_encode_meaning_normalized(model, dummy_input):
    emb = model.encode_meaning(dummy_input, normalize=True)
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(BATCH_SIZE), atol=1e-5)


def test_encode_meaning_no_grad(model, dummy_input):
    dummy_input.requires_grad_(True)
    emb = model.encode_meaning(dummy_input)
    assert not emb.requires_grad


def test_different_dims():
    """Model should work with any embedding_dim (e.g. XLM-R-large = 1024)."""
    m = DREAMModel(embedding_dim=1024, num_languages=10)
    x = torch.randn(2, 1024)
    meaning, language, logits = m(x)
    assert meaning.shape == (2, 1024)
    assert logits.shape  == (2, 10)


def test_extra_repr(model):
    r = repr(model)
    assert "embedding_dim=768" in r
    assert "num_languages=8"   in r
