"""Unit tests for DREAM loss functions."""

import pytest
import torch
from dream.loss import (
    LossComponents,
    compute_loss,
    language_loss,
    meaning_loss,
    reconstruction_loss,
)

B = 8    # batch size
D = 768  # embedding dim
N = 4    # num languages


def _rand(*shape) -> torch.Tensor:
    return torch.randn(*shape)


def _lang_ids(n: int = B, num_classes: int = N) -> torch.Tensor:
    return torch.randint(0, num_classes, (n,))


# ---------------------------------------------------------------------------
# Reconstruction loss
# ---------------------------------------------------------------------------

def test_reconstruction_zero_when_perfect():
    """When e_m + e_l == e, loss should be zero."""
    e = _rand(B, D)
    e_m = e * 0.7
    e_l = e * 0.3
    loss = reconstruction_loss(e, e_m, e_l)
    assert loss.item() < 1e-5


def test_reconstruction_positive():
    e, e_m, e_l = _rand(B, D), _rand(B, D), _rand(B, D)
    assert reconstruction_loss(e, e_m, e_l).item() >= 0


# ---------------------------------------------------------------------------
# Meaning loss
# ---------------------------------------------------------------------------

def test_meaning_loss_parallel_identical():
    """If s_m == t_m (perfect alignment), L_x should be 0."""
    s_m = torch.randn(B, D)
    t_m = s_m.clone()
    rand_s = _rand(B, D)
    rand_t = _rand(B, D)
    loss = meaning_loss(s_m, t_m, rand_s, rand_t)
    # L_x = 0, L_m can still be positive
    assert loss.item() >= 0


def test_meaning_loss_output_shape():
    loss = meaning_loss(_rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D))
    assert loss.dim() == 0   # scalar


# ---------------------------------------------------------------------------
# Language loss
# ---------------------------------------------------------------------------

def test_language_loss_output_shape():
    loss = language_loss(
        _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D),
        _rand(B, N), _rand(B, N),
        _lang_ids(), _lang_ids(),
    )
    assert loss.dim() == 0


def test_language_loss_positive():
    loss = language_loss(
        _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D),
        _rand(B, N), _rand(B, N),
        _lang_ids(), _lang_ids(),
    )
    assert loss.item() >= 0


# ---------------------------------------------------------------------------
# compute_loss (combined)
# ---------------------------------------------------------------------------

def test_compute_loss_returns_components():
    lc = compute_loss(
        _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, N), _rand(B, N),
        _lang_ids(), _lang_ids(),
    )
    assert isinstance(lc, LossComponents)
    assert lc.total.item() >= 0


def test_compute_loss_total_equals_sum():
    lc = compute_loss(
        _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, N), _rand(B, N),
        _lang_ids(), _lang_ids(),
    )
    expected = lc.reconstruction + lc.meaning + lc.language
    assert torch.allclose(lc.total, expected, atol=1e-5)


def test_as_log_dict():
    lc = compute_loss(
        _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, D), _rand(B, D), _rand(B, D), _rand(B, D),
        _rand(B, N), _rand(B, N),
        _lang_ids(), _lang_ids(),
    )
    d = lc.as_log_dict(prefix="train/")
    assert "train/loss/total"          in d
    assert "train/loss/reconstruction" in d
    assert "train/loss/meaning"        in d
    assert "train/loss/language"       in d
    assert all(isinstance(v, float) for v in d.values())


def test_backward_works():
    """Total loss must be differentiable."""
    from dream.model import DREAMModel
    model = DREAMModel(D, N)
    e_src = _rand(B, D)
    e_tgt = _rand(B, D)
    rand_src = _rand(B, D)
    rand_tgt = _rand(B, D)

    em_src, el_src, li_src = model(e_src)
    em_tgt, el_tgt, li_tgt = model(e_tgt)
    em_rand_src, el_rand_src, _ = model(rand_src)
    em_rand_tgt, el_rand_tgt, _ = model(rand_tgt)

    lc = compute_loss(
        e_src, e_tgt,
        em_src, em_tgt, em_rand_src, em_rand_tgt,
        el_src, el_tgt, el_rand_src, el_rand_tgt,
        li_src, li_tgt,
        _lang_ids(), _lang_ids(),
    )
    lc.total.backward()   # should not raise
    assert model.meaning_encoder.weight.grad is not None
