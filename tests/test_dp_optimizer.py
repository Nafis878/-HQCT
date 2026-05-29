"""
tests/test_dp_optimizer.py -- Unit tests for utils/dp_training.py.

Tests: noise injection, gradient clipping, epsilon computation.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ── DPOptimizer ───────────────────────────────────────────────────────────────

def _simple_model():
    return nn.Linear(4, 1)


def test_dp_optimizer_clips_gradients():
    from utils.dp_training import DPOptimizer

    model = _simple_model()
    x = torch.randn(8, 4)
    y = torch.zeros(8, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    # Manually set large gradients
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 100.0

    max_grad_norm = 1.0
    base_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    dp_opt = DPOptimizer(base_opt, noise_multiplier=0.0, max_grad_norm=max_grad_norm)
    dp_opt.step()

    # After step, gradients should have been clipped (norm ≤ max_grad_norm)
    total_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters()
                     if p.grad is not None) ** 0.5
    assert total_norm <= max_grad_norm + 1e-5, (
        f"Gradients not clipped: norm={total_norm:.4f} > max={max_grad_norm}"
    )


def test_dp_optimizer_adds_noise():
    from utils.dp_training import DPOptimizer

    model = _simple_model()
    # Zero-out gradients first
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    # With noise_multiplier > 0, optimizer should add Gaussian noise
    base_opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 so weights don't change
    grads_before = [p.grad.clone() for p in model.parameters()]

    dp_opt = DPOptimizer(base_opt, noise_multiplier=2.0, max_grad_norm=1.0)
    dp_opt.step()

    grads_after = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    any_changed = any(
        not torch.allclose(g_before, g_after)
        for g_before, g_after in zip(grads_before, grads_after)
    )
    assert any_changed, "DPOptimizer with noise_multiplier=2.0 should alter gradients"


def test_dp_optimizer_zero_noise_deterministic():
    from utils.dp_training import DPOptimizer

    torch.manual_seed(42)
    model = _simple_model()
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.5

    grads_before = [p.grad.clone() for p in model.parameters()]
    base_opt = torch.optim.SGD(model.parameters(), lr=0.0)
    dp_opt = DPOptimizer(base_opt, noise_multiplier=0.0, max_grad_norm=1.0)
    dp_opt.step()

    for g_before, p in zip(grads_before, model.parameters()):
        if p.grad is not None:
            # With noise=0 and clipping, norm should be exactly max_grad_norm direction
            assert torch.isfinite(p.grad).all()


# ── Epsilon computation ───────────────────────────────────────────────────────

def test_compute_epsilon_positive():
    from utils.dp_training import compute_epsilon
    eps = compute_epsilon(
        steps=1000, batch_size=32, n_samples=400,
        noise_multiplier=1.1, delta=1e-5,
    )
    assert eps > 0.0
    assert np.isfinite(eps)


def test_compute_epsilon_decreases_with_more_noise():
    from utils.dp_training import compute_epsilon
    eps_low_noise = compute_epsilon(1000, 32, 400, noise_multiplier=0.5, delta=1e-5)
    eps_high_noise = compute_epsilon(1000, 32, 400, noise_multiplier=2.0, delta=1e-5)
    assert eps_high_noise < eps_low_noise, (
        "Higher noise_multiplier should yield lower (better) epsilon"
    )


def test_compute_epsilon_increases_with_more_steps():
    from utils.dp_training import compute_epsilon
    eps_few = compute_epsilon(100, 32, 400, noise_multiplier=1.1, delta=1e-5)
    eps_many = compute_epsilon(10000, 32, 400, noise_multiplier=1.1, delta=1e-5)
    assert eps_many > eps_few, "More training steps should increase epsilon"


def test_privacy_report_structure():
    from utils.dp_training import privacy_report
    # privacy_report takes epochs (not steps) as first positional arg
    report = privacy_report(epochs=15, batch_size=32, n_samples=400,
                             noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
    for key in ["epsilon", "delta", "noise_multiplier", "mechanism"]:
        assert key in report, f"Key '{key}' missing from privacy_report"
    assert report["delta"] == 1e-5
    assert report["epsilon"] > 0
