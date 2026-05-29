"""
tests/test_quantum_layer.py -- Unit tests for the VQC and HybridTabTransformer.

Tests: VQC output shape, gradient flow, param count, QuantumCircuitConfig,
DEFAULT_QC_CFG vs LEGACY_QC_CFG behaviour.
"""

import pytest
import numpy as np
import torch


@pytest.fixture(scope="module")
def n_features():
    return 24  # CKD feature count


@pytest.fixture(scope="module")
def default_cfg():
    from models.hybrid_quantum_transformer import DEFAULT_QC_CFG
    return DEFAULT_QC_CFG


@pytest.fixture(scope="module")
def legacy_cfg():
    from models.hybrid_quantum_transformer import LEGACY_QC_CFG
    return LEGACY_QC_CFG


def test_quantum_circuit_config_n_params(default_cfg, legacy_cfg):
    assert default_cfg.n_params == 2 * 6 * 3  # RY + RZ, 6 qubits, 3 layers
    assert legacy_cfg.n_params == 2 * 4 * 2   # RY + RZ, 4 qubits, 2 layers


def test_quantum_circuit_config_weight_shapes(default_cfg):
    ws = default_cfg.weight_shapes()
    assert "weights" in ws
    assert ws["weights"] == (default_cfg.n_vqc_layers, 2, default_cfg.n_qubits)


def test_hybrid_model_output_shape(n_features, default_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer
    model = HybridTabTransformer(
        n_features=n_features, d_model=32, n_heads=4, n_layers=2,
        dropout=0.0, qc_cfg=default_cfg,
    )
    x = torch.randn(4, n_features)
    out = model(x)
    assert out.shape == (4, 1), f"Expected (4,1), got {out.shape}"


def test_hybrid_model_gradient_flow(n_features, default_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer
    model = HybridTabTransformer(
        n_features=n_features, d_model=32, n_heads=4, n_layers=2,
        dropout=0.0, qc_cfg=default_cfg,
    )
    x = torch.randn(2, n_features, requires_grad=False)
    y = torch.zeros(2, 1)
    loss = torch.nn.BCEWithLogitsLoss()(model(x), y)
    loss.backward()
    # Check at least one parameter has a gradient
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "No gradients found — backprop through VQC failed"


def test_hybrid_model_param_count(n_features, default_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer
    model = HybridTabTransformer(
        n_features=n_features, d_model=32, n_heads=4, n_layers=2,
        dropout=0.0, qc_cfg=default_cfg,
    )
    n_transformer_layers = 2  # matches n_layers passed above
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts = model.param_counts()
    assert counts["total"] == total
    # Each transformer layer contains one VQC; total quantum params scales with n_layers
    assert counts["quantum"] == n_transformer_layers * default_cfg.n_params
    assert counts["classical"] == total - counts["quantum"]


def test_legacy_vs_default_param_count_differs(n_features, default_cfg, legacy_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer
    m_def = HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                  n_layers=2, dropout=0.0, qc_cfg=default_cfg)
    m_leg = HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                  n_layers=2, dropout=0.0, qc_cfg=legacy_cfg)
    assert m_def.param_counts()["quantum"] != m_leg.param_counts()["quantum"]


def test_predict_proba_range(n_features, default_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer
    model = HybridTabTransformer(
        n_features=n_features, d_model=32, n_heads=4, n_layers=2,
        dropout=0.0, qc_cfg=default_cfg,
    )
    model.eval()
    x = torch.randn(8, n_features)
    proba = model.predict_proba(x)
    assert proba.shape == (8,)
    assert float(proba.min()) >= 0.0
    assert float(proba.max()) <= 1.0


def test_hybrid_model_deterministic_with_seed(n_features, default_cfg):
    from models.hybrid_quantum_transformer import HybridTabTransformer

    def _make_out():
        torch.manual_seed(42)
        model = HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                      n_layers=2, dropout=0.0, qc_cfg=default_cfg)
        model.eval()
        with torch.no_grad():
            return model(torch.ones(1, n_features)).item()

    assert abs(_make_out() - _make_out()) < 1e-6
