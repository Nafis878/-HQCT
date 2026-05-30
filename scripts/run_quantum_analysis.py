"""
scripts/run_quantum_analysis.py -- Run quantum circuit analyses end-to-end.

Computes and saves:
  - Expressibility + entanglement capability + circuit stats for the three
    adaptive configs (4q-2L, 6q-2L, 6q-3L)  -> results/quantum_circuit_metrics.json
  - Kernel Target Alignment (quantum vs RBF) on CKD              -> results/quantum_advantage.json
  - Barren-plateau gradient-variance analysis (6q-3L HybridQT)  -> results/figures/barren_plateau.pdf

Balanced budget: expressibility 1000 samples, entanglement 300, KTA on 100 samples.

Run: python scripts/run_quantum_analysis.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

# Adaptive configs to characterise (label -> (n_qubits, n_layers))
CONFIGS = [("4q-2L", 4, 2), ("6q-2L", 6, 2), ("6q-3L", 6, 3)]


def _build_state_circuit(n_qubits: int, n_layers: int):
    """
    Return circuit_fn(params) -> statevector for the HEA ansatz with zero data
    input (isolating the variational parameter-space coverage). params is a flat
    vector of length 2*n_qubits*n_layers, reshaped to (n_layers, 2, n_qubits) to
    match build_hea_layer's weight layout.
    """
    import pennylane as qml

    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(params):
        w = np.asarray(params).reshape(n_layers, 2, n_qubits)
        for l in range(n_layers):
            # zero data input => RY(0)=I (data-encoding is identity here)
            for i in range(n_qubits):
                qml.RY(w[l, 0, i], wires=i)
                qml.RZ(w[l, 1, i], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i + 1) % n_qubits])
        return qml.state()

    return circuit


def run_circuit_metrics(fast: bool) -> dict:
    from utils.quantum_metrics import (
        expressibility, entanglement_capability, circuit_stats,
    )
    n_expr = 1000 if not fast else 200
    n_ent = 300 if not fast else 100

    out = {}
    for label, n_q, n_l in CONFIGS:
        n_params = 2 * n_q * n_l
        print(f"\n[circuit metrics] {label}  ({n_q} qubits, {n_l} layers, {n_params} params)")
        circ = _build_state_circuit(n_q, n_l)
        print(f"  expressibility ({n_expr} samples)...", flush=True)
        expr = expressibility(circ, n_q, n_params, n_samples=n_expr)
        print(f"  entanglement ({n_ent} samples)...", flush=True)
        ent = entanglement_capability(circ, n_q, n_params, n_samples=n_ent)
        stats = circuit_stats(circ, np.zeros(n_params))
        out[label] = {
            "n_qubits": n_q, "n_layers": n_l, "n_params": n_params,
            "expressibility": round(expr, 6),
            "entanglement_capability": round(ent, 6),
            "circuit_stats": stats,
        }
        print(f"  -> expr={expr:.4f}  ent={ent:.4f}")
    return out


def run_kta() -> dict:
    """KTA of the quantum (RY+CNOT) kernel vs RBF on CKD (PCA to 4 dims)."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    from models.baselines import build_quantum_kernel, compute_kernel_matrix
    from utils.quantum_advantage import kernel_target_alignment

    xf = DATA_DIR / "X_full.npy"
    yf = DATA_DIR / "y_full.npy"
    if not xf.exists():
        return {"error": "CKD X_full.npy not found; run preprocessing.py first"}

    X = np.load(xf)
    y = np.load(yf)

    # PCA to 4 dims (QSVM convention) then scale to [-pi, pi]
    Xp = PCA(n_components=4, random_state=42).fit_transform(X)
    Xp = MinMaxScaler((-np.pi, np.pi)).fit_transform(Xp).astype(float)

    kernel = build_quantum_kernel()
    qkfn = lambda Xs: compute_kernel_matrix(kernel, Xs, Xs, verbose=False)

    print("\n[KTA] computing quantum vs RBF alignment on CKD (100 samples)...", flush=True)
    res = kernel_target_alignment(Xp, y, qkfn, classical_kernel="rbf", n_samples=100)
    print(f"  KTA quantum-vs-ideal={res.get('kta_quantum_vs_ideal')}, "
          f"rbf-vs-ideal={res.get('kta_rbf_vs_ideal')}")
    return res


def run_barren_plateau() -> dict:
    """Gradient-variance analysis on a 6q-3L HybridQT (untrained)."""
    import torch
    from models.hybrid_quantum_transformer import HybridTabTransformer, QuantumCircuitConfig
    from utils.quantum_advantage import barren_plateau_check

    xf = DATA_DIR / "X_full.npy"
    n_feat = int(np.load(xf).shape[1]) if xf.exists() else 24

    torch.manual_seed(42)
    cfg = QuantumCircuitConfig(n_qubits=6, n_vqc_layers=3, data_reuploading=True)
    model = HybridTabTransformer(
        n_features=n_feat, d_model=32, n_heads=4, n_layers=2, dropout=0.0, qc_cfg=cfg,
    )
    print("\n[barren plateau] gradient variance over random inputs (6q-3L)...", flush=True)
    res = barren_plateau_check(model, n_samples=200,
                               out_path=str(FIG_DIR / "barren_plateau.pdf"))
    res["config"] = "6q-3L"
    print(f"  mean gradient variance={res.get('mean_gradient_variance'):.3e}  "
          f"plateau_detected={res.get('plateau_detected')}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Run quantum circuit analyses")
    ap.add_argument("--fast", action="store_true", help="Fewer samples (quick check)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QUANTUM CIRCUIT ANALYSIS")
    print("=" * 60)

    circ_metrics = run_circuit_metrics(args.fast)
    from utils.quantum_metrics import save_quantum_metrics
    save_quantum_metrics(circ_metrics, str(RESULTS_DIR / "quantum_circuit_metrics.json"))

    advantage = {}
    try:
        advantage["kta_ckd"] = run_kta()
    except Exception as exc:
        print(f"  [WARNING] KTA failed: {exc}")
        advantage["kta_ckd"] = {"error": str(exc)}
    try:
        advantage["barren_plateau_6q3L"] = run_barren_plateau()
    except Exception as exc:
        print(f"  [WARNING] Barren plateau failed: {exc}")
        advantage["barren_plateau_6q3L"] = {"error": str(exc)}

    from utils.quantum_advantage import save_quantum_advantage
    save_quantum_advantage(advantage, str(RESULTS_DIR / "quantum_advantage.json"))

    print("\nQuantum analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
