"""
utils/quantum_metrics.py -- Quantum circuit expressibility and entanglement metrics.

Implements:
  - Expressibility (Meyer-Wallach approximation): how well the ansatz covers
    the Hilbert space. Higher = more expressive.
  - Entanglement capability (Meyer-Wallach Q measure): mean entanglement
    produced by random parameter settings.
  - Circuit statistics: gate counts, depth, T-count from PennyLane tape.

References:
  Sim et al. (2019) "Expressibility and Entangling Capability of Parameterized
  Quantum Circuits for Hybrid Quantum-Classical Algorithms"
  arXiv:1905.10876
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ── Statevector helper ────────────────────────────────────────────────────────

def _random_statevector(circuit_fn: Callable, n_params: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample a random parameter vector and return the circuit statevector."""
    import pennylane as qml
    params = rng.uniform(-np.pi, np.pi, n_params)
    return np.array(circuit_fn(params), dtype=complex)


# ── Expressibility ────────────────────────────────────────────────────────────

def expressibility(
    circuit_fn: Callable,
    n_qubits: int,
    n_params: int,
    n_samples: int = 2000,
    rng_seed: int = 42,
) -> float:
    """
    Meyer-Wallach expressibility approximation.

    Samples 2000 pairs of random parameter vectors, computes the mean
    pairwise state fidelity |<psi_1|psi_2>|^2, then returns
        expr = 1 - mean_fidelity
    Higher expr (closer to 1) means the circuit covers the Hilbert space better.

    Parameters
    ----------
    circuit_fn : function(params) -> statevector (complex ndarray, len=2**n_qubits)
    n_qubits   : number of qubits
    n_params   : total number of variational parameters
    n_samples  : number of random parameter pairs to sample
    """
    rng = np.random.RandomState(rng_seed)
    fidelities = []

    for _ in range(n_samples):
        sv1 = _random_statevector(circuit_fn, n_params, rng)
        sv2 = _random_statevector(circuit_fn, n_params, rng)
        fidelity = abs(np.dot(sv1.conj(), sv2)) ** 2
        fidelities.append(float(fidelity))

    mean_fid = float(np.mean(fidelities))
    # Haar random expected fidelity = 1 / (2^n_qubits + 1)
    haar_fid = 1.0 / (2 ** n_qubits + 1)
    expr = 1.0 - mean_fid
    expr_normalized = expr / (1.0 - haar_fid + 1e-12)  # normalized vs Haar
    return float(np.clip(expr_normalized, 0.0, 1.0))


# ── Entanglement capability ───────────────────────────────────────────────────

def _meyer_wallach_Q(statevector: np.ndarray, n_qubits: int) -> float:
    """
    Compute the Meyer-Wallach entanglement measure Q for a single statevector.
    Q = (4/n) * sum_k [ 1 - Tr(rho_k^2) ]
    where rho_k is the reduced density matrix for qubit k.
    """
    dim = 2 ** n_qubits
    sv = statevector.reshape(-1)
    assert len(sv) == dim

    total = 0.0
    for k in range(n_qubits):
        # Reshape to trace out all qubits except k
        sv_mat = sv.reshape([2] * n_qubits)
        # Move qubit k to first axis
        sv_mat = np.moveaxis(sv_mat, k, 0)
        sv_flat = sv_mat.reshape(2, -1)
        rho = sv_flat @ sv_flat.conj().T
        purity = np.real(np.trace(rho @ rho))
        total += 1.0 - purity

    return float(4.0 / n_qubits * total)


def entanglement_capability(
    circuit_fn: Callable,
    n_qubits: int,
    n_params: int,
    n_samples: int = 500,
    rng_seed: int = 42,
) -> float:
    """
    Meyer-Wallach entanglement capability: mean Q over random parameter settings.
    Q=0 means no entanglement; Q=1 means maximally entangled.
    """
    rng = np.random.RandomState(rng_seed)
    qs = []
    for _ in range(n_samples):
        sv = _random_statevector(circuit_fn, n_params, rng)
        sv = sv / (np.linalg.norm(sv) + 1e-12)
        qs.append(_meyer_wallach_Q(sv, n_qubits))
    return float(np.mean(qs))


# ── Circuit statistics ────────────────────────────────────────────────────────

def circuit_stats(circuit_fn: Callable, sample_params: np.ndarray) -> dict:
    """
    Parse a PennyLane tape to count gate types, depth, and parameters.
    Returns a dict with gate_counts, circuit_depth, n_params, t_count.
    """
    import pennylane as qml

    # Use qml.specs to get circuit info
    try:
        specs = qml.specs(circuit_fn)(sample_params)
        gate_sizes = specs.get("gate_sizes", {})
        gate_types = specs.get("gate_types", {})
        depth = specs.get("depth", 0)
        n_params_circuit = specs.get("num_trainable_params", len(sample_params))
        return {
            "gate_types": {str(k): int(v) for k, v in gate_types.items()},
            "gate_sizes": {str(k): int(v) for k, v in gate_sizes.items()},
            "circuit_depth": int(depth),
            "n_params": int(n_params_circuit),
            "t_count": int(gate_types.get("T", 0) + gate_types.get("RZ", 0)),
        }
    except Exception as e:
        return {"error": str(e), "n_params": len(sample_params)}


# ── Save / load ───────────────────────────────────────────────────────────────

def save_quantum_metrics(metrics: dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(metrics, indent=2, default=str))
    print(f"  [quantum_metrics] Saved: {out_path}")


def load_cached_metrics(out_path: str) -> Optional[dict]:
    p = Path(out_path)
    if p.exists():
        return json.loads(p.read_text())
    return None


# ── Top-level runner ──────────────────────────────────────────────────────────

def compute_and_save_quantum_metrics(
    circuit_fn: Callable,
    n_qubits: int,
    n_params: int,
    out_path: str,
    fast_mode: bool = False,
    n_expr_samples: int = 2000,
    n_ent_samples: int = 500,
) -> dict:
    """
    Compute expressibility + entanglement + circuit stats; save to JSON.
    If fast_mode and cache exists, load cache instead of recomputing.
    """
    if fast_mode:
        cached = load_cached_metrics(out_path)
        if cached is not None:
            print(f"  [quantum_metrics] Loaded cached metrics from {out_path}")
            return cached

    print(f"  [quantum_metrics] Computing expressibility ({n_expr_samples} samples)...")
    expr = expressibility(circuit_fn, n_qubits, n_params, n_samples=n_expr_samples)

    print(f"  [quantum_metrics] Computing entanglement ({n_ent_samples} samples)...")
    ent = entanglement_capability(circuit_fn, n_qubits, n_params, n_samples=n_ent_samples)

    sample_p = np.zeros(n_params)
    stats_dict = circuit_stats(circuit_fn, sample_p)

    metrics = {
        "n_qubits": n_qubits,
        "n_params": n_params,
        "expressibility": expr,
        "entanglement_capability": ent,
        "circuit_stats": stats_dict,
    }
    save_quantum_metrics(metrics, out_path)
    return metrics
