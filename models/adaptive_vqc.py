"""
models/adaptive_vqc.py -- Data-adaptive variational quantum circuit selection.

Motivation: a 6-qubit / 3-layer HEA has 36 variational parameters. On small
datasets (e.g. CKD n=400, ~360 training samples per fold) this over-parameterised
circuit cannot be reliably trained, hurting the hybrid model. Conversely, large
datasets benefit from the more expressive circuit. AdaptiveVQCSelector picks the
circuit complexity from the per-fold training-set size, so the quantum capacity
scales with the available data.

Thresholds (n_train = number of training samples actually fed to the VQC):
    n < 500     -> 4 qubits, 2 layers  (16 params)   "4q-2L"
    500-2000    -> 6 qubits, 2 layers  (24 params)   "6q-2L"
    n >= 2000   -> 6 qubits, 3 layers  (36 params)   "6q-3L"

select() returns kwargs compatible with QuantumCircuitConfig.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VQCChoice:
    n_qubits: int
    n_vqc_layers: int
    label: str

    @property
    def n_params(self) -> int:
        return 2 * self.n_qubits * self.n_vqc_layers


class AdaptiveVQCSelector:
    """Selects a VQC configuration based on training-set size."""

    SMALL = 500
    MEDIUM = 2000

    @staticmethod
    def select(n_train: int) -> dict:
        """
        Return circuit kwargs for the given training-set size.

        Keys: n_qubits, n_vqc_layers, label (all consumable by
        QuantumCircuitConfig(**{k: v for k, v in select(n) if k != 'label'})).
        """
        if n_train < AdaptiveVQCSelector.SMALL:
            choice = VQCChoice(4, 2, "4q-2L")
        elif n_train < AdaptiveVQCSelector.MEDIUM:
            choice = VQCChoice(6, 2, "6q-2L")
        else:
            choice = VQCChoice(6, 3, "6q-3L")
        return {
            "n_qubits": choice.n_qubits,
            "n_vqc_layers": choice.n_vqc_layers,
            "label": choice.label,
            "n_params": choice.n_params,
        }

    @staticmethod
    def make_config(n_train: int, data_reuploading: bool = True,
                    entanglement: str = "ring"):
        """
        Build a QuantumCircuitConfig sized for n_train. Imported lazily to avoid
        a hard dependency when only select() is needed.
        """
        from models.hybrid_quantum_transformer import QuantumCircuitConfig
        sel = AdaptiveVQCSelector.select(n_train)
        cfg = QuantumCircuitConfig(
            n_qubits=sel["n_qubits"],
            n_vqc_layers=sel["n_vqc_layers"],
            data_reuploading=data_reuploading,
            entanglement=entanglement,
        )
        return cfg, sel["label"]


if __name__ == "__main__":
    for n in [200, 360, 700, 800, 1500, 3800]:
        sel = AdaptiveVQCSelector.select(n)
        print(f"n_train={n:5d} -> {sel['label']:6s} "
              f"({sel['n_qubits']}q, {sel['n_vqc_layers']}L, {sel['n_params']} params)")
