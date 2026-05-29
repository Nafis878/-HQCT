"""
config.py -- Central experiment configuration for the HQCT Q1 pipeline.

Usage:
    from config import ExperimentConfig, load_config, save_config
    cfg = load_config("config.yaml")   # or ExperimentConfig() for defaults
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    random_seed: int = 42
    test_size: float = 0.2
    n_cv_folds: int = 10

    # ── Transformer ───────────────────────────────────────────────────────────
    d_model: int = 32
    n_heads: int = 4
    n_transformer_layers: int = 2
    dropout: float = 0.1

    # ── Quantum circuit ───────────────────────────────────────────────────────
    n_qubits: int = 6           # upgraded: 6-qubit HEA (was 4)
    n_vqc_layers: int = 3       # upgraded: 3 layers (was 2)
    data_reuploading: bool = True
    entanglement: str = "ring"  # "ring" | "full"

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 50
    lr: float = 5e-4
    batch_size: int = 32
    patience: int = 10

    # ── Differential Privacy ──────────────────────────────────────────────────
    dp_noise_multiplier: float = 1.1
    dp_max_grad_norm: float = 1.0
    dp_delta: float = 1e-5

    # ── Integrity ─────────────────────────────────────────────────────────────
    enable_sha256: bool = True

    # ── Performance flags ─────────────────────────────────────────────────────
    fast_mode: bool = False      # skip expensive quantum metrics; use cache
    skip_qsvm: bool = False
    dp_train: bool = False
    run_ablation: bool = False
    run_explainability: bool = True
    run_calibration: bool = True
    run_quantum_metrics: bool = True


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """Load config from JSON or YAML; fall back to defaults."""
    if path is None:
        return ExperimentConfig()
    p = Path(path)
    if not p.exists():
        return ExperimentConfig()
    text = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            data = yaml.safe_load(text) or {}
        except ImportError:
            import json as _json
            data = _json.loads(text)
    else:
        import json as _json
        data = _json.loads(text)
    return ExperimentConfig(**{k: v for k, v in data.items()
                               if k in ExperimentConfig.__dataclass_fields__})


def save_config(cfg: ExperimentConfig, path: str) -> None:
    """Persist config to JSON."""
    Path(path).write_text(json.dumps(asdict(cfg), indent=2))


DEFAULT_CFG = ExperimentConfig()
