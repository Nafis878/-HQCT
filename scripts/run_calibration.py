"""
scripts/run_calibration.py -- Calibration analysis across all datasets.

Reads per-sample fold probabilities saved by the CV scripts
({prefix}_fold_probas.npz) + the matching y_full arrays, then computes ECE/MCE
(+ Platt / isotonic) per model and saves reliability diagrams.

Outputs:
  results/calibration_metrics.json          (keyed by dataset -> model -> metrics)
  results/figures/calibration_{dataset}.pdf

Run: python scripts/run_calibration.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

# dataset -> (fold_probas npz, y_full npy)
DATASETS = {
    "CKD":       ("ckd_fold_probas.npz",       "y_full.npy"),
    "FHS":       ("fhs_fold_probas.npz",       "fhs_y_full.npy"),
    "PIMA":      ("pima_fold_probas.npz",      "pima_y_full.npy"),
    "Cleveland": ("cleveland_fold_probas.npz", "cleveland_y_full.npy"),
}

# proba-key -> display name (matches calibration.py MODEL_COLORS)
KEY_TO_NAME = {
    "xgb": "XGBoost", "tab": "TabTransformer", "lgb": "LightGBM",
    "mlp": "MLP", "hqct": "HybridQT", "qsvm": "QSVM",
}


def main():
    from utils.calibration import compute_calibration_metrics, reliability_diagram, save_calibration_metrics

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CALIBRATION ANALYSIS (all datasets)")
    print("=" * 60)

    all_metrics = {}
    for ds, (npz_name, y_name) in DATASETS.items():
        npz_path = RESULTS_DIR / npz_name
        y_path = DATA_DIR / y_name
        if not npz_path.exists() or not y_path.exists():
            print(f"  [skip] {ds}: missing {npz_name} or {y_name}")
            continue

        probas = np.load(npz_path)
        y_full = np.load(y_path)

        models_probas = {}
        for key in probas.files:
            name = KEY_TO_NAME.get(key, key)
            p = probas[key]
            # Skip arrays that were never populated (all -1 / all 0)
            if p.shape[0] != y_full.shape[0] or np.allclose(p, 0):
                continue
            models_probas[name] = (y_full.astype(int), p.astype(float))

        if not models_probas:
            print(f"  [skip] {ds}: no usable probability arrays")
            continue

        print(f"\n[{ds}] models: {list(models_probas.keys())}")
        metrics = compute_calibration_metrics(models_probas, n_bins=10)
        all_metrics[ds] = metrics
        reliability_diagram(models_probas, str(FIG_DIR / f"calibration_{ds.lower()}.pdf"),
                            n_bins=10, dataset_name=ds)
        for m, vals in metrics.items():
            print(f"    {m:16s} ECE={vals['ece_raw']:.4f}  MCE={vals['mce_raw']:.4f}")

    save_calibration_metrics(all_metrics, str(RESULTS_DIR / "calibration_metrics.json"))
    print("\nCalibration analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
