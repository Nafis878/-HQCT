"""
scripts/run_explainability.py -- SHAP + quantum feature attribution.

For each dataset:
  - Fit XGBoost on the train split and produce a TreeExplainer SHAP beeswarm +
    mean|SHAP| importance CSV on the held-out test split (medical-AI requirement).
For the flagship datasets (CKD, PIMA):
  - Train a HybridQT (adaptive circuit) and compute quantum feature attribution
    (gradient of the model output w.r.t. each input feature) — a quantum-specific
    importance distinct from SHAP.

Outputs:
  results/figures/shap_summary_XGBoost_{DATASET}.png
  results/shap_importance_XGBoost_{DATASET}.csv
  results/quantum_feature_attribution_{DATASET}.csv

Run: python scripts/run_explainability.py [--epochs 30] [--quantum-datasets CKD,PIMA]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"
SEED = 42


def _feature_names(dataset: str, n_feat: int) -> list:
    """Best-effort feature names per dataset; fall back to generic."""
    try:
        if dataset == "CKD":
            from preprocessing import NUMERIC_COLS, CAT_COLS
            return list(NUMERIC_COLS) + list(CAT_COLS)
        if dataset == "PIMA":
            from pima_preprocessing import FEATURE_COLS
            return list(FEATURE_COLS)
        if dataset == "Cleveland":
            from cleveland_preprocessing import FEATURE_COLS
            return list(FEATURE_COLS)
        if dataset == "FHS":
            import fhs_preprocessing as fp
            for attr in ("CONTINUOUS_COLS", "BINARY_COLS", "ORDINAL_COLS",
                         "NUMERIC_COLS", "CAT_COLS", "FEATURE_COLS"):
                pass
            names = []
            for attr in ("CONTINUOUS_COLS", "BINARY_COLS"):
                names += list(getattr(fp, attr, []))
            if len(names) == n_feat:
                return names
    except Exception:
        pass
    return [f"feat_{i}" for i in range(n_feat)]


# dataset -> (train X, train y, test X, prefix for split files)
DATASETS = {
    "CKD":       ("X_train.npy", "y_train.npy", "X_test.npy"),
    "FHS":       ("fhs_X_train.npy", "fhs_y_train.npy", "fhs_X_test.npy"),
    "PIMA":      ("pima_X_train.npy", "pima_y_train.npy", "pima_X_test.npy"),
    "Cleveland": ("cleveland_X_train.npy", "cleveland_y_train.npy", "cleveland_X_test.npy"),
}


def run_xgb_shap(dataset: str):
    from xgboost import XGBClassifier
    from utils.explainability import shap_xgboost

    xtr_n, ytr_n, xte_n = DATASETS[dataset]
    xtr, ytr, xte = DATA_DIR / xtr_n, DATA_DIR / ytr_n, DATA_DIR / xte_n
    if not (xtr.exists() and ytr.exists() and xte.exists()):
        print(f"  [skip] {dataset}: split files missing")
        return

    X_train, y_train, X_test = np.load(xtr), np.load(ytr), np.load(xte)
    names = _feature_names(dataset, X_train.shape[1])

    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
        random_state=SEED,
    )
    clf.fit(X_train, y_train)
    print(f"\n[{dataset}] XGBoost SHAP on {len(X_test)} held-out samples...")
    shap_xgboost(clf, X_test, names, out_dir=str(FIG_DIR), dataset_label=dataset)


def run_quantum_attr(dataset: str, epochs: int):
    import torch
    from models.hybrid_quantum_transformer import HybridTabTransformer
    from models.adaptive_vqc import AdaptiveVQCSelector
    from utils.explainability import quantum_feature_attribution

    xtr_n, ytr_n, xte_n = DATASETS[dataset]
    xtr, ytr, xte = DATA_DIR / xtr_n, DATA_DIR / ytr_n, DATA_DIR / xte_n
    if not (xtr.exists() and ytr.exists() and xte.exists()):
        print(f"  [skip quantum-attr] {dataset}: split files missing")
        return

    X_train, y_train, X_test = np.load(xtr), np.load(ytr), np.load(xte)
    names = _feature_names(dataset, X_train.shape[1])

    cfg, label = AdaptiveVQCSelector.make_config(len(y_train))
    torch.manual_seed(SEED)
    model = HybridTabTransformer(
        n_features=X_train.shape[1], d_model=32, n_heads=4, n_layers=2,
        dropout=0.1, qc_cfg=cfg,
    )

    # Light training so attributions reflect a fitted model
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    print(f"\n[{dataset}] training HybridQT ({label}) {epochs} epochs for attribution...")
    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        logits = model(Xt).squeeze(-1)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    print(f"  quantum feature attribution on {len(X_test)} samples...")
    quantum_feature_attribution(model, X_test, names, out_dir=str(RESULTS_DIR),
                                dataset_label=dataset, n_samples=min(100, len(X_test)))


def main():
    ap = argparse.ArgumentParser(description="SHAP + quantum feature attribution")
    ap.add_argument("--epochs", type=int, default=30,
                    help="Epochs for HybridQT attribution training (default 30)")
    ap.add_argument("--quantum-datasets", type=str, default="CKD,PIMA",
                    help="Comma-separated datasets for quantum attribution")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("EXPLAINABILITY: SHAP (all) + quantum attribution (flagship)")
    print("=" * 60)

    for ds in DATASETS:
        try:
            run_xgb_shap(ds)
        except Exception as exc:
            print(f"  [WARNING] {ds} XGBoost SHAP failed: {exc}")

    q_sets = [s.strip() for s in args.quantum_datasets.split(",") if s.strip()]
    for ds in q_sets:
        if ds not in DATASETS:
            continue
        try:
            run_quantum_attr(ds, args.epochs)
        except Exception as exc:
            print(f"  [WARNING] {ds} quantum attribution failed: {exc}")

    print("\nExplainability analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
