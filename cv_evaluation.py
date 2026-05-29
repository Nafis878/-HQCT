"""
cv_evaluation.py -- 10-fold stratified cross-validation for all CKD models.

SMOTE is applied INSIDE each training fold only (no data leakage).
Q1 upgrade: expanded metrics (MCC, kappa, brier, AUC-PR, inference time),
statistical tests (Wilcoxon, Friedman, bootstrap CIs), McNemar contingency
detail, and ckd_fold_probas.npz for ROC plotting.

Usage:
  python cv_evaluation.py                   # All models (~45 min on CPU)
  python cv_evaluation.py --skip-qsvm       # Skip QSVM (~25 min)
  python cv_evaluation.py --skip-quantum    # Skip QSVM + HybridQT (~5 min)
  python cv_evaluation.py --cv-epochs 20   # Fewer epochs/fold for neural models
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    from utils.statistics import run_all_pairwise_tests, save_statistical_tests
    _HAS_STATS = True
except ImportError:
    _HAS_STATS = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# ── Seeds ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Constants ──────────────────────────────────────────────────────────────────
N_FOLDS = 10
N_NUM = 14          # Number of numeric features (indices 0–13, scaled per fold)
N_QUBITS = 4        # QSVM PCA components
QSVM_MAX_PER_CLASS = 50

SEP  = "=" * 46
DASH = "-" * 46


# ══════════════════════════════════════════════════════════════════════════════
# Shared fold helpers
# ══════════════════════════════════════════════════════════════════════════════

def _scale_fold(X_tr: np.ndarray, X_va: np.ndarray) -> tuple:
    """Fit StandardScaler on fold train, apply to both. Returns (X_tr_s, X_va_s)."""
    X_tr = X_tr.copy()
    X_va = X_va.copy()
    sc = StandardScaler()
    X_tr[:, :N_NUM] = sc.fit_transform(X_tr[:, :N_NUM])
    X_va[:, :N_NUM] = sc.transform(X_va[:, :N_NUM])
    return X_tr, X_va


def _smote_fold(X: np.ndarray, y: np.ndarray) -> tuple:
    """Apply SMOTE to training fold only."""
    sm = SMOTE(random_state=SEED)
    return sm.fit_resample(X, y)


def _fold_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    inference_ms: float = 0.0,
) -> dict:
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    spec = tn / (tn + fp + 1e-12)
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    npv_ = tn / (tn + fn + 1e-12)
    return {
        "acc":          accuracy_score(y_true, y_pred),
        "prec":         precision_score(y_true, y_pred, average="binary", zero_division=0),
        "rec":          recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1":           f1_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc":          roc_auc_score(y_true, y_proba),
        "auc_pr":       average_precision_score(y_true, y_proba),
        "mcc":          float(matthews_corrcoef(y_true, y_pred)),
        "kappa":        float(cohen_kappa_score(y_true, y_pred)),
        "specificity":  spec,
        "npv":          npv_,
        "brier":        float(brier_score_loss(y_true, y_proba)),
        "log_loss_val": float(log_loss(y_true, y_proba)),
        "inference_ms": float(inference_ms),
    }


def _banner(title: str) -> None:
    print(f"\n{SEP}")
    print(f"10-FOLD CV: {title}")
    print(SEP)
    sys.stdout.flush()


def _print_fold(fold_i: int, m: dict) -> None:
    print(
        f"Fold {fold_i:2d}/{N_FOLDS}  "
        f"Acc={m['acc']*100:.2f}%  "
        f"F1={m['f1']*100:.2f}%  "
        f"AUC={m['auc']:.4f}"
    )
    sys.stdout.flush()


def _summarize_folds(fold_metrics: list, model_name: str) -> dict:
    """Print mean ± std summary and return CV result dict."""
    def _m(key): return [m[key] for m in fold_metrics]

    accs  = _m("acc")
    precs = _m("prec")
    recs  = _m("rec")
    f1s   = _m("f1")
    aucs  = _m("auc")

    print(f"\n{DASH}")
    print(
        f"MEAN  "
        f"Acc={np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%  "
        f"F1={np.mean(f1s)*100:.2f}% +/- {np.std(f1s)*100:.2f}%  "
        f"AUC={np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  "
        f"MCC={np.mean(_m('mcc')):.4f}  "
        f"Brier={np.mean(_m('brier')):.4f}"
    )
    print(SEP)
    sys.stdout.flush()

    summary = {
        "Model":            model_name,
        "Accuracy":         round(float(np.mean(accs)), 6),
        "Accuracy_std":     round(float(np.std(accs)), 6),
        "Precision":        round(float(np.mean(precs)), 6),
        "Precision_std":    round(float(np.std(precs)), 6),
        "Recall":           round(float(np.mean(recs)), 6),
        "Recall_std":       round(float(np.std(recs)), 6),
        "F1":               round(float(np.mean(f1s)), 6),
        "F1_std":           round(float(np.std(f1s)), 6),
        "ROC_AUC":          round(float(np.mean(aucs)), 6),
        "ROC_AUC_std":      round(float(np.std(aucs)), 6),
        # Extended Q1 metrics
        "AUC_PR":           round(float(np.mean(_m("auc_pr"))), 6),
        "MCC":              round(float(np.mean(_m("mcc"))), 6),
        "Kappa":            round(float(np.mean(_m("kappa"))), 6),
        "Specificity":      round(float(np.mean(_m("specificity"))), 6),
        "NPV":              round(float(np.mean(_m("npv"))), 6),
        "Brier":            round(float(np.mean(_m("brier"))), 6),
        "LogLoss":          round(float(np.mean(_m("log_loss_val"))), 6),
        "InferenceMs":      round(float(np.mean(_m("inference_ms"))), 4),
        # Per-fold vectors (for statistical tests)
        "_fold_accs":       [round(a, 6) for a in accs],
        "_fold_aucs":       [round(a, 6) for a in aucs],
        "_fold_mccs":       [round(m, 6) for m in _m("mcc")],
    }
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Neural model training helper
# ══════════════════════════════════════════════════════════════════════════════

class _EarlyStopping:
    def __init__(self, patience: int = 10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - 1e-4:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def _train_fold_nn(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    lr: float,
    epochs: int,
    batch_size: int = 32,
    patience: int = 10,
) -> tuple:
    """
    Generic training loop for TabTransformer / HybridQT in a CV fold.
    Returns (y_pred, y_proba) on the validation fold.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    es = _EarlyStopping(patience)

    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_ds = TensorDataset(
        torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=gen)

    best_val = float("inf")
    best_state: dict = {}

    model.to(DEVICE)
    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validate
        model.eval()
        X_va_t = torch.FloatTensor(X_va).to(DEVICE)
        y_va_t = torch.FloatTensor(y_va.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            logits = model(X_va_t).squeeze(-1)
            val_loss = criterion(logits, y_va_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if es(val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        X_va_t = torch.FloatTensor(X_va).to(DEVICE)
        y_proba = torch.sigmoid(model(X_va_t).squeeze(-1)).cpu().numpy()

    y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


# ══════════════════════════════════════════════════════════════════════════════
# Model CV functions
# ══════════════════════════════════════════════════════════════════════════════

def cv_xgboost(X_full: np.ndarray, y_full: np.ndarray, skf: StratifiedKFold) -> tuple:
    """10-fold CV for XGBoost with per-fold SMOTE+scaling."""
    _banner("XGBoost")

    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, eval_metric="logloss",
            tree_method="hist", device="cpu",
        )
        clf.fit(X_tr_sm, y_tr_sm, verbose=False)

        t_inf = time.perf_counter()
        y_pred  = clf.predict(X_va)
        y_proba = clf.predict_proba(X_va)[:, 1]
        inf_ms = (time.perf_counter() - t_inf) / len(y_va) * 1000

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba

        m = _fold_metrics(y_va, y_pred, y_proba, inference_ms=inf_ms)
        fold_metrics.append(m)
        _print_fold(fold_i, m)

    return _summarize_folds(fold_metrics, "XGBoost"), preds_all, proba_all, fold_metrics


def cv_qsvm(X_full: np.ndarray, y_full: np.ndarray, skf: StratifiedKFold) -> tuple:
    """10-fold CV for QSVM with per-fold quantum kernel (slow: ~20 min on CPU)."""
    _banner("Quantum SVM")
    print("  NOTE: Each fold computes 2 kernel matrices. Use --skip-qsvm to skip.")
    sys.stdout.flush()

    sys.path.insert(0, str(BASE_DIR))
    from models.baselines import build_quantum_kernel, compute_kernel_matrix

    kernel_fn = build_quantum_kernel()  # build once, reuse across folds

    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        t0 = time.perf_counter()
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        # PCA: N_QUBITS components; normalize to [-pi, pi]
        pca = PCA(n_components=N_QUBITS, random_state=SEED)
        X_tr_pca = pca.fit_transform(X_tr_sm)
        X_va_pca = pca.transform(X_va)

        max_abs = np.abs(X_tr_pca).max(axis=0, keepdims=True) + 1e-8
        X_tr_pca = X_tr_pca / max_abs * np.pi
        X_va_pca = X_va_pca / max_abs * np.pi

        # Stratified subset for kernel
        rng = np.random.default_rng(SEED + fold_i)
        subset_idx = []
        for cls in np.unique(y_tr_sm):
            cls_idx = np.where(y_tr_sm == cls)[0]
            n_take = min(QSVM_MAX_PER_CLASS, len(cls_idx))
            subset_idx.extend(rng.choice(cls_idx, n_take, replace=False).tolist())
        subset_idx = np.array(subset_idx)

        ker_X_tr = X_tr_pca[subset_idx]
        ker_y_tr = y_tr_sm[subset_idx]

        n_sub = len(ker_X_tr)
        print(f"  Fold {fold_i:2d}/{N_FOLDS} — kernel ({n_sub}x{n_sub} train, "
              f"{len(X_va_pca)}x{n_sub} val) ...", end="", flush=True)

        K_tr = compute_kernel_matrix(kernel_fn, ker_X_tr, ker_X_tr, verbose=False)
        K_tr += 1e-6 * np.eye(n_sub)

        svc = SVC(kernel="precomputed", probability=True, random_state=SEED, C=10.0)
        svc.fit(K_tr, ker_y_tr)

        K_va = compute_kernel_matrix(kernel_fn, X_va_pca, ker_X_tr, verbose=False)

        y_pred  = svc.predict(K_va)
        y_proba = svc.predict_proba(K_va)[:, 1]

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba

        elapsed = time.perf_counter() - t0
        m = _fold_metrics(y_va, y_pred, y_proba)
        fold_metrics.append(m)
        print(f" done ({elapsed:.0f}s)  Acc={m['acc']*100:.2f}%  F1={m['f1']*100:.2f}%  AUC={m['auc']:.4f}")
        sys.stdout.flush()

    return _summarize_folds(fold_metrics, "Quantum SVM"), preds_all, proba_all, fold_metrics


def cv_tab_transformer(
    X_full: np.ndarray,
    y_full: np.ndarray,
    skf: StratifiedKFold,
    epochs: int,
) -> tuple:
    """10-fold CV for classical TabTransformer."""
    _banner("Classical TabTransformer")
    sys.path.insert(0, str(BASE_DIR))
    from models.tab_transformer import TabTransformer

    n_features = X_full.shape[1]
    config = {
        "n_features": n_features, "d_model": 32, "n_heads": 4,
        "n_layers": 2, "dim_ff": 128, "dropout": 0.1,
    }

    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        torch.manual_seed(SEED)
        model = TabTransformer(**config)
        y_pred, y_proba = _train_fold_nn(
            model, X_tr_sm, y_tr_sm, X_va, y_va,
            lr=1e-3, epochs=epochs, batch_size=32, patience=10,
        )

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba

        m = _fold_metrics(y_va, y_pred, y_proba)
        fold_metrics.append(m)
        _print_fold(fold_i, m)

    return _summarize_folds(fold_metrics, "Classical TabTransformer"), preds_all, proba_all, fold_metrics


def cv_lightgbm(X_full: np.ndarray, y_full: np.ndarray, skf: StratifiedKFold) -> tuple:
    """10-fold CV for LightGBM baseline."""
    _banner("LightGBM")
    if not _HAS_LGB:
        print("  LightGBM not installed — skipping.")
        return None, np.full(len(y_full), -1, dtype=int), np.zeros(len(y_full)), []

    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        clf = lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1,
        )
        clf.fit(X_tr_sm, y_tr_sm)

        t_inf = time.perf_counter()
        y_pred  = clf.predict(X_va)
        y_proba = clf.predict_proba(X_va)[:, 1]
        inf_ms = (time.perf_counter() - t_inf) / len(y_va) * 1000

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        m = _fold_metrics(y_va, y_pred, y_proba, inference_ms=inf_ms)
        fold_metrics.append(m)
        _print_fold(fold_i, m)

    return _summarize_folds(fold_metrics, "LightGBM"), preds_all, proba_all, fold_metrics


def cv_mlp(
    X_full: np.ndarray,
    y_full: np.ndarray,
    skf: StratifiedKFold,
    epochs: int,
) -> tuple:
    """10-fold CV for MLP baseline."""
    _banner("MLP Baseline")
    sys.path.insert(0, str(BASE_DIR))
    from models.baselines import MLP

    n_features = X_full.shape[1]
    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        torch.manual_seed(SEED)
        model = MLP(n_features)
        y_pred, y_proba = _train_fold_nn(
            model, X_tr_sm, y_tr_sm, X_va, y_va,
            lr=1e-3, epochs=epochs, batch_size=32, patience=10,
        )

        t_inf = time.perf_counter()
        _ = model.predict_proba(torch.FloatTensor(X_va))
        inf_ms = (time.perf_counter() - t_inf) / len(y_va) * 1000

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        m = _fold_metrics(y_va, y_pred, y_proba, inference_ms=inf_ms)
        fold_metrics.append(m)
        _print_fold(fold_i, m)

    return _summarize_folds(fold_metrics, "MLP"), preds_all, proba_all, fold_metrics


def cv_hybrid_transformer(
    X_full: np.ndarray,
    y_full: np.ndarray,
    skf: StratifiedKFold,
    epochs: int,
) -> tuple:
    """10-fold CV for Hybrid Quantum-Classical Transformer (6-qubit HEA)."""
    _banner("Hybrid Quantum-Classical Transformer (6q HEA)")
    print("  NOTE: VQC evaluation makes each fold slower than classical.")
    sys.stdout.flush()

    sys.path.insert(0, str(BASE_DIR))
    from models.hybrid_quantum_transformer import HybridTabTransformer, DEFAULT_QC_CFG

    n_features = X_full.shape[1]

    fold_metrics = []
    preds_all = np.full(len(y_full), -1, dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        torch.manual_seed(SEED)
        model = HybridTabTransformer(
            n_features=n_features, d_model=32, n_heads=4,
            n_layers=2, dropout=0.1, qc_cfg=DEFAULT_QC_CFG,
        )
        y_pred, y_proba = _train_fold_nn(
            model, X_tr_sm, y_tr_sm, X_va, y_va,
            lr=5e-4, epochs=epochs, batch_size=32, patience=10,
        )

        t_inf = time.perf_counter()
        _ = model.predict_proba(torch.FloatTensor(X_va))
        inf_ms = (time.perf_counter() - t_inf) / len(y_va) * 1000

        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba

        m = _fold_metrics(y_va, y_pred, y_proba, inference_ms=inf_ms)
        fold_metrics.append(m)
        _print_fold(fold_i, m)

    return _summarize_folds(fold_metrics, "Hybrid Quantum Transformer"), preds_all, proba_all, fold_metrics


# ══════════════════════════════════════════════════════════════════════════════
# McNemar's test
# ══════════════════════════════════════════════════════════════════════════════

def run_mcnemar(
    hqct_preds: np.ndarray,
    xgb_preds: np.ndarray,
    y_true: np.ndarray,
) -> tuple:
    """
    McNemar's test comparing HQCT vs XGBoost per-sample predictions.
    Returns (p_value, statistic, result_str).
    """
    print(f"\n{SEP}")
    print("McNEMAR'S TEST: HQCT vs XGBoost")
    print(SEP)

    hqct_ok = hqct_preds == y_true
    xgb_ok  = xgb_preds  == y_true

    a = int(np.sum( hqct_ok &  xgb_ok))   # both correct
    b = int(np.sum( hqct_ok & ~xgb_ok))   # HQCT correct, XGBoost wrong
    c = int(np.sum(~hqct_ok &  xgb_ok))   # HQCT wrong, XGBoost correct
    d = int(np.sum(~hqct_ok & ~xgb_ok))   # both wrong

    print("Contingency table:")
    print(f"  HQCT correct   & XGBoost correct : {a:3d} samples")
    print(f"  HQCT correct   & XGBoost wrong   : {b:3d} samples  (b)")
    print(f"  HQCT wrong     & XGBoost correct : {c:3d} samples  (c)")
    print(f"  HQCT wrong     & XGBoost wrong   : {d:3d} samples")
    print()

    try:
        from statsmodels.stats.contingency_tables import mcnemar as _mcnemar
        table = np.array([[a, b], [c, d]])
        res = _mcnemar(table, exact=True)
        p_val = float(res.pvalue)
        stat  = float(res.statistic)

        sig_str = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"

        if b + c == 0:
            interp = ("Models make identical predictions on every sample; "
                      "McNemar test is undefined — performance is indistinguishable.")
        elif p_val < 0.05:
            better = "HQCT" if b > c else "XGBoost"
            interp = (f"{better} makes significantly fewer errors (p={p_val:.4f} < 0.05), "
                      f"supporting a quantum-advantage claim in the paper.")
        else:
            interp = (f"No statistically significant difference (p={p_val:.4f} >= 0.05); "
                      f"report performance parity rather than quantum advantage.")

        print(f"McNemar statistic : {stat:.4f}")
        print(f"p-value           : {p_val:.4f}")
        print(f"Result            : {sig_str} at alpha=0.05")
        print(f"\nInterpretation: {interp}")
        print(SEP)
        sys.stdout.flush()
        return p_val, stat, sig_str, a, b, c, d

    except ImportError:
        print("WARNING: statsmodels not installed — McNemar test skipped.")
        print("Install with: pip install statsmodels")
        print(SEP)
        sys.stdout.flush()
        return None, None, "N/A (statsmodels not installed)", a, b, c, d


# ══════════════════════════════════════════════════════════════════════════════
# Summary table + save
# ══════════════════════════════════════════════════════════════════════════════

def _print_summary_table(cv_rows: list) -> None:
    """Print final CV summary table."""
    print(f"\n{'='*80}")
    print("FINAL 10-FOLD CV SUMMARY TABLE")
    print('='*80)

    display_rows = []
    for row in cv_rows:
        display_rows.append({
            "Model":     row["Model"],
            "Accuracy":  f"{row['Accuracy']*100:.2f}% +/- {row['Accuracy_std']*100:.2f}%",
            "F1-Score":  f"{row['F1']*100:.2f}% +/- {row['F1_std']*100:.2f}%",
            "ROC-AUC":   f"{row['ROC_AUC']:.4f} +/- {row['ROC_AUC_std']:.4f}",
        })

    df_disp = pd.DataFrame(display_rows)
    try:
        from tabulate import tabulate
        print(tabulate(df_disp, headers="keys", tablefmt="grid", showindex=False))
    except ImportError:
        print(df_disp.to_string(index=False))

    print('='*80)
    sys.stdout.flush()


def _save_results(
    cv_rows: list,
    mcnemar_info: tuple,
    mcnemar_abcd: tuple | None,
    fold_metrics_dict: dict,
    fold_probas: dict,
) -> None:
    """Save cv_results.csv, mcnemar_result.txt, mcnemar_detail.json,
    full_metrics_ckd.csv, and ckd_fold_probas.npz."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # cv_results.csv — strip private _fold_* keys
    public_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in cv_rows]
    df = pd.DataFrame(public_rows)
    csv_path = RESULTS_DIR / "cv_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  results/cv_results.csv saved ({len(df)} models)")

    # mcnemar_result.txt (preserve existing format)
    p_val, stat, sig_str = mcnemar_info
    txt_path = RESULTS_DIR / "mcnemar_result.txt"
    lines = [f"p_value={p_val}", f"statistic={stat}", f"result={sig_str}"]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  results/mcnemar_result.txt saved")

    # mcnemar_detail.json — contingency table a,b,c,d (additive to txt)
    if mcnemar_abcd is not None:
        a, b, c, d = mcnemar_abcd
        detail = {
            "a_both_correct": a,
            "b_hqct_correct_xgb_wrong": b,
            "c_hqct_wrong_xgb_correct": c,
            "d_both_wrong": d,
            "p_value": p_val,
            "statistic": stat,
            "result": sig_str,
        }
        json_path = RESULTS_DIR / "mcnemar_detail.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(detail, f, indent=2)
        print(f"  results/mcnemar_detail.json saved")

    # full_metrics_ckd.csv — one row per fold per model (14 metrics)
    full_rows = []
    for model_name, folds in fold_metrics_dict.items():
        for fold_i, m in enumerate(folds, 1):
            row = {"model": model_name, "fold": fold_i}
            row.update(m)
            full_rows.append(row)
    if full_rows:
        df_full = pd.DataFrame(full_rows)
        full_csv = RESULTS_DIR / "full_metrics_ckd.csv"
        df_full.to_csv(full_csv, index=False)
        print(f"  results/full_metrics_ckd.csv saved ({len(df_full)} rows)")

    # ckd_fold_probas.npz — probability arrays for ROC/PR plotting
    if fold_probas:
        npz_path = RESULTS_DIR / "ckd_fold_probas.npz"
        np.savez(npz_path, **{k: v for k, v in fold_probas.items() if v is not None})
        print(f"  results/ckd_fold_probas.npz saved (keys: {list(fold_probas.keys())})")

    sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="10-fold stratified CV for CKD pipeline (SMOTE inside folds)"
    )
    parser.add_argument("--skip-qsvm",    action="store_true",
                        help="Skip Quantum SVM (~20 min savings)")
    parser.add_argument("--skip-quantum", action="store_true",
                        help="Skip both QSVM and Hybrid Quantum Transformer")
    parser.add_argument("--cv-epochs",   type=int, default=50,
                        help="Max epochs per fold for neural models (default: 50)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skip_qsvm   = args.skip_qsvm or args.skip_quantum
    skip_hqct   = args.skip_quantum

    print("=" * 60)
    print("  10-FOLD STRATIFIED CROSS-VALIDATION — CKD PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Device : {DEVICE}")
    print(f"  Config : cv_epochs={args.cv_epochs}, skip_qsvm={skip_qsvm}, "
          f"skip_quantum={skip_hqct}")
    print("=" * 60)

    # ── Load full dataset (pre-scale, pre-SMOTE) ───────────────────────────────
    for fname in ["X_full.npy", "y_full.npy"]:
        if not (DATA_DIR / fname).exists():
            print(f"ERROR: {DATA_DIR / fname} not found.")
            print("Run preprocessing.py first to generate X_full.npy and y_full.npy.")
            sys.exit(1)

    X_full = np.load(DATA_DIR / "X_full.npy")
    y_full = np.load(DATA_DIR / "y_full.npy")
    print(f"\nLoaded X_full {X_full.shape}, y_full {y_full.shape}")
    unique, counts = np.unique(y_full, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {'CKD' if cls == 1 else 'Not-CKD'}: {cnt} samples")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    cv_rows: list = []
    fold_metrics_dict: dict = {}
    fold_probas: dict = {}

    # ── XGBoost ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    xgb_summary, xgb_preds, xgb_proba, xgb_folds = cv_xgboost(X_full, y_full, skf)
    cv_rows.append(xgb_summary)
    fold_metrics_dict["XGBoost"] = xgb_folds
    fold_probas["xgb"] = xgb_proba
    print(f"  [XGBoost CV done in {time.perf_counter()-t0:.0f}s]")

    # ── QSVM ──────────────────────────────────────────────────────────────────
    if skip_qsvm:
        print(f"\n{SEP}")
        print("10-FOLD CV: Quantum SVM  [SKIPPED via --skip-qsvm / --skip-quantum]")
        print(SEP)
    else:
        t0 = time.perf_counter()
        qsvm_summary, _, qsvm_proba, qsvm_folds = cv_qsvm(X_full, y_full, skf)
        cv_rows.append(qsvm_summary)
        fold_metrics_dict["Quantum SVM"] = qsvm_folds
        fold_probas["qsvm"] = qsvm_proba
        print(f"  [QSVM CV done in {time.perf_counter()-t0:.0f}s]")

    # ── Classical TabTransformer ───────────────────────────────────────────────
    t0 = time.perf_counter()
    tab_summary, _, tab_proba, tab_folds = cv_tab_transformer(
        X_full, y_full, skf, args.cv_epochs
    )
    cv_rows.append(tab_summary)
    fold_metrics_dict["Classical TabTransformer"] = tab_folds
    fold_probas["tab"] = tab_proba
    print(f"  [TabTransformer CV done in {time.perf_counter()-t0:.0f}s]")

    # ── LightGBM ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    lgb_summary, _, lgb_proba, lgb_folds = cv_lightgbm(X_full, y_full, skf)
    if lgb_summary is not None:
        cv_rows.append(lgb_summary)
        fold_metrics_dict["LightGBM"] = lgb_folds
        fold_probas["lgb"] = lgb_proba
    print(f"  [LightGBM CV done in {time.perf_counter()-t0:.0f}s]")

    # ── MLP Baseline ──────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    mlp_summary, _, mlp_proba, mlp_folds = cv_mlp(X_full, y_full, skf, args.cv_epochs)
    cv_rows.append(mlp_summary)
    fold_metrics_dict["MLP"] = mlp_folds
    fold_probas["mlp"] = mlp_proba
    print(f"  [MLP CV done in {time.perf_counter()-t0:.0f}s]")

    # ── Hybrid Quantum Transformer ─────────────────────────────────────────────
    if skip_hqct:
        print(f"\n{SEP}")
        print("10-FOLD CV: Hybrid Quantum Transformer  [SKIPPED via --skip-quantum]")
        print(SEP)
        hqct_preds = None
        hqct_proba = None
    else:
        t0 = time.perf_counter()
        hqct_summary, hqct_preds, hqct_proba, hqct_folds = cv_hybrid_transformer(
            X_full, y_full, skf, args.cv_epochs
        )
        cv_rows.append(hqct_summary)
        fold_metrics_dict["Hybrid Quantum Transformer"] = hqct_folds
        fold_probas["hqct"] = hqct_proba
        print(f"  [HybridQT CV done in {time.perf_counter()-t0:.0f}s]")

    # ── McNemar's test ────────────────────────────────────────────────────────
    if hqct_preds is not None and np.all(xgb_preds >= 0):
        mcnemar_result = run_mcnemar(hqct_preds, xgb_preds, y_full)
        mcnemar_info = mcnemar_result[:3]
        mcnemar_abcd = mcnemar_result[3:]
    else:
        print(f"\n{SEP}")
        print("McNEMAR'S TEST: Skipped (HybridQT was not run).")
        print(SEP)
        mcnemar_info = (None, None, "N/A (HybridQT skipped)")
        mcnemar_abcd = None

    # ── Statistical tests (Wilcoxon, Friedman, bootstrap CIs) ────────────────
    if _HAS_STATS:
        print(f"\n{SEP}")
        print("STATISTICAL TESTS (Wilcoxon, Friedman, Bootstrap CIs)")
        print(SEP)
        fold_scores = {
            row["Model"]: row["_fold_aucs"]
            for row in cv_rows if "_fold_aucs" in row
        }
        fold_probas_stats = {
            "XGBoost": fold_probas.get("xgb"),
            "Hybrid Quantum Transformer": fold_probas.get("hqct"),
        }
        try:
            stat_results = run_all_pairwise_tests(fold_scores, fold_probas_stats, y_full)
            stat_out = RESULTS_DIR / "statistical_tests.json"
            save_statistical_tests(stat_results, str(stat_out))
            print(f"  results/statistical_tests.json saved")
        except Exception as exc:
            print(f"  WARNING: Statistical tests failed: {exc}")
        sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary_table(cv_rows)
    _save_results(cv_rows, mcnemar_info, mcnemar_abcd, fold_metrics_dict, fold_probas)

    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
