"""
fhs_cv_evaluation.py -- 10-fold stratified cross-validation on FHS data.
Identical protocol to cv_evaluation.py; only N_NUM=8 and n_features=15 differ.
Step FHS-3 of the QIP 2027 dual-dataset pipeline.

Usage:
  python fhs_cv_evaluation.py [--skip-qsvm] [--skip-quantum] [--cv-epochs N]
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_NUM = 8       # FHS continuous features (indices 0-7)
N_SPLITS = 10

sys.path.insert(0, str(BASE_DIR))


# ── Fold helpers ───────────────────────────────────────────────────────────────

def _scale_fold(X_tr: np.ndarray, X_va: np.ndarray):
    """Fit StandardScaler on continuous cols of train, transform both."""
    X_tr = X_tr.copy(); X_va = X_va.copy()
    sc = StandardScaler()
    X_tr[:, :N_NUM] = sc.fit_transform(X_tr[:, :N_NUM])
    X_va[:, :N_NUM] = sc.transform(X_va[:, :N_NUM])
    return X_tr, X_va


def _smote_fold(X: np.ndarray, y: np.ndarray):
    """Apply SMOTE (sampling_strategy=1.0) to training fold."""
    return SMOTE(sampling_strategy=1.0, random_state=SEED).fit_resample(X, y)


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    gen = torch.Generator(); gen.manual_seed(SEED)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=gen if shuffle else None)


def _train_fold_nn(model, X_tr, y_tr, X_va, y_va, lr, epochs, batch_size=32, patience=10):
    """Generic training loop for one CV fold. Returns (y_pred, y_proba)."""
    from models.tab_transformer import EarlyStopping
    train_loader = _make_loader(X_tr, y_tr, batch_size, shuffle=True)
    val_loader = _make_loader(X_va, y_va, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()
    early_stop = EarlyStopping(patience=patience)

    best_val_loss = float("inf")
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(criterion(model(Xb).squeeze(-1), yb).item())
        avg_val = float(np.mean(val_losses))
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if early_stop(avg_val):
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Xv = torch.FloatTensor(X_va).to(DEVICE)
        logits = model(Xv).squeeze(-1)
        y_proba = torch.sigmoid(logits).cpu().numpy()
        y_pred = (y_proba > 0.5).astype(int)
    return y_pred, y_proba


def _fold_metrics(y_true, y_pred, y_proba):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_proba),
    }


def _summarize(name, metrics_list):
    keys = ["acc", "prec", "rec", "f1", "auc"]
    means = {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}
    stds = {k: float(np.std([m[k] for m in metrics_list])) for k in keys}
    print(f"\n  {name}:")
    print(f"    Acc={means['acc']*100:.2f}% +/-{stds['acc']*100:.2f}% | "
          f"F1={means['f1']*100:.2f}% +/-{stds['f1']*100:.2f}% | "
          f"AUC={means['auc']:.4f} +/-{stds['auc']:.4f}")
    return {"Model": name,
            "Accuracy": means["acc"], "Accuracy_std": stds["acc"],
            "Precision": means["prec"], "Precision_std": stds["prec"],
            "Recall": means["rec"], "Recall_std": stds["rec"],
            "F1": means["f1"], "F1_std": stds["f1"],
            "ROC_AUC": means["auc"], "ROC_AUC_std": stds["auc"]}


# ── Per-model CV functions ─────────────────────────────────────────────────────

def cv_xgboost(X_full, y_full, skf):
    print("\nRunning FHS XGBoost 10-fold CV...")
    metrics_list = []
    preds_all = np.zeros(len(y_full), dtype=int)
    proba_all = np.zeros(len(y_full))
    fold_probas, fold_labels = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])

        clf = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED,
            eval_metric="logloss", tree_method="hist", device="cpu",
        )
        clf.fit(X_tr_sm, y_tr_sm, verbose=False)

        y_pred = clf.predict(X_va)
        y_proba = clf.predict_proba(X_va)[:, 1]
        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        fold_probas.append(y_proba)
        fold_labels.append(y_full[va_idx])
        metrics_list.append(_fold_metrics(y_full[va_idx], y_pred, y_proba))
        print(f"  Fold {fold:2d} | Acc={metrics_list[-1]['acc']*100:.2f}%  F1={metrics_list[-1]['f1']*100:.2f}%  AUC={metrics_list[-1]['auc']:.4f}")
        sys.stdout.flush()

    return _summarize("XGBoost", metrics_list), preds_all, proba_all, fold_probas, fold_labels


def cv_qsvm(X_full, y_full, skf):
    print("\nRunning FHS QSVM 10-fold CV (this may take a while)...")
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from models.baselines import build_quantum_kernel, compute_kernel_matrix

    qkernel = build_quantum_kernel()
    metrics_list = []
    preds_all = np.zeros(len(y_full), dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])

        rng = np.random.RandomState(SEED + fold)
        idx0 = np.where(y_tr_sm == 0)[0]; idx1 = np.where(y_tr_sm == 1)[0]
        sub_size = min(50, len(idx0), len(idx1))
        sub_idx = np.concatenate([
            rng.choice(idx0, sub_size, replace=False),
            rng.choice(idx1, sub_size, replace=False),
        ])
        X_sub = X_tr_sm[sub_idx]; y_sub = y_tr_sm[sub_idx]

        pca = PCA(n_components=4, random_state=SEED)
        X_sub_pca = pca.fit_transform(X_sub)
        X_va_pca = pca.transform(X_va)

        norm_max = np.abs(X_sub_pca).max(axis=0) + 1e-8
        X_sub_n = X_sub_pca / norm_max * np.pi
        X_va_n = X_va_pca / norm_max * np.pi

        K_train = compute_kernel_matrix(X_sub_n, X_sub_n, qkernel)
        K_val = compute_kernel_matrix(X_va_n, X_sub_n, qkernel)

        svm = SVC(kernel="precomputed", probability=True, C=10.0, random_state=SEED)
        svm.fit(K_train, y_sub)
        y_pred = svm.predict(K_val)
        y_proba = svm.predict_proba(K_val)[:, 1]
        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        metrics_list.append(_fold_metrics(y_full[va_idx], y_pred, y_proba))
        print(f"  Fold {fold:2d} | Acc={metrics_list[-1]['acc']*100:.2f}%  F1={metrics_list[-1]['f1']*100:.2f}%  AUC={metrics_list[-1]['auc']:.4f}")
        sys.stdout.flush()

    return _summarize("QSVM", metrics_list), preds_all, proba_all


def cv_tab_transformer(X_full, y_full, skf, epochs):
    print(f"\nRunning FHS Classical TabTransformer 10-fold CV (epochs={epochs})...")
    from models.tab_transformer import TabTransformer
    n_features = X_full.shape[1]
    config = {"n_features": n_features, "d_model": 32, "n_heads": 4,
              "n_layers": 2, "dim_ff": 128, "dropout": 0.1}
    metrics_list = []
    preds_all = np.zeros(len(y_full), dtype=int)
    proba_all = np.zeros(len(y_full))

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])

        model = TabTransformer(**config).to(DEVICE)
        y_pred, y_proba = _train_fold_nn(
            model, X_tr_sm, y_tr_sm, X_va, y_full[va_idx],
            lr=1e-3, epochs=epochs, patience=10
        )
        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        metrics_list.append(_fold_metrics(y_full[va_idx], y_pred, y_proba))
        print(f"  Fold {fold:2d} | Acc={metrics_list[-1]['acc']*100:.2f}%  F1={metrics_list[-1]['f1']*100:.2f}%  AUC={metrics_list[-1]['auc']:.4f}")
        sys.stdout.flush()

    return _summarize("Classical TabTransformer", metrics_list), preds_all, proba_all


def cv_hybrid_transformer(X_full, y_full, skf, epochs, hqct_subsample: int = 0):
    """
    hqct_subsample > 0: stratified subsample of SMOTE'd training data to this size.
    Analogous to QSVM's 50/class subset — necessary on large datasets because each
    forward pass runs B*n_features quantum circuits per batch.
    """
    label = f"epochs={epochs}"
    if hqct_subsample > 0:
        label += f", subsample={hqct_subsample}"
    print(f"\nRunning FHS Hybrid Quantum Transformer 10-fold CV ({label})...")
    from models.hybrid_quantum_transformer import HybridTabTransformer
    n_features = X_full.shape[1]
    config = {"n_features": n_features, "d_model": 32, "n_heads": 4,
              "n_layers": 2, "dropout": 0.1}
    metrics_list = []
    preds_all = np.zeros(len(y_full), dtype=int)
    proba_all = np.zeros(len(y_full))
    fold_probas, fold_labels = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])

        # Subsample for computational tractability on large datasets
        if hqct_subsample > 0 and len(X_tr_sm) > hqct_subsample:
            rng = np.random.RandomState(SEED + fold)
            idx0 = np.where(y_tr_sm == 0)[0]
            idx1 = np.where(y_tr_sm == 1)[0]
            per_class = hqct_subsample // 2
            sub_idx = np.concatenate([
                rng.choice(idx0, min(per_class, len(idx0)), replace=False),
                rng.choice(idx1, min(per_class, len(idx1)), replace=False),
            ])
            X_tr_sm = X_tr_sm[sub_idx]
            y_tr_sm = y_tr_sm[sub_idx]

        model = HybridTabTransformer(**config).to(DEVICE)
        y_pred, y_proba = _train_fold_nn(
            model, X_tr_sm, y_tr_sm, X_va, y_full[va_idx],
            lr=5e-4, epochs=epochs, patience=10
        )
        preds_all[va_idx] = y_pred
        proba_all[va_idx] = y_proba
        fold_probas.append(y_proba)
        fold_labels.append(y_full[va_idx])
        metrics_list.append(_fold_metrics(y_full[va_idx], y_pred, y_proba))
        print(f"  Fold {fold:2d} | Acc={metrics_list[-1]['acc']*100:.2f}%  F1={metrics_list[-1]['f1']*100:.2f}%  AUC={metrics_list[-1]['auc']:.4f}")
        sys.stdout.flush()

    return _summarize("Hybrid Quantum Transformer", metrics_list), preds_all, proba_all, fold_probas, fold_labels


# ── McNemar's test ─────────────────────────────────────────────────────────────

def run_mcnemar(hqct_preds, xgb_preds, y_true):
    """McNemar's exact test between HQCT and XGBoost predictions."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar as mc_test
    except ImportError:
        print("\n  [WARNING] statsmodels not installed -- McNemar test skipped.")
        return None, None, "N/A"

    hqct_correct = (hqct_preds == y_true)
    xgb_correct = (xgb_preds == y_true)

    a = int(np.sum(hqct_correct & xgb_correct))
    b = int(np.sum(hqct_correct & ~xgb_correct))
    c = int(np.sum(~hqct_correct & xgb_correct))
    d = int(np.sum(~hqct_correct & ~xgb_correct))

    print(f"\n  McNemar contingency table:")
    print(f"    Both correct (a): {a}  |  HQCT only (b): {b}")
    print(f"    XGB only (c): {c}    |  Both wrong (d): {d}")

    table = [[a, b], [c, d]]
    result = mc_test(table, exact=True)
    p_val = float(result.pvalue)
    stat = float(result.statistic)
    sig_str = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
    print(f"\n  McNemar: statistic={stat:.4f}  p={p_val:.4f}  -> {sig_str} at alpha=0.05")
    return p_val, stat, sig_str


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FHS 10-fold CV evaluation")
    parser.add_argument("--skip-qsvm", action="store_true", help="Skip QSVM")
    parser.add_argument("--skip-quantum", action="store_true", help="Skip QSVM and HybridQT")
    parser.add_argument("--cv-epochs", type=int, default=50, help="Epochs per fold for neural models")
    parser.add_argument("--hqct-subsample", type=int, default=800,
                        help="Max SMOTE'd training samples per fold for HybridQT (0=no limit; default 800)")
    args = parser.parse_args()

    if args.skip_quantum:
        args.skip_qsvm = True

    print("=" * 60)
    print("FHS STEP 3 -- 10-FOLD CROSS-VALIDATION")
    print("=" * 60)
    print(f"Using device: {DEVICE}")

    for fname in ["fhs_X_full.npy", "fhs_y_full.npy"]:
        if not (DATA_DIR / fname).exists():
            raise FileNotFoundError(f"{DATA_DIR / fname} not found. Run fhs_preprocessing.py first.")

    X_full = np.load(DATA_DIR / "fhs_X_full.npy")
    y_full = np.load(DATA_DIR / "fhs_y_full.npy")
    print(f"\nLoaded: X_full={X_full.shape}  y_full={y_full.shape}")
    print(f"Class distribution: CHD={y_full.sum()} ({y_full.mean()*100:.1f}%)  "
          f"noCHD={len(y_full)-y_full.sum()} ({(1-y_full.mean())*100:.1f}%)")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    xgb_preds = hqct_preds = None
    xgb_fold_probas = xgb_fold_labels = None
    hqct_fold_probas = hqct_fold_labels = None

    # XGBoost
    print("\n" + "-" * 60)
    xgb_summary, xgb_preds, xgb_proba, xgb_fold_probas, xgb_fold_labels = cv_xgboost(X_full, y_full, skf)
    all_summaries.append(xgb_summary)

    # QSVM
    if not args.skip_qsvm:
        print("\n" + "-" * 60)
        qsvm_summary, _, _ = cv_qsvm(X_full, y_full, skf)
        all_summaries.append(qsvm_summary)
    else:
        print("\nQSVM skipped.")

    # Classical TabTransformer
    print("\n" + "-" * 60)
    tt_summary, _, _ = cv_tab_transformer(X_full, y_full, skf, args.cv_epochs)
    all_summaries.append(tt_summary)

    # Hybrid Quantum Transformer
    if not args.skip_quantum:
        print("\n" + "-" * 60)
        hqct_summary, hqct_preds, hqct_proba, hqct_fold_probas, hqct_fold_labels = cv_hybrid_transformer(
            X_full, y_full, skf, args.cv_epochs, hqct_subsample=args.hqct_subsample
        )
        all_summaries.append(hqct_summary)
    else:
        print("\nHybrid Quantum Transformer skipped.")

    # Results table
    print("\n" + "=" * 60)
    print("FHS 10-FOLD CV RESULTS SUMMARY")
    print("=" * 60)
    import pandas as pd
    df = pd.DataFrame(all_summaries)
    print(df.to_string(index=False))

    csv_path = RESULTS_DIR / "fhs_cv_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # McNemar's test
    mcnemar_path = RESULTS_DIR / "fhs_mcnemar_result.txt"
    if hqct_preds is not None and xgb_preds is not None:
        print("\n" + "-" * 60)
        print("FHS McNemar's Test (HQCT vs XGBoost)")
        print("-" * 60)
        p_val, stat, sig_str = run_mcnemar(hqct_preds, xgb_preds, y_full)
        if p_val is not None:
            result_str = (f"statistic={stat:.4f}\n"
                          f"p_value={p_val:.4f}\n"
                          f"result={sig_str} at alpha=0.05\n"
                          f"discordant_b={int(np.sum((hqct_preds == y_full) & (xgb_preds != y_full)))}\n"
                          f"discordant_c={int(np.sum((hqct_preds != y_full) & (xgb_preds == y_full)))}")
            mcnemar_path.write_text(result_str)
            print(f"Saved: {mcnemar_path}")
    else:
        mcnemar_path.write_text("p_value=N/A\nresult=McNemar test not run (HQCT skipped)")

    # Save per-fold proba arrays for ROC curve plotting
    if xgb_fold_probas is not None:
        max_fold_len = max(len(p) for p in xgb_fold_probas)
        xgb_p_arr = np.full((N_SPLITS, max_fold_len), np.nan)
        xgb_l_arr = np.full((N_SPLITS, max_fold_len), np.nan)
        for i, (p, l) in enumerate(zip(xgb_fold_probas, xgb_fold_labels)):
            xgb_p_arr[i, :len(p)] = p
            xgb_l_arr[i, :len(l)] = l

        save_dict = {"xgb_probas": xgb_p_arr, "xgb_labels": xgb_l_arr}

        if hqct_fold_probas is not None:
            hqct_p_arr = np.full((N_SPLITS, max_fold_len), np.nan)
            hqct_l_arr = np.full((N_SPLITS, max_fold_len), np.nan)
            for i, (p, l) in enumerate(zip(hqct_fold_probas, hqct_fold_labels)):
                hqct_p_arr[i, :len(p)] = p
                hqct_l_arr[i, :len(l)] = l
            save_dict["hqct_probas"] = hqct_p_arr
            save_dict["hqct_labels"] = hqct_l_arr

        np.savez(RESULTS_DIR / "fhs_fold_probas.npz", **save_dict)
        print(f"Saved: {RESULTS_DIR / 'fhs_fold_probas.npz'}")

    print("\n" + "=" * 60)
    print("FHS cross-validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
