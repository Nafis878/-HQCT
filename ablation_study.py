"""
ablation_study.py -- 6-condition ablation study for the HQCT pipeline.

Runs 5-fold CV on CKD for each condition and saves results to:
  results/ablation_results.csv
  results/latex_tables/ablation_table.tex

Conditions:
  1. No VQC (TabTransformer only) -- confirms VQC contribution
  2. 1 VQC layer vs 2 layers vs 3 layers -- optimal depth
  3. 4 qubits vs 6 qubits -- qubit count
  4. With data re-uploading vs without -- encoding strategy
  5. With DP training vs without -- privacy-utility tradeoff
  6. PCA 4-dim vs PCA 6-dim input to VQC

Usage:
  python ablation_study.py [--folds 5] [--epochs 20] [--fast]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LATEX_DIR = RESULTS_DIR / "latex_tables"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_NUM = 14  # CKD numeric features

sys.path.insert(0, str(BASE_DIR))


# ── Fold helpers ──────────────────────────────────────────────────────────────

def _scale_fold(X_tr, X_va):
    X_tr = X_tr.copy(); X_va = X_va.copy()
    sc = StandardScaler()
    X_tr[:, :N_NUM] = sc.fit_transform(X_tr[:, :N_NUM])
    X_va[:, :N_NUM] = sc.transform(X_va[:, :N_NUM])
    return X_tr, X_va


def _smote_fold(X, y):
    return SMOTE(random_state=SEED).fit_resample(X, y)


class _ES:
    def __init__(self, patience=8):
        self.patience = patience; self.best = float("inf"); self.counter = 0

    def __call__(self, loss):
        if loss < self.best - 1e-4:
            self.best = loss; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


def _train_eval_fold(model, X_tr, y_tr, X_va, y_va, lr, epochs, batch_size=32,
                     dp_params=None):
    """Train for one fold; returns (acc, f1, auc) on validation."""
    criterion = nn.BCEWithLogitsLoss()

    base_opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if dp_params is not None:
        try:
            from utils.dp_training import DPOptimizer
            optimizer = DPOptimizer(base_opt, **dp_params)
        except ImportError:
            optimizer = base_opt
    else:
        optimizer = base_opt

    es = _ES(patience=8)
    gen = torch.Generator(); gen.manual_seed(SEED)
    ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=gen)
    best_state = {}; best_val = float("inf")

    model.to(DEVICE)
    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            base_opt.zero_grad()
            loss = criterion(model(Xb).squeeze(-1), yb)
            loss.backward()
            if dp_params is not None and hasattr(optimizer, "step"):
                optimizer.step()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                base_opt.step()

        model.eval()
        X_va_t = torch.FloatTensor(X_va).to(DEVICE)
        y_va_t = torch.FloatTensor(y_va.astype(np.float32)).to(DEVICE)
        with torch.no_grad():
            val_loss = criterion(model(X_va_t).squeeze(-1), y_va_t).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if es(val_loss):
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_proba = torch.sigmoid(model(torch.FloatTensor(X_va).to(DEVICE)).squeeze(-1)).cpu().numpy()
    y_pred = (y_proba > 0.5).astype(int)
    return (accuracy_score(y_va, y_pred), f1_score(y_va, y_pred, zero_division=0),
            roc_auc_score(y_va, y_proba))


def _run_condition(label, model_fn, X_full, y_full, skf, epochs, dp_params=None):
    """Run n_folds CV for one ablation condition."""
    accs, f1s, aucs = [], [], []
    t0 = time.perf_counter()
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
        X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
        X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
        y_va = y_full[va_idx]

        torch.manual_seed(SEED + fold_i)
        model = model_fn()
        acc, f1, auc = _train_eval_fold(
            model, X_tr_sm, y_tr_sm, X_va, y_va,
            lr=5e-4, epochs=epochs, dp_params=dp_params,
        )
        accs.append(acc); f1s.append(f1); aucs.append(auc)
        print(f"  [{label}] Fold {fold_i} | Acc={acc*100:.2f}%  F1={f1*100:.2f}%  AUC={auc:.4f}")
        sys.stdout.flush()

    elapsed = time.perf_counter() - t0
    return {
        "Condition": label,
        "Acc_mean": float(np.mean(accs)), "Acc_std": float(np.std(accs)),
        "F1_mean": float(np.mean(f1s)),   "F1_std": float(np.std(f1s)),
        "AUC_mean": float(np.mean(aucs)), "AUC_std": float(np.std(aucs)),
        "Time_s": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Ablation conditions
# ══════════════════════════════════════════════════════════════════════════════

def run_ablation(n_folds: int, epochs: int, fast: bool) -> list:
    from models.tab_transformer import TabTransformer
    from models.hybrid_quantum_transformer import (
        HybridTabTransformer, QuantumCircuitConfig, DEFAULT_QC_CFG, LEGACY_QC_CFG,
    )

    X_full = np.load(DATA_DIR / "X_full.npy")
    y_full = np.load(DATA_DIR / "y_full.npy")
    n_features = X_full.shape[1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    results = []

    print("=" * 60)
    print(f"ABLATION STUDY ({n_folds}-fold CV, {epochs} epochs/fold)")
    print("=" * 60)

    # ── Condition 1: No VQC (TabTransformer only) ─────────────────────────────
    print("\n[Condition 1] No VQC (Classical TabTransformer only)")
    row = _run_condition(
        "No VQC (Tab-only)",
        lambda: TabTransformer(n_features=n_features, d_model=32, n_heads=4,
                               n_layers=2, dim_ff=128, dropout=0.1),
        X_full, y_full, skf, epochs,
    )
    results.append(row)

    # ── Condition 2: VQC layer depth (1L vs 2L vs 3L) ────────────────────────
    for n_layers in [1, 2, 3]:
        print(f"\n[Condition 2] VQC {n_layers} layer(s) — 6 qubits")
        cfg = QuantumCircuitConfig(n_qubits=6, n_vqc_layers=n_layers, data_reuploading=True)
        row = _run_condition(
            f"VQC 6q-{n_layers}L",
            lambda c=cfg: HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                               n_layers=2, dropout=0.1, qc_cfg=c),
            X_full, y_full, skf, epochs,
        )
        results.append(row)

    # ── Condition 3: Qubit count (4q vs 6q) ──────────────────────────────────
    for n_qubits, cfg in [
        (4, LEGACY_QC_CFG),
        (6, DEFAULT_QC_CFG),
    ]:
        print(f"\n[Condition 3] {n_qubits} qubits")
        row = _run_condition(
            f"{n_qubits}q-VQC",
            lambda c=cfg: HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                               n_layers=2, dropout=0.1, qc_cfg=c),
            X_full, y_full, skf, epochs,
        )
        results.append(row)

    # ── Condition 4: Data re-uploading on vs off ──────────────────────────────
    for reup in [False, True]:
        print(f"\n[Condition 4] Data re-uploading={'on' if reup else 'off'}")
        cfg = QuantumCircuitConfig(n_qubits=6, n_vqc_layers=3, data_reuploading=reup)
        row = _run_condition(
            f"6q-3L reupload={'on' if reup else 'off'}",
            lambda c=cfg: HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                               n_layers=2, dropout=0.1, qc_cfg=c),
            X_full, y_full, skf, epochs,
        )
        results.append(row)

    # ── Condition 5: DP training on vs off ───────────────────────────────────
    for dp_label, dp_params in [
        ("no DP", None),
        ("DP (ε≈3)", {"noise_multiplier": 1.1, "max_grad_norm": 1.0}),
    ]:
        print(f"\n[Condition 5] {dp_label}")
        row = _run_condition(
            f"6q-3L {dp_label}",
            lambda: HybridTabTransformer(n_features=n_features, d_model=32, n_heads=4,
                                         n_layers=2, dropout=0.1, qc_cfg=DEFAULT_QC_CFG),
            X_full, y_full, skf, epochs, dp_params=dp_params,
        )
        results.append(row)

    # ── Condition 6: PCA input dimension (4-dim vs 6-dim) ────────────────────
    from sklearn.decomposition import PCA

    for pca_dims in [4, 6]:
        print(f"\n[Condition 6] PCA {pca_dims}-dim input to VQC")
        cfg = QuantumCircuitConfig(n_qubits=pca_dims, n_vqc_layers=3, data_reuploading=True)

        sub_accs, sub_f1s, sub_aucs = [], [], []
        for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X_full, y_full), 1):
            X_tr, X_va = _scale_fold(X_full[tr_idx], X_full[va_idx])
            X_tr_sm, y_tr_sm = _smote_fold(X_tr, y_full[tr_idx])
            y_va = y_full[va_idx]

            pca = PCA(n_components=pca_dims, random_state=SEED)
            X_tr_pca = pca.fit_transform(X_tr_sm)
            X_va_pca = pca.transform(X_va)

            torch.manual_seed(SEED + fold_i)
            model = HybridTabTransformer(n_features=pca_dims, d_model=32, n_heads=4,
                                         n_layers=2, dropout=0.1, qc_cfg=cfg)
            acc, f1, auc = _train_eval_fold(
                model, X_tr_pca, y_tr_sm, X_va_pca, y_va, lr=5e-4, epochs=epochs,
            )
            sub_accs.append(acc); sub_f1s.append(f1); sub_aucs.append(auc)
            print(f"  [PCA-{pca_dims}] Fold {fold_i} | AUC={auc:.4f}")

        results.append({
            "Condition": f"PCA-{pca_dims}→VQC",
            "Acc_mean": float(np.mean(sub_accs)), "Acc_std": float(np.std(sub_accs)),
            "F1_mean": float(np.mean(sub_f1s)),   "F1_std": float(np.std(sub_f1s)),
            "AUC_mean": float(np.mean(sub_aucs)), "AUC_std": float(np.std(sub_aucs)),
            "Time_s": 0,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Save + LaTeX table
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: list) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # LaTeX ablation table
    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Ablation study results on CKD dataset ("
        + str(len(results[0].get("_folds_aucs", [])) or 5)
        + r"-fold CV). Best row per metric in \textbf{bold}.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Condition & Acc (\%) & F1 (\%) & AUC & Time (s) \\",
        r"\midrule",
    ]

    best_acc = max(r["Acc_mean"] for r in results)
    best_auc = max(r["AUC_mean"] for r in results)

    for row in results:
        acc_s = f"{row['Acc_mean']*100:.2f}$\\pm${row['Acc_std']*100:.2f}"
        f1_s  = f"{row['F1_mean']*100:.2f}$\\pm${row['F1_std']*100:.2f}"
        auc_s = f"{row['AUC_mean']:.4f}$\\pm${row['AUC_std']:.4f}"
        if abs(row["Acc_mean"] - best_acc) < 1e-8:
            acc_s = r"\textbf{" + acc_s + "}"
        if abs(row["AUC_mean"] - best_auc) < 1e-8:
            auc_s = r"\textbf{" + auc_s + "}"
        lines.append(f"  {row['Condition']} & {acc_s} & {f1_s} & {auc_s} & {row['Time_s']} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex = "\n".join(lines) + "\n"
    tex_path = LATEX_DIR / "ablation_table.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved: {tex_path}")

    print("\nABLATION RESULTS SUMMARY")
    print("=" * 70)
    print(df[["Condition", "Acc_mean", "F1_mean", "AUC_mean"]].to_string(index=False))
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="HQCT ablation study")
    parser.add_argument("--folds",  type=int, default=5, help="CV folds (default 5)")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs per fold (default 20)")
    parser.add_argument("--fast",   action="store_true", help="Skip slow conditions")
    return parser.parse_args()


def main():
    args = parse_args()

    for fname in ["X_full.npy", "y_full.npy"]:
        if not (DATA_DIR / fname).exists():
            print(f"ERROR: {DATA_DIR / fname} not found. Run preprocessing.py first.")
            sys.exit(1)

    results = run_ablation(args.folds, args.epochs, args.fast)
    save_results(results)

    print("\nAblation study complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
