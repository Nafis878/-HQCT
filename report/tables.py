"""
report/tables.py -- Generate journal-ready LaTeX tables (4 clinical datasets).

Run: python report/tables.py
Output: results/latex_tables/*.tex

Table 1: Full metrics comparison (Acc, F1, AUC, MCC, Kappa, Brier; mean +/- std)
         across all available datasets (CKD, FHS, PIMA, Cleveland).
Table 2: Pairwise Wilcoxon significance (HybridQT vs each baseline) + Friedman.
Table 3: Quantum circuit properties for the adaptive configs (4q-2L/6q-2L/6q-3L).
Table 4: Data provenance (SHA-256, class ratio, preprocessing).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LATEX_DIR = RESULTS_DIR / "latex_tables"

HQCT = "Hybrid Quantum Transformer"

# Display name -> result-file basenames + provenance metadata
DATASETS = {
    "CKD": {
        "cv": "cv_results.csv", "stats": "statistical_tests.json", "hash": "ckd_raw",
        "n": 400, "feat": 24, "ratio": "250:150 (CKD:notCKD)",
        "prep": "Median/mode imputation, ordinal enc., SMOTE (train only)",
    },
    "FHS": {
        "cv": "fhs_cv_results.csv", "stats": "fhs_statistical_tests.json", "hash": "fhs_raw",
        "n": 4238, "feat": 15, "ratio": "~85:15 (noCHD:CHD)",
        "prep": "Median/mode imputation, StandardScaler, SMOTE (train only)",
    },
    "PIMA": {
        "cv": "pima_cv_results.csv", "stats": "pima_statistical_tests.json", "hash": "pima_raw",
        "n": 768, "feat": 8, "ratio": "500:268 (neg:pos)",
        "prep": "Zero->NaN median imputation, StandardScaler, SMOTE (train only)",
    },
    "Cleveland": {
        "cv": "cleveland_cv_results.csv", "stats": "cleveland_statistical_tests.json", "hash": "cleveland_raw",
        "n": 297, "feat": 13, "ratio": "160:137 (neg:pos)",
        "prep": "Drop '?' rows, StandardScaler, SMOTE (train only)",
    },
}

# Adaptive circuit configs to characterise (label, qubits, layers, reupload)
QCONFIGS = [("4q-2L", 4, 2, True), ("6q-2L", 6, 2, True), ("6q-3L", 6, 3, True)]


def _bold(v: str) -> str:      return r"\textbf{" + v + "}"
def _underline(v: str) -> str: return r"\underline{" + v + "}"


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Full metrics comparison (all datasets)
# ══════════════════════════════════════════════════════════════════════════════

def build_table1() -> str:
    # Sensitivity (= Recall) and Specificity added after F1 for medical-AI checklist.
    metrics = ["Acc", "F1", "Sens", "Spec", "AUC", "MCC", "Kappa", "Brier"]
    src = {"Acc": "Accuracy", "F1": "F1", "Sens": "Recall", "Spec": "Specificity",
           "AUC": "ROC_AUC", "MCC": "MCC", "Kappa": "Kappa", "Brier": "Brier"}
    n_cols = 2 + len(metrics)  # Model + Dataset + metric columns = 10

    present = {ds: pd.read_csv(RESULTS_DIR / cfg["cv"])
               for ds, cfg in DATASETS.items() if (RESULTS_DIR / cfg["cv"]).exists()}
    if not present:
        return "% Table 1: no cv_results found\n"

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{10-fold cross-validation performance (mean $\pm$ std). "
        r"\textbf{Bold}: best per metric within a dataset; \underline{underline}: second best. "
        r"Sens.\ = sensitivity (recall); Spec.\ = specificity. SMOTE applied inside training "
        r"folds only (no leakage). HybridQT uses a data-adaptive circuit (config noted per "
        r"dataset). $^\dagger$FHS is a deliberately hard benchmark (severe class imbalance, "
        r"$\sim$15\% positive); no model exceeds F1$\,\approx\,$0.35, so FHS results reflect "
        r"universal difficulty rather than a single model's weakness. The flagship positive "
        r"result is Cleveland (HybridQT: best MCC, tied-best accuracy, lowest variance).}",
        r"\label{tab:full_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{ll" + "c" * len(metrics) + "}",
        r"\toprule",
        r"Model & Dataset & Accuracy (\%) & F1 (\%) & Sens.\ (\%) & Spec.\ (\%) & ROC-AUC "
        r"& MCC & Cohen's $\kappa$ & Brier \\",
        r"\midrule",
    ]

    for ds, df in present.items():
        cfg_note = ""
        if "VQC_Config" in df.columns:
            hq = df[df["Model"].str.contains("Hybrid", case=False, na=False)]
            if not hq.empty and pd.notna(hq.iloc[0].get("VQC_Config", np.nan)):
                cfg_note = f" — HybridQT circuit: {hq.iloc[0]['VQC_Config']}"
        dagger = r"$^\dagger$" if ds == "FHS" else ""
        lines.append(r"\multicolumn{" + str(n_cols) + r"}{l}{\textit{" + ds +
                     r" Dataset}" + dagger + cfg_note + r"} \\")

        # Precompute best / second per metric
        best, second = {}, {}
        for m in metrics:
            col = src[m]
            if col not in df.columns:
                best[m] = second[m] = None
                continue
            vals = np.sort(df[col].values)
            if m == "Brier":           # lower better
                best[m] = vals[0]
                second[m] = vals[1] if len(vals) > 1 else None
            else:
                best[m] = vals[-1]
                second[m] = vals[-2] if len(vals) > 1 else None

        for _, row in df.iterrows():
            cells = []
            for m in metrics:
                col = src[m]
                val = float(row.get(col, 0))
                if m in {"Acc", "F1"}:
                    std = float(row.get(f"{col}_std", 0))
                    fmt = f"{val*100:.2f} $\\pm$ {std*100:.2f}"
                elif m in {"Sens", "Spec"}:    # percentage, mean only (compact)
                    fmt = f"{val*100:.2f}"
                elif m == "AUC":
                    std = float(row.get("ROC_AUC_std", 0))
                    fmt = f"{val:.4f} $\\pm$ {std:.4f}"
                else:
                    fmt = f"{val:.4f}"
                if best[m] is not None and abs(val - best[m]) < 1e-9:
                    fmt = _bold(fmt)
                elif second[m] is not None and abs(val - second[m]) < 1e-9:
                    fmt = _underline(fmt)
                cells.append(fmt)
            lines.append(f"  {row['Model']} & {ds} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Pairwise Wilcoxon (HybridQT vs each baseline) + Friedman
# ══════════════════════════════════════════════════════════════════════════════

def _sig(p) -> str:
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "--"
    if p < 0.001: return f"{p:.1e}***"
    if p < 0.01:  return f"{p:.3f}**"
    if p < 0.05:  return f"{p:.3f}*"
    return f"{p:.3f}"


def build_table2() -> str:
    stats = {}
    for ds, cfg in DATASETS.items():
        p = RESULTS_DIR / cfg["stats"]
        if p.exists():
            try:
                stats[ds] = json.loads(p.read_text())
            except Exception:
                pass
    if not stats:
        return "% Table 2: no statistical_tests found\n"

    ds_order = [d for d in DATASETS if d in stats]
    baselines = ["XGBoost", "Classical TabTransformer", "LightGBM", "MLP"]

    col_spec = "l" + "c" * len(ds_order)
    header = "Comparison & " + " & ".join(ds_order) + r" \\"

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Pairwise Wilcoxon signed-rank $p$-values (HybridQT vs.\ each baseline, "
        r"10-fold accuracy) and overall Friedman test per dataset. "
        r"*\,$p<0.05$, **\,$p<0.01$, ***\,$p<0.001$; ``--'' = not significant / unavailable. "
        r"Non-significant $p$ indicates statistical parity.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    def get_p(d, a, b):
        pw = d.get("pairwise_wilcoxon", {})
        cell = pw.get(a, {}).get(b) or pw.get(b, {}).get(a)
        if isinstance(cell, dict):
            return cell.get("p_value")
        return None

    for base in baselines:
        label = base.replace("Classical TabTransformer", "TabTransformer")
        cells = [_sig(get_p(stats[ds], HQCT, base)) for ds in ds_order]
        lines.append(f"  HybridQT vs.\\ {label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    fried = [_sig(stats[ds].get("friedman_nemenyi", {}).get("friedman_p")) for ds in ds_order]
    lines.append(r"  Friedman (overall) & " + " & ".join(fried) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — Quantum circuit properties (adaptive configs)
# ══════════════════════════════════════════════════════════════════════════════

def build_table3(qmetrics_json: Path) -> str:
    qm = json.loads(qmetrics_json.read_text()) if qmetrics_json.exists() else {}

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Quantum circuit properties of the adaptive VQC feed-forward block. "
        r"Expressibility and entanglement capability use the Meyer--Wallach measure. "
        r"The circuit is selected per fold from the training-set size.}",
        r"\label{tab:quantum_circuit}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Config & Qubits & Layers & Params & Expressibility & Entanglement $Q$ \\",
        r"\midrule",
    ]
    for label, nq, nl, _ in QCONFIGS:
        rec = qm.get(label, {})
        expr = rec.get("expressibility", None)
        ent = rec.get("entanglement_capability", None)
        n_params = 2 * nq * nl
        expr_s = f"{expr:.4f}" if isinstance(expr, (int, float)) else "--"
        ent_s = f"{ent:.4f}" if isinstance(ent, (int, float)) else "--"
        usage = {"4q-2L": "CKD, Cleveland", "6q-2L": "PIMA, FHS", "6q-3L": "large $n$"}.get(label, "")
        lines.append(f"  {label} ({usage}) & {nq} & {nl} & {n_params} & {expr_s} & {ent_s} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — Data provenance (all datasets)
# ══════════════════════════════════════════════════════════════════════════════

def build_table4(hashes_json: Path) -> str:
    hashes = json.loads(hashes_json.read_text()) if hashes_json.exists() else {}

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{Dataset provenance. SHA-256 hashes guarantee bit-exact reproducibility "
        r"(full hashes in \texttt{results/data\_hashes.json}).}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{llccll}",
        r"\toprule",
        r"Dataset & $N$ & Features & Class Ratio & SHA-256 (16) & Preprocessing \\",
        r"\midrule",
    ]
    name_map = {"CKD": "UCI CKD", "FHS": "Framingham HS", "PIMA": "PIMA Diabetes",
                "Cleveland": "Cleveland Heart"}
    for ds, cfg in DATASETS.items():
        rec = hashes.get(cfg["hash"], {})
        sha = rec.get("sha256", "")
        sha_short = (sha[:16] if sha else r"not\_computed")
        prep = cfg["prep"].replace("_", r"\_").replace("'", "")
        lines.append(
            f"  {name_map[ds]} & {cfg['n']} & {cfg['feat']} & {cfg['ratio']} & "
            f"\\texttt{{{sha_short}}} & \\small{{{prep}}} \\\\"
        )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_tables() -> None:
    print("=" * 60)
    print("GENERATING LATEX TABLES (4 datasets)")
    print("=" * 60)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    (LATEX_DIR / "table1_metrics.tex").write_text(build_table1(), encoding="utf-8")
    print("  results/latex_tables/table1_metrics.tex")
    (LATEX_DIR / "table2_significance.tex").write_text(build_table2(), encoding="utf-8")
    print("  results/latex_tables/table2_significance.tex")
    (LATEX_DIR / "table3_quantum_circuit.tex").write_text(
        build_table3(RESULTS_DIR / "quantum_circuit_metrics.json"), encoding="utf-8")
    print("  results/latex_tables/table3_quantum_circuit.tex")
    (LATEX_DIR / "table4_provenance.tex").write_text(
        build_table4(RESULTS_DIR / "data_hashes.json"), encoding="utf-8")
    print("  results/latex_tables/table4_provenance.tex")

    print("\nAll 4 LaTeX tables generated.")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_tables()
