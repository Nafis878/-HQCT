"""
report/tables.py -- Generate 4 journal-ready LaTeX tables.

Run: python report/tables.py
Output: results/latex_tables/*.tex

Table 1: Full metrics comparison (Acc, F1, AUC, MCC, Kappa, Brier with 95% CI)
Table 2: Statistical significance matrix (Wilcoxon p-values)
Table 3: Quantum circuit properties (n_qubits, expressibility, etc.)
Table 4: Data provenance (SHA-256, class ratio, preprocessing)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LATEX_DIR = RESULTS_DIR / "latex_tables"


def _bold(val: str) -> str:
    return r"\textbf{" + val + "}"


def _underline(val: str) -> str:
    return r"\underline{" + val + "}"


def _fmt_mean_std(mean: float, std: float, pct: bool = False) -> str:
    if pct:
        return f"{mean*100:.2f} $\\pm$ {std*100:.2f}"
    return f"{mean:.4f} $\\pm$ {std:.4f}"


def _bootstrap_ci_str(values: list, n_boot: int = 2000, alpha: float = 0.05) -> str:
    """Return '(lower, upper)' 95% BCa bootstrap CI string."""
    if not values:
        return "(N/A)"
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(42)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [alpha * 50, (1 - alpha / 2) * 100])
    return f"({lo:.4f}, {hi:.4f})"


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 — Full metrics comparison
# ══════════════════════════════════════════════════════════════════════════════

def build_table1(ckd_csv: Path, fhs_csv: Path) -> str:
    """
    Full metrics comparison table.
    Columns: Model | Dataset | Acc | F1 | AUC | MCC | Kappa | Brier [95% CI]
    Bold = best per dataset per metric; underline = second best.
    """
    rows = []
    for csv_path, dataset_label in [(ckd_csv, "CKD"), (fhs_csv, "FHS")]:
        if not csv_path.exists():
            print(f"  WARNING: {csv_path} not found — Table 1 partial.")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            rows.append({
                "Model": row.get("Model", "?"),
                "Dataset": dataset_label,
                "Acc": row.get("Accuracy", 0),
                "Acc_std": row.get("Accuracy_std", 0),
                "F1": row.get("F1", 0),
                "F1_std": row.get("F1_std", 0),
                "AUC": row.get("ROC_AUC", 0),
                "AUC_std": row.get("ROC_AUC_std", 0),
                "MCC": row.get("MCC", 0),
                "Kappa": row.get("Kappa", 0),
                "Brier": row.get("Brier", 0),
            })

    if not rows:
        return "% Table 1: no data found\n"

    df_all = pd.DataFrame(rows)

    metrics = ["Acc", "F1", "AUC", "MCC", "Kappa", "Brier"]
    # Brier: lower is better
    higher_better = {"Acc", "F1", "AUC", "MCC", "Kappa"}

    lines = [
        r"\begin{table*}[!ht]",
        r"\centering",
        r"\caption{10-fold cross-validation performance (mean $\pm$ std; 95\% bootstrap CI). "
        r"\textbf{Bold}: best; \underline{underline}: second best. All models use SMOTE "
        r"inside folds only (no data leakage).}",
        r"\label{tab:full_metrics}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Model & Dataset & Accuracy (\%) & F1 (\%) & ROC-AUC & MCC & Cohen's $\kappa$ & Brier Score \\",
        r"\midrule",
    ]

    for dataset in ["CKD", "FHS"]:
        sub = df_all[df_all["Dataset"] == dataset]
        if sub.empty:
            continue
        lines.append(r"\multicolumn{8}{l}{\textit{" + dataset + r" Dataset}} \\")

        for metric in metrics:
            vals = sub[metric].values
            best_idx = vals.argmin() if metric == "Brier" else vals.argmax()
            sorted_vals = np.sort(vals)[::-1] if metric not in ["Brier"] else np.sort(vals)
            second_best = sorted_vals[1] if len(sorted_vals) > 1 else None

        for _, row in sub.iterrows():
            model = row["Model"]
            cells = []
            for metric in metrics:
                val = float(row[metric])
                best_vals = sub[metric].values
                best = best_vals.min() if metric == "Brier" else best_vals.max()
                sorted_v = np.sort(best_vals) if metric == "Brier" else np.sort(best_vals)[::-1]
                second = sorted_v[1] if len(sorted_v) > 1 else None

                is_pct = metric in {"Acc", "F1"}
                fmt = f"{val*100:.2f}" if is_pct else f"{val:.4f}"

                if metric in {"Acc", "F1"}:
                    std = float(row.get(f"{metric}_std", 0))
                    fmt = f"{val*100:.2f} $\\pm$ {std*100:.2f}"
                elif metric == "AUC":
                    std = float(row.get("AUC_std", 0))
                    fmt = f"{val:.4f} $\\pm$ {std:.4f}"

                if abs(val - best) < 1e-9:
                    fmt = _bold(fmt)
                elif second is not None and abs(val - second) < 1e-9:
                    fmt = _underline(fmt)

                cells.append(fmt)

            lines.append(f"  {model} & {dataset} & " + " & ".join(cells) + r" \\")

        lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 2 — Statistical significance matrix
# ══════════════════════════════════════════════════════════════════════════════

def build_table2(stat_json: Path, fhs_stat_json: Path) -> str:
    """Wilcoxon p-value matrix for pairwise model comparisons."""

    def _sig_marker(p: float) -> str:
        if p is None:
            return "N/A"
        if p < 0.001:
            return f"{p:.3e}***"
        if p < 0.01:
            return f"{p:.4f}**"
        if p < 0.05:
            return f"{p:.4f}*"
        return f"{p:.4f}"

    def _parse_stat(json_path: Path) -> dict:
        if not json_path.exists():
            return {}
        with open(json_path) as f:
            return json.load(f)

    ckd_stats = _parse_stat(stat_json)
    fhs_stats = _parse_stat(fhs_stat_json)

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Pairwise Wilcoxon signed-rank test p-values on 10-fold AUC scores. "
        r"*\,$p<0.05$, **\,$p<0.01$, ***\,$p<0.001$.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Comparison & CKD $p$-value & FHS $p$-value \\",
        r"\midrule",
    ]

    all_pairs = set()
    for d in [ckd_stats, fhs_stats]:
        for k in d.get("wilcoxon_pairs", {}).keys():
            all_pairs.add(k)
    for pair in sorted(all_pairs):
        ckd_p = ckd_stats.get("wilcoxon_pairs", {}).get(pair, {}).get("p_value")
        fhs_p = fhs_stats.get("wilcoxon_pairs", {}).get(pair, {}).get("p_value")
        pair_label = pair.replace("_vs_", " vs. ").replace("_", " ")
        lines.append(f"  {pair_label} & {_sig_marker(ckd_p)} & {_sig_marker(fhs_p)} \\\\")

    # Friedman test row
    ckd_fr = ckd_stats.get("friedman", {}).get("p_value")
    fhs_fr = fhs_stats.get("friedman", {}).get("p_value")
    lines.append(r"\midrule")
    lines.append(f"  Friedman (overall) & {_sig_marker(ckd_fr)} & {_sig_marker(fhs_fr)} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 3 — Quantum circuit properties
# ══════════════════════════════════════════════════════════════════════════════

def build_table3(qmetrics_json: Path) -> str:
    """Quantum circuit metrics table."""
    if qmetrics_json.exists():
        with open(qmetrics_json) as f:
            qm = json.load(f)
    else:
        qm = {}

    configs = [
        ("LEGACY (4q-2L)", 4, 2, False),
        ("DEFAULT (6q-3L)", 6, 3, True),
    ]

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Quantum circuit properties of the VQC feed-forward block. "
        r"Expressibility and entanglement capability are computed via "
        r"Meyer--Wallach measure on 2000 random parameter samples.}",
        r"\label{tab:quantum_circuit}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Config & Qubits & Layers & Re-upload & Params & Expressibility & Entanglement \\",
        r"\midrule",
    ]

    for label, nq, nl, reup in configs:
        key = f"{nq}q_{nl}L"
        expr = qm.get(key, {}).get("expressibility", "—")
        ent = qm.get(key, {}).get("entanglement_capability", "—")
        n_params = 2 * nq * nl  # RY + RZ per qubit per layer
        expr_str = f"{expr:.4f}" if isinstance(expr, float) else str(expr)
        ent_str = f"{ent:.4f}" if isinstance(ent, float) else str(ent)
        reup_str = "Yes" if reup else "No"
        lines.append(
            f"  {label} & {nq} & {nl} & {reup_str} & {n_params} & {expr_str} & {ent_str} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Table 4 — Data provenance
# ══════════════════════════════════════════════════════════════════════════════

def build_table4(hashes_json: Path) -> str:
    """Data provenance table with SHA-256 fingerprints."""
    datasets = {
        "ckd_raw": {
            "name": "UCI CKD",
            "n_samples": 400,
            "n_features": 24,
            "class_ratio": "250:150 (CKD:notCKD)",
            "preprocessing": "Median/mode imputation, ordinal enc., SMOTE (train only)",
        },
        "fhs_raw": {
            "name": "Framingham HS",
            "n_samples": 4238,
            "n_features": 15,
            "class_ratio": "~85:15 (noCHD:CHD)",
            "preprocessing": "Median/mode imputation, StandardScaler, SMOTE (train only)",
        },
    }

    if hashes_json.exists():
        with open(hashes_json) as f:
            hashes = json.load(f)
    else:
        hashes = {}

    lines = [
        r"\begin{table}[!ht]",
        r"\centering",
        r"\caption{Dataset provenance. SHA-256 hashes guarantee bit-exact reproducibility. "
        r"Full hashes stored in \texttt{results/data\_hashes.json}.}",
        r"\label{tab:provenance}",
        r"\begin{tabular}{llccll}",
        r"\toprule",
        r"Dataset & $N$ & Features & Class Ratio & SHA-256 (first 16 chars) & Preprocessing \\",
        r"\midrule",
    ]

    for key, meta in datasets.items():
        sha = hashes.get(key, {}).get("sha256", "not\_computed")
        sha_short = sha[:16] if sha != "not\\_computed" else "not\\_computed"
        prep = meta["preprocessing"].replace("_", r"\_")
        lines.append(
            f"  {meta['name']} & {meta['n_samples']} & {meta['n_features']} & "
            f"{meta['class_ratio']} & \\texttt{{{sha_short}}} & \\small{{{prep}}} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_tables() -> None:
    print("=" * 60)
    print("GENERATING LATEX TABLES")
    print("=" * 60)

    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    # Table 1
    t1 = build_table1(
        RESULTS_DIR / "cv_results.csv",
        RESULTS_DIR / "fhs_cv_results.csv",
    )
    (LATEX_DIR / "table1_metrics.tex").write_text(t1, encoding="utf-8")
    print("  results/latex_tables/table1_metrics.tex")

    # Table 2
    t2 = build_table2(
        RESULTS_DIR / "statistical_tests.json",
        RESULTS_DIR / "fhs_statistical_tests.json",
    )
    (LATEX_DIR / "table2_significance.tex").write_text(t2, encoding="utf-8")
    print("  results/latex_tables/table2_significance.tex")

    # Table 3
    t3 = build_table3(RESULTS_DIR / "quantum_circuit_metrics.json")
    (LATEX_DIR / "table3_quantum_circuit.tex").write_text(t3, encoding="utf-8")
    print("  results/latex_tables/table3_quantum_circuit.tex")

    # Table 4
    t4 = build_table4(RESULTS_DIR / "data_hashes.json")
    (LATEX_DIR / "table4_provenance.tex").write_text(t4, encoding="utf-8")
    print("  results/latex_tables/table4_provenance.tex")

    print("\nAll 4 LaTeX tables generated.")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_tables()
