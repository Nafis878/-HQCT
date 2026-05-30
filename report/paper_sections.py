"""
report/paper_sections.py -- Auto-generate LaTeX paper sections from actual results.

Run: python report/paper_sections.py
Output: results/latex_tables/{abstract,methods_quantum,results_table,reproducibility}.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
LATEX_DIR = RESULTS_DIR / "latex_tables"


HQCT_NAME = "Hybrid Quantum Transformer"

# dataset display name -> cv_results CSV
_DATASET_CSVS = {
    "CKD": "cv_results.csv",
    "FHS": "fhs_cv_results.csv",
    "PIMA": "pima_cv_results.csv",
    "Cleveland": "cleveland_cv_results.csv",
}


def _load_best_result(csv_path: Path, model_name: str = HQCT_NAME) -> dict:
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    row = df[df["Model"].str.contains("Hybrid", case=False, na=False)]
    if row.empty:
        row = df.sort_values("ROC_AUC", ascending=False).head(1)
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _analyze_dataset(csv_path: Path) -> dict:
    """
    Return an honest per-dataset summary of where HybridQT stands:
    its metrics, its rank by F1 / accuracy (1 = best), and whether it has the
    lowest cross-fold std (variance stability) on accuracy or F1.
    """
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns or df.empty:
        return {}
    hq = df[df["Model"].str.contains("Hybrid", case=False, na=False)]
    if hq.empty:
        return {}
    hq = hq.iloc[0]

    def rank(metric):
        if metric not in df.columns:
            return None
        order = df.sort_values(metric, ascending=False).reset_index(drop=True)
        idx = order.index[order["Model"].str.contains("Hybrid", case=False, na=False)]
        return int(idx[0]) + 1 if len(idx) else None

    def lowest_std(stdcol):
        if stdcol not in df.columns:
            return False
        return bool(df.loc[df[stdcol].idxmin(), "Model"].lower().find("hybrid") >= 0)

    return {
        "n_models": len(df),
        "acc": float(hq.get("Accuracy", 0)) * 100,
        "acc_std": float(hq.get("Accuracy_std", 0)) * 100,
        "f1": float(hq.get("F1", 0)) * 100,
        "f1_std": float(hq.get("F1_std", 0)) * 100,
        "auc": float(hq.get("ROC_AUC", 0)),
        "mcc": float(hq.get("MCC", 0)) if "MCC" in df.columns else None,
        "rank_f1": rank("F1"),
        "rank_acc": rank("Accuracy"),
        "best_acc_stability": lowest_std("Accuracy_std"),
        "best_f1_stability": lowest_std("F1_std"),
        "vqc_config": str(hq.get("VQC_Config", "")) if "VQC_Config" in df.columns else "",
    }


def generate_abstract() -> str:
    """
    Honest, data-driven abstract. Claim strength adapts to the real numbers:
    no 'outperforms on all metrics' unless the CSVs support it. Emphasises where
    HybridQT genuinely leads (e.g. variance stability, best F1/accuracy on a
    given dataset) and reports competitive parity elsewhere.
    """
    summ = {ds: _analyze_dataset(RESULTS_DIR / csv) for ds, csv in _DATASET_CSVS.items()}
    summ = {k: v for k, v in summ.items() if v}
    n_ds = len(summ)

    # Quantum circuit metrics (6q-3L expressibility/entanglement) if available
    expr = ent = None
    qpath = RESULTS_DIR / "quantum_circuit_metrics.json"
    if qpath.exists():
        try:
            qm = json.loads(qpath.read_text())
            flagship = qm.get("6q-3L") or next(iter(qm.values()))
            expr = flagship.get("expressibility")
            ent = flagship.get("entanglement_capability")
        except Exception:
            pass

    # Honest standings
    wins_f1 = [ds for ds, s in summ.items() if s.get("rank_f1") == 1]
    top3_f1 = [ds for ds, s in summ.items() if s.get("rank_f1") and s["rank_f1"] <= 3]
    stable = [ds for ds, s in summ.items()
              if s.get("best_acc_stability") or s.get("best_f1_stability")]

    # Performance clause
    if wins_f1:
        perf = (f"achieves the highest F1-score among all five models on "
                f"{_join(wins_f1)} while remaining competitive on the "
                f"remaining benchmark{'s' if n_ds - len(wins_f1) != 1 else ''}")
    elif len(top3_f1) >= max(1, n_ds // 2):
        perf = ("ranks among the top models by F1-score across the majority of "
                "benchmarks, matching strong classical baselines")
    else:
        perf = ("attains performance competitive with strong classical baselines "
                "(XGBoost, LightGBM, TabTransformer, MLP)")

    # Stability clause
    stab = ""
    if stable:
        stab = (f" Notably, HQCT exhibits the lowest cross-validation variance on "
                f"{_join(stable)}, indicating more stable generalisation under "
                f"limited and imbalanced clinical data.")

    # Per-dataset numbers
    lines = []
    for ds, s in summ.items():
        lines.append(
            f"{ds}: {s['acc']:.2f}\\% accuracy, F1 {s['f1']:.2f}\\%, "
            f"ROC-AUC {s['auc']:.4f}"
        )
    perds = "; ".join(lines) if lines else "—"

    qclause = ""
    if expr is not None and ent is not None:
        qclause = (f" Circuit characterisation gives a Meyer--Wallach "
                   f"expressibility of {expr:.3f} and entanglement capability "
                   f"{ent:.3f} for the 6-qubit, 3-layer ansatz.")

    ds_names = _join(list(summ.keys())) if summ else "multiple clinical datasets"

    return rf"""% Abstract — auto-generated by report/paper_sections.py (honest, data-driven)
\begin{{abstract}}
We present a Hybrid Quantum-Classical Transformer (HQCT) for clinical risk
stratification that integrates a variational quantum circuit (VQC) into the
feed-forward sublayer of a TabTransformer architecture. To match circuit
capacity to the available data, the VQC complexity is selected adaptively per
training fold (4--6 qubits, 2--3 layers of a Hardware-Efficient Ansatz with
data re-uploading). We evaluate HQCT on {n_ds} clinically validated benchmarks
({ds_names}) using 10-fold stratified cross-validation with SMOTE confined to
training folds, against XGBoost, LightGBM, TabTransformer, and MLP baselines.
HQCT {perf} ({perds}).{stab}
Statistical comparisons use McNemar's exact test, Wilcoxon signed-rank, and
DeLong AUC tests; we report effect sizes and bootstrap confidence intervals
rather than significance alone.{qclause}
To support trustworthy clinical deployment, we add SHA-256 cryptographic
provenance for all artifacts, SHAP and quantum gradient feature attribution,
calibration analysis, and optional differential-privacy training (DP-SGD).
All code, data hashes, and reproducibility scripts are provided at
\url{{[repository\_url]}}.
\end{{abstract}}
"""


def _join(items: list) -> str:
    """Oxford-style join: ['A'] -> 'A'; ['A','B'] -> 'A and B'; more -> 'A, B, and C'."""
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def generate_methods_quantum() -> str:
    return r"""% Methods: Quantum VQC description — auto-generated by report/paper_sections.py
\subsection{Variational Quantum Circuit (VQC) Feed-Forward Block}
\label{sec:vqc}

The VQC replaces the classical feed-forward sublayer in the second Transformer
encoder layer.
For an input embedding $\mathbf{h} \in \mathbb{R}^{d_\text{model}}$, a linear
projection $W_\text{in} \in \mathbb{R}^{n_q \times d_\text{model}}$ maps to
$n_q = 6$ qubit inputs, scaled to $[-\pi, \pi]$ by a $\tanh(\cdot) \times \pi$
activation.

\paragraph{Hardware-Efficient Ansatz (HEA).}
Each of the $L = 3$ variational layers applies:
\begin{enumerate}
  \item \emph{Data re-uploading}: $\text{RY}(x_i)$ applied to qubit $i$, encoding
        the input at every layer (not only the first), following the expressibility
        analysis of Schuld et al.~\cite{schuld2021effect}.
  \item \emph{Parametric rotations}: $\text{RY}(\theta^{(l)}_{i,1})$ and
        $\text{RZ}(\theta^{(l)}_{i,2})$ per qubit $i$ in layer $l$, totalling
        $2 \times n_q \times L = 36$ variational parameters.
  \item \emph{Ring CNOT entanglement}: $\text{CNOT}(i, (i{+}1) \bmod n_q)$
        for $i = 0,\ldots,n_q{-}1$.
\end{enumerate}
The circuit is implemented in PennyLane~\cite{bergholm2018pennylane} with
\texttt{default.qubit} (classical simulation) and \texttt{diff\_method="backprop"}
for seamless PyTorch gradient flow.
Expectation values $\langle Z_i \rangle$ on all $n_q$ qubits form the circuit
output $\mathbf{q} \in [-1, 1]^{n_q}$, which is mapped back to
$d_\text{model}$ dimensions via a linear layer $W_\text{out}$.

\paragraph{Expressibility and Entanglement.}
We quantify VQC capacity using the Meyer--Wallach measure~\cite{meyerwallach2002}:
expressibility $\mathcal{E} = 1 - \overline{\mathcal{F}}$ (mean pairwise state
fidelity deviation from Haar-random), and entanglement capability
$Q_\text{cap}$ averaged over 2000 random parameter samples.
Full values are reported in Table~\ref{tab:quantum_circuit}.
"""


def generate_results_table() -> str:
    """Include all 4 tables in a single LaTeX snippet."""
    includes = []
    for fname in ["table1_metrics", "table2_significance",
                  "table3_quantum_circuit", "table4_provenance"]:
        tex = LATEX_DIR / f"{fname}.tex"
        if tex.exists():
            includes.append(rf"\input{{results/latex_tables/{fname}}}")
        else:
            includes.append(f"% {fname}.tex not yet generated")

    header = "% Results tables — auto-generated by report/paper_sections.py\n"
    return header + "\n\n".join(includes) + "\n"


def generate_reproducibility() -> str:
    """Reproducibility section with SHA-256 hashes and environment info."""
    hashes = {}
    hash_path = RESULTS_DIR / "data_hashes.json"
    if hash_path.exists():
        with open(hash_path) as f:
            hashes = json.load(f)

    ckd_hash = hashes.get("ckd_raw", {}).get("sha256", "not computed")
    fhs_hash = hashes.get("fhs_raw", {}).get("sha256", "not computed")
    ckd_short = ckd_hash[:32] if len(ckd_hash) > 16 else ckd_hash
    fhs_short = fhs_hash[:32] if len(fhs_hash) > 16 else fhs_hash

    return rf"""% Reproducibility section — auto-generated by report/paper_sections.py
\section{{Reproducibility}}
\label{{sec:reproducibility}}

All experiments are fully reproducible with fixed seeds (\texttt{{SEED=42}}).
Dataset integrity is verified via SHA-256 checksums stored in
\texttt{{results/data\_hashes.json}}:

\begin{{itemize}}
  \item UCI CKD dataset: \texttt{{{ckd_short}...}}
  \item Framingham Heart Study: \texttt{{{fhs_short}...}}
\end{{itemize}}

Model provenance records (SHA-256 of weights, timestamp, library versions)
are appended to \texttt{{results/provenance\_log.json}} after each training run.
A sanity-check script (\texttt{{python scripts/sanity\_check.py}}) verifies
that all reported metrics reproduce within $\pm 0.5\%$.

\paragraph{{Environment.}}
Python 3.10, PyTorch 2.x, PennyLane 0.38, scikit-learn 1.4, XGBoost 2.x,
LightGBM 4.x. CPU-only experiments were run on [hardware description].
A \texttt{{Dockerfile}} and \texttt{{environment.yml}} are provided for
exact environment recreation.

\paragraph{{Code availability.}}
All source code, preprocessing scripts, trained model checkpoints, and
LaTeX table generators are available at \url{{[repository\_url]}}.
"""


def generate_all_sections() -> None:
    print("=" * 60)
    print("GENERATING LATEX PAPER SECTIONS")
    print("=" * 60)

    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    sections = {
        "abstract.tex": generate_abstract(),
        "methods_quantum.tex": generate_methods_quantum(),
        "results_table.tex": generate_results_table(),
        "reproducibility.tex": generate_reproducibility(),
    }

    for fname, content in sections.items():
        path = LATEX_DIR / fname
        path.write_text(content, encoding="utf-8")
        print(f"  results/latex_tables/{fname}")

    print("\nAll paper sections generated.")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_sections()
