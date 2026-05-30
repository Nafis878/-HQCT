"""
scripts/build_submission.py -- Assemble a portable submission/ package.

Copies REAL repo artifacts (figures, tables, manuscript sections, supplementary
data), writes a cover letter (honest claims), a data-availability statement with
SHA-256 auto-filled from results/data_hashes.json, a zipped code archive, and a
manifest. Target journal: Expert Systems with Applications.

Run: python scripts/build_submission.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
LATEX = RESULTS / "latex_tables"
SUB = BASE / "submission"

GITHUB_URL = "https://github.com/Nafis878/-HQCT"
AUTHOR = "Nafis"


def _fresh_dir(p: Path) -> None:
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _copy_glob(patterns, dest: Path) -> int:
    n = 0
    for pat in patterns:
        for f in glob.glob(str(pat)):
            shutil.copy(f, dest)
            n += 1
    return n


# ── Supplementary: bootstrap CIs from the per-dataset stat-test JSONs ──────────

def _build_bootstrap_csv(dest: Path) -> None:
    import pandas as pd
    rows = []
    for ds, fn in [("CKD", "statistical_tests.json"), ("FHS", "fhs_statistical_tests.json"),
                   ("PIMA", "pima_statistical_tests.json"), ("Cleveland", "cleveland_statistical_tests.json")]:
        p = RESULTS / fn
        if not p.exists():
            continue
        data = json.loads(p.read_text())
        for model, ci in data.get("bootstrap_ci", {}).items():
            rows.append({
                "Dataset": ds, "Model": model,
                "metric": ci.get("metric", "roc_auc"),
                "mean": ci.get("mean"),
                "ci_95_lower": ci.get("ci_95_lower"),
                "ci_95_upper": ci.get("ci_95_upper"),
                "std": ci.get("std"),
            })
    if rows:
        pd.DataFrame(rows).to_csv(dest / "bootstrap_confidence_intervals.csv", index=False)
        print(f"  bootstrap_confidence_intervals.csv ({len(rows)} rows)")


# ── Cover letter ───────────────────────────────────────────────────────────────

COVER_BODY = r"""We submit for your consideration our manuscript titled
\textbf{``A Data-Adaptive Hybrid Quantum-Classical Transformer for Medical
Tabular Classification''} for publication in \textit{Expert Systems with
Applications}.

\textbf{Summary of contribution.}
We integrate a variational quantum circuit (VQC) into the feed-forward sublayer
of a TabTransformer encoder, and select the VQC complexity adaptively per
cross-validation fold from the training-set size. Evaluated on four medical
datasets (UCI CKD, PIMA Diabetes, Cleveland Heart Disease, Framingham Heart
Study) via 10-fold stratified cross-validation against XGBoost, LightGBM,
TabTransformer, and an MLP, the model attains the best Matthews correlation,
tied-best accuracy, and the lowest cross-fold variance on the Cleveland Heart
Disease dataset, and ranks second of five by F1 on the deliberately hard,
class-imbalanced Framingham benchmark. We report results honestly: on the
near-saturated CKD and on PIMA the model is competitive but does not lead, and
we state this plainly rather than over-claiming.

\textbf{Novel contributions.}
\begin{itemize}
  \item A data-adaptive circuit selector that scales VQC depth/width to the
        available data, with an ablation showing that on small datasets a
        shallower circuit outperforms the deeper one -- a data-regime sensitivity
        rarely reported for hybrid quantum models.
  \item Quantitative circuit characterisation -- Meyer--Wallach expressibility
        (0.995--0.9996) and entanglement capability (0.84--0.95), plus kernel
        target alignment (reported even though the quantum kernel does not beat
        an RBF kernel here, in the interest of honest benchmarking).
  \item Quantum gradient feature attribution complementing classical SHAP.
  \item Full SHA-256 cryptographic provenance for datasets and model artifacts,
        with a 45-test automated verification suite and a one-command sanity check.
\end{itemize}

\textbf{Reproducibility.}
All code, preprocessing pipelines, and results are public at
\url{%s}. Running \texttt{python scripts/sanity\_check.py} verifies all reported
results against SHA-256-signed artifacts in under two minutes.

\textbf{Scope fit.}
The work sits at the intersection of quantum machine learning and clinical
decision support -- an emerging area aligned with ESWA's focus on intelligent
systems for real-world applications.

We confirm this manuscript is original, not under consideration elsewhere, and
that all authors have approved the submission.""" % GITHUB_URL


def _write_cover_letter(dest: Path) -> None:
    tex = (
        "\\documentclass[12pt]{letter}\n"
        "\\usepackage{geometry}\\geometry{margin=1in}\n"
        "\\usepackage{enumitem}\\usepackage{hyperref}\n\n"
        "\\begin{document}\n"
        "\\begin{letter}{Editor-in-Chief\\\\Expert Systems with Applications\\\\Elsevier}\n\n"
        "\\opening{Dear Editor,}\n\n"
        + COVER_BODY +
        "\n\n\\closing{Sincerely,}\n\n"
        f"{AUTHOR}\\\\\n[Institution]\\\\\n[Email]\\\\\n[Date]\n\n"
        "\\end{letter}\n\\end{document}\n"
    )
    (dest / "cover_letter.tex").write_text(tex, encoding="utf-8")

    # Plain-text version (strip the most common LaTeX markup)
    import re
    txt = COVER_BODY
    txt = txt.replace(r"\textbf{", "").replace(r"\textit{", "").replace(r"\emph{", "")
    txt = txt.replace(r"\begin{itemize}", "").replace(r"\end{itemize}", "")
    txt = txt.replace(r"\item", "  -").replace(r"\url{", "").replace("}", "")
    txt = txt.replace("``", '"').replace("''", '"').replace("--", "-").replace(r"\\", "")
    txt = re.sub(r"[ \t]+\n", "\n", txt)
    plain = (
        "Cover Letter -- Expert Systems with Applications\n"
        "Editor-in-Chief, Expert Systems with Applications, Elsevier\n\n"
        "Dear Editor,\n\n" + txt + "\n\nSincerely,\n"
        f"{AUTHOR}\n[Institution]\n[Email]\n[Date]\n"
    )
    (dest / "cover_letter.txt").write_text(plain, encoding="utf-8")
    print("  cover_letter.tex + cover_letter.txt  (fill [Institution]/[Email]/[Date])")


# ── Data availability statement ────────────────────────────────────────────────

def _write_data_availability(dest: Path) -> None:
    hashes = json.loads((RESULTS / "data_hashes.json").read_text()) if (RESULTS / "data_hashes.json").exists() else {}

    def sha(key):
        return hashes.get(key, {}).get("sha256", "(not available)")

    text = f"""DATA AVAILABILITY STATEMENT

All datasets used in this study are publicly available. SHA-256 hashes of the
exact raw files used are given for bit-level verification (see also
results/data_hashes.json).

1. UCI Chronic Kidney Disease
   UCI ML Repository (id=336), auto-downloaded by preprocessing.py
   SHA-256: {sha('ckd_raw')}

2. PIMA Indians Diabetes
   Public mirror: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
   SHA-256: {sha('pima_raw')}

3. Cleveland Heart Disease
   UCI ML Repository: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
   SHA-256: {sha('cleveland_raw')}

4. Framingham Heart Study
   Kaggle (public): https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset
   SHA-256: {sha('fhs_raw')}

All preprocessing pipelines, results, and SHA-256-signed model provenance are
available at: {GITHUB_URL} (tagged release: v1.0-submission)

Independent verification (no retraining, ~2 minutes):
  python scripts/sanity_check.py
Expected: ALL CHECKS PASSED
"""
    (dest / "data_availability_statement.txt").write_text(text, encoding="utf-8")
    print("  data_availability_statement.txt  (SHA-256 auto-filled)")


# ── Code archive ────────────────────────────────────────────────────────────────

def _build_code_archive(dest: Path) -> None:
    patterns = [
        "main.py", "main_fhs.py", "config.py",
        "preprocessing.py", "fhs_preprocessing.py",
        "pima_preprocessing.py", "cleveland_preprocessing.py",
        "cv_evaluation.py", "fhs_cv_evaluation.py",
        "pima_cv_evaluation.py", "cleveland_cv_evaluation.py",
        "ablation_study.py",
        "models/*.py", "utils/*.py", "report/*.py", "scripts/*.py", "tests/*.py",
        "requirements.txt", "environment.yml", "Dockerfile", "setup.py",
        "README.md", "REPRODUCIBILITY.md", "CONTRIBUTING.md",
        "results/data_hashes.json", "results/provenance_log.json",
        "results/quantum_circuit_metrics.json", "results/quantum_advantage.json",
        "results/calibration_metrics.json", "results/ablation_results.csv",
    ]
    archive = dest / "HQCT_code_v1.0.zip"
    count = 0
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for pat in patterns:
            for fp in glob.glob(str(BASE / pat)):
                rel = Path(fp).relative_to(BASE)
                zf.write(fp, arcname=str(rel))
                count += 1
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dest / "archive_sha256.txt").write_text(
        f"HQCT_code_v1.0.zip\nSHA-256: {sha}\nfiles: {count}\n", encoding="utf-8")
    print(f"  HQCT_code_v1.0.zip ({count} files)  sha256={sha[:16]}...")


# ── Manifest ─────────────────────────────────────────────────────────────────--

def _write_manifest() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    fig_map = [
        ("fig1_architecture", "Fig. 1", "HQCT architecture (6-qubit HEA feed-forward)"),
        ("fig2_roc_curves", "Fig. 2", "ROC curves (pooled out-of-fold), 4 datasets"),
        ("fig3_pr_curves", "Fig. 3", "Multi-metric comparison, 4 datasets"),
        ("fig4_critical_difference", "Fig. 4", "Model ranking by AUC, per dataset"),
        ("fig5_quantum_scatter", "Fig. 5", "Measured expressibility vs entanglement"),
        ("fig6_barren_plateau", "Fig. 6", "Trainability / convergence (illustrative)"),
        ("fig7_shap_summary", "Fig. 7", "XGBoost TreeSHAP feature importance"),
        ("fig8_calibration", "Fig. 8", "Reliability diagrams + ECE"),
    ]
    fig_rows = "\n".join(f"| {n}.pdf/.png | {num} | {cap} |" for n, num, cap in fig_map)
    md = f"""# HQCT Submission Package Manifest
Generated: {ts}
Target: Expert Systems with Applications (editorialmanager.com/eswa)
Manuscript type: Research Paper
Repository: {GITHUB_URL} (tag v1.0-submission)

> Honest positioning: HybridQT leads on Cleveland (best MCC, tied-best accuracy,
> lowest variance) and is 2nd of 5 by F1 on the hard, imbalanced FHS; it is
> competitive but not best on the near-saturated CKD and on PIMA. All numbers
> trace to results/*.csv and *.json.

## figures/  (8 main figures; submit as separate files)
| File | Figure | Caption |
|------|--------|---------|
{fig_rows}
(Plus shap_summary_XGBoost_*, barren_plateau, calibration_* as supporting PDFs.)

## tables/
| File | Table | Content |
|------|-------|---------|
| table1_metrics.tex | Table 1 | Acc/F1/Sens/Spec/AUC/MCC/Kappa/Brier, 5 models x 4 datasets |
| table2_significance.tex | Table 2 | Pairwise Wilcoxon + Friedman per dataset |
| table3_quantum_circuit.tex | Table 3 | Expressibility, entanglement, params (3 configs) |
| table4_provenance.tex | Table 4 | Dataset SHA-256 + preprocessing |
| ablation_table.tex | Table 5 | 6-condition ablation |

## manuscript/   (LaTeX fragments — \\input into your main .tex)
abstract.tex, methods_quantum.tex, results_table.tex, reproducibility.tex, fhs_context.tex

## supplementary/
full_metrics_{{ckd,pima,cleveland,fhs}}.csv, bootstrap_confidence_intervals.csv,
quantum_circuit_metrics.json, quantum_advantage.json, calibration_metrics.json,
ablation_results.csv, *_mcnemar_detail.json

## cover_letter/   cover_letter.tex + cover_letter.txt
## data_availability/   data_availability_statement.txt (SHA-256 filled)
## code_archive/   HQCT_code_v1.0.zip + archive_sha256.txt

## Submission checklist
- [ ] Personalise cover letter ([Institution], [Email], [Date])
- [ ] Compile manuscript (pdflatex unavailable in build env — compile locally)
- [ ] Confirm Table 1 shows Sens/Spec columns
- [ ] Tag pushed: v1.0-submission  •  sanity_check.py = ALL CHECKS PASSED
- [ ] Keywords: quantum machine learning, variational quantum circuit,
      TabTransformer, medical classification, hybrid quantum-classical,
      explainable AI, SHAP
"""
    (SUB / "SUBMISSION_MANIFEST.md").write_text(md, encoding="utf-8")
    print("  SUBMISSION_MANIFEST.md")


def main() -> None:
    _fresh_dir(SUB)
    for sub in ["manuscript", "figures", "tables", "supplementary",
                "cover_letter", "data_availability", "code_archive"]:
        (SUB / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BUILDING SUBMISSION PACKAGE")
    print("=" * 60)

    nfig = _copy_glob([FIGS / "*.pdf", FIGS / "*.png"], SUB / "figures")
    print(f"  figures/: {nfig} files")
    ntab = _copy_glob([LATEX / "table*_*.tex", LATEX / "ablation_table.tex"], SUB / "tables")
    print(f"  tables/: {ntab} files")
    _copy_glob([LATEX / "abstract.tex", LATEX / "methods_quantum.tex",
                LATEX / "results_table.tex", LATEX / "reproducibility.tex",
                LATEX / "fhs_context.tex"], SUB / "manuscript")
    print("  manuscript/: paper sections")

    sup = SUB / "supplementary"
    _copy_glob([RESULTS / "full_metrics_*.csv",
                RESULTS / "quantum_circuit_metrics.json", RESULTS / "quantum_advantage.json",
                RESULTS / "calibration_metrics.json", RESULTS / "ablation_results.csv",
                RESULTS / "*_mcnemar_detail.json", RESULTS / "mcnemar_detail.json"], sup)
    _build_bootstrap_csv(sup)
    print("  supplementary/: metrics CSVs + JSONs")

    _write_cover_letter(SUB / "cover_letter")
    _write_data_availability(SUB / "data_availability")
    _build_code_archive(SUB / "code_archive")
    _write_manifest()

    # Summary
    total = sum(f.stat().st_size for f in SUB.rglob("*") if f.is_file())
    nfiles = sum(1 for _ in SUB.rglob("*") if _.is_file())
    print("\n" + "=" * 48)
    print("  SUBMISSION PACKAGE COMPLETE")
    print(f"  Location : submission/")
    print(f"  Files    : {nfiles}")
    print(f"  Size     : {total/1024/1024:.1f} MB")
    print(f"  Target   : Expert Systems with Applications")
    print("=" * 48)


if __name__ == "__main__":
    main()
