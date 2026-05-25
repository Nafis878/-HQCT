"""
main_fhs.py -- Orchestrator for the FHS dual-dataset validation pipeline.
Runs Steps FHS-1 through FHS-5 in sequence with timing.
Step FHS-6 of the QIP 2027 dual-dataset pipeline.

Usage:
  python main_fhs.py [--skip-qsvm] [--cv-epochs N]

Prerequisite: data/framingham.csv must exist before running.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent


def _banner(text: str, char: str = "=", width: int = 60) -> None:
    print(char * width)
    print(text)
    print(char * width)


def _run_step(label: str, cmd: list[str]) -> float:
    """Run a subprocess step and return elapsed seconds."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"\n  [{label} completed in {elapsed:.1f}s]")
    return elapsed


def _load_cv_summary(csv_name: str) -> str:
    """Return a short summary string from a cv_results CSV."""
    results_dir = BASE_DIR / "results"
    csv_path = results_dir / csv_name
    if not csv_path.exists():
        return "  (results not available)"
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"  {row['Model']:30s}  "
                f"Acc={row['Accuracy']*100:.2f}%+/-{row['Accuracy_std']*100:.2f}%  "
                f"F1={row['F1']*100:.2f}%+/-{row['F1_std']*100:.2f}%  "
                f"AUC={row['ROC_AUC']:.4f}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"  (error reading results: {exc})"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FHS dual-dataset validation pipeline orchestrator"
    )
    parser.add_argument("--skip-qsvm", action="store_true",
                        help="Skip QSVM in training and CV steps (recommended for quick run)")
    parser.add_argument("--cv-epochs", type=int, default=50,
                        help="Epochs per fold for neural models in CV (default: 50)")
    args = parser.parse_args()

    _banner("QIP 2027 -- FHS DUAL-DATASET VALIDATION PIPELINE")

    # Prerequisite check
    csv_path = BASE_DIR / "data" / "framingham.csv"
    if not csv_path.exists():
        print(f"\n[ERROR] data/framingham.csv not found at: {csv_path}")
        print("Please download from Kaggle:")
        print("  https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset")
        print(f"and place it at: {csv_path}")
        sys.exit(1)

    print(f"\nConfiguration:")
    print(f"  skip-qsvm  : {args.skip_qsvm}")
    print(f"  cv-epochs  : {args.cv_epochs}")
    print(f"  Python     : {sys.executable}")

    timings: dict[str, float] = {}
    pipeline_start = time.time()

    # FHS-1: Preprocessing
    timings["FHS-1 Preprocessing"] = _run_step(
        "FHS-1: PREPROCESSING",
        [sys.executable, "fhs_preprocessing.py"]
    )

    # FHS-2: Train models
    train_cmd = [sys.executable, "fhs_train_models.py"]
    if args.skip_qsvm:
        train_cmd.append("--skip-qsvm")
    timings["FHS-2 Train Models"] = _run_step("FHS-2: TRAIN MODELS", train_cmd)

    # FHS-3: Cross-validation
    cv_cmd = [sys.executable, "fhs_cv_evaluation.py", "--cv-epochs", str(args.cv_epochs)]
    if args.skip_qsvm:
        cv_cmd.append("--skip-qsvm")
    timings["FHS-3 Cross-Validation"] = _run_step("FHS-3: 10-FOLD CROSS-VALIDATION", cv_cmd)

    # FHS-4: Combined report
    timings["FHS-4 Combined Report"] = _run_step(
        "FHS-4: COMBINED REPORT",
        [sys.executable, "combined_report.py"]
    )

    # FHS-5: Combined plots
    timings["FHS-5 Combined Plots"] = _run_step(
        "FHS-5: COMBINED PLOTS",
        [sys.executable, "combined_plots.py"]
    )

    total_elapsed = time.time() - pipeline_start

    # Final summary
    print("\n")
    _banner("PIPELINE COMPLETE -- FINAL SUMMARY")

    print("\nStep timings:")
    for step, secs in timings.items():
        mins, sec = divmod(int(secs), 60)
        print(f"  {step:30s}: {mins:2d}m {sec:02d}s")
    total_mins, total_sec = divmod(int(total_elapsed), 60)
    print(f"  {'TOTAL':30s}: {total_mins:2d}m {total_sec:02d}s")

    print("\nCKD 10-fold CV Results:")
    print(_load_cv_summary("cv_results.csv"))

    print("\nFHS 10-fold CV Results:")
    print(_load_cv_summary("fhs_cv_results.csv"))

    print("\nOutput files:")
    output_files = [
        "results/fhs_cv_results.csv",
        "results/fhs_mcnemar_result.txt",
        "results/combined_latex_table.tex",
        "results/combined_summary.txt",
        "results/dual_accuracy_comparison.png",
        "results/dual_variance_comparison.png",
        "results/dual_roc_curves.png",
    ]
    for f in output_files:
        path = BASE_DIR / f
        status = "OK" if path.exists() else "MISSING"
        print(f"  [{status}] {f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
