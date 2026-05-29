"""
main_fhs.py — FHS dual-dataset validation pipeline orchestrator.

Usage:
  python main_fhs.py                  # Run full FHS pipeline
  python main_fhs.py --skip-quantum   # Skip QSVM and HybridQT (fast mode)
  python main_fhs.py --skip-preprocessing  # Skip if .npy files already exist
  python main_fhs.py --epochs 20      # Override training epochs

All output is echoed to fhs_run_log.txt in the project root.

Prerequisite: data/framingham.csv must exist before running.
"""

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FHS Hybrid Quantum-Classical Transformer Pipeline"
    )
    parser.add_argument(
        "--skip-quantum", action="store_true",
        help="Skip Quantum SVM and Hybrid Quantum Transformer (much faster)",
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing if .npy files already exist",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs for Transformer models (default: 50)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Batch size for Transformer training (default: 32)",
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="Run ablation study after main pipeline",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip expensive quantum metrics (expressibility, entanglement)",
    )
    parser.add_argument(
        "--hqct-subsample", type=int, default=800,
        help="Max training samples/fold for HybridQT (0 = no limit, default: 800)",
    )
    return parser.parse_args()


class TeeLogger:
    """Write output to both stdout and a log buffer simultaneously."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = io.StringIO()

    def write(self, data: str) -> int:
        self.stream.write(data)
        self.buffer.write(data)
        self.stream.flush()
        return len(data)

    def flush(self) -> None:
        self.stream.flush()

    def getvalue(self) -> str:
        return self.buffer.getvalue()


def step_header(n: int, total: int, name: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    bar = "=" * 42
    print(f"\n{bar}")
    print(f">> STEP {n}/{total}: {name} -- {ts}")
    print(f"{bar}")
    sys.stdout.flush()


def format_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m} min {s:02d} sec" if m > 0 else f"{s} sec"


def run_step(n: int, total: int, name: str, func, *args, **kwargs) -> tuple:
    step_header(n, total, name)
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n[ERROR] {name} failed after {format_duration(elapsed)}: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    elapsed = time.perf_counter() - t0
    print(f"\n[DONE] {name} completed in {format_duration(elapsed)}")
    sys.stdout.flush()
    return result, elapsed


def read_best_fhs_model() -> tuple:
    import pandas as pd
    csv = RESULTS_DIR / "fhs_cv_results.csv"
    if not csv.exists():
        return "Unknown", 0.0
    df = pd.read_csv(csv)
    if "F1" not in df.columns or df.empty:
        return "Unknown", 0.0
    best = df.loc[df["F1"].idxmax()]
    return best["Model"], best.get("Accuracy", 0.0) * 100


def print_final_box(total_elapsed: float, best_model: str, best_acc: float) -> None:
    duration = format_duration(total_elapsed)
    line1 = "FHS PIPELINE COMPLETE"
    line2 = f"Total time: {duration}"
    line3 = f"Best model: {best_model} ({best_acc:.2f}% acc)"
    line4 = "Results in: ./results/"

    width = max(len(line1), len(line2), len(line3), len(line4)) + 4
    top    = "+" + "=" * width + "+"
    bottom = "+" + "=" * width + "+"
    mid    = "|"

    print(f"\n{top}")
    for line in [line1, line2, line3, line4]:
        padding = width - len(line)
        print(f"{mid}  {line}{' ' * padding}{mid}")
    print(bottom)


def main() -> None:
    args = parse_args()

    # Prerequisite check
    csv_path = BASE_DIR / "data" / "framingham.csv"
    if not csv_path.exists():
        print(f"\n[ERROR] data/framingham.csv not found at: {csv_path}")
        print("Download from Kaggle: https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset")
        print(f"and place it at: {csv_path}")
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tee = TeeLogger(sys.stdout)
    sys.stdout = tee

    run_start = datetime.now()
    wall_start = time.perf_counter()
    timings: dict = {}
    total_steps = 8

    print("=" * 60)
    print("  FHS HYBRID QUANTUM-CLASSICAL TRANSFORMER PIPELINE")
    print(f"  Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config : epochs={args.epochs}, batch={args.batch_size}, "
          f"skip_quantum={args.skip_quantum}, ablation={args.ablation}, "
          f"hqct_subsample={args.hqct_subsample}")
    print("=" * 60)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    if args.skip_preprocessing:
        print("\n[SKIPPED] Step 1: Preprocessing (--skip-preprocessing flag)")
    else:
        from fhs_preprocessing import run_preprocessing as fhs_preprocess
        _, timings["Preprocessing"] = run_step(
            1, total_steps, "FHS Data Preprocessing", fhs_preprocess
        )

    # ── Step 2: Train models ───────────────────────────────────────────────────
    from fhs_train_models import run_training as fhs_train
    _, timings["Train Models"] = run_step(
        2, total_steps, "FHS Model Training",
        fhs_train, skip_quantum=args.skip_quantum
    )

    # ── Step 3: 10-fold CV evaluation ──────────────────────────────────────────
    try:
        from fhs_cv_evaluation import main as fhs_cv_main
        import sys as _sys
        old_argv = _sys.argv[:]
        _sys.argv = [_sys.argv[0]]
        if args.skip_quantum:
            _sys.argv.append("--skip-qsvm")
        _sys.argv += ["--cv-epochs", str(args.epochs)]
        if args.hqct_subsample > 0:
            _sys.argv += ["--hqct-subsample", str(args.hqct_subsample)]
        _, timings["CV Evaluation"] = run_step(
            3, total_steps, "FHS 10-Fold CV Evaluation", fhs_cv_main
        )
        _sys.argv = old_argv
    except Exception as exc:
        print(f"\n[WARNING] FHS CV Evaluation step skipped: {exc}")
        timings["CV Evaluation"] = 0.0

    # ── Step 4: Combined report ────────────────────────────────────────────────
    try:
        from combined_report import main as combined_report_main
        _, timings["Combined Report"] = run_step(
            4, total_steps, "Combined Dual-Dataset Report", combined_report_main
        )
    except Exception as exc:
        print(f"\n[WARNING] Combined report step skipped: {exc}")
        timings["Combined Report"] = 0.0

    # ── Step 5: Combined plots ─────────────────────────────────────────────────
    try:
        from combined_plots import main as combined_plots_main
        _, timings["Combined Plots"] = run_step(
            5, total_steps, "Combined Dual-Dataset Plots", combined_plots_main
        )
    except Exception as exc:
        print(f"\n[WARNING] Combined plots step skipped: {exc}")
        timings["Combined Plots"] = 0.0

    # ── Step 6: LaTeX tables ───────────────────────────────────────────────────
    try:
        from report.tables import generate_all_tables
        _, timings["LaTeX Tables"] = run_step(
            6, total_steps, "LaTeX Tables Generation", generate_all_tables
        )
    except Exception as exc:
        print(f"\n[WARNING] LaTeX tables step skipped: {exc}")
        timings["LaTeX Tables"] = 0.0

    # ── Step 7: Publication figures ────────────────────────────────────────────
    try:
        from utils.publication_plots import generate_all_figures
        _, timings["Pub Figures"] = run_step(
            7, total_steps, "IEEE-Style Publication Figures", generate_all_figures
        )
    except Exception as exc:
        print(f"\n[WARNING] Publication figures step skipped: {exc}")
        timings["Pub Figures"] = 0.0

    # ── Step 8: Ablation study (optional) ─────────────────────────────────────
    if args.ablation:
        try:
            from ablation_study import run_ablation, save_results as save_ablation
            def _run_ablation():
                results = run_ablation(n_folds=5, epochs=20, fast=args.fast)
                save_ablation(results)
            _, timings["Ablation"] = run_step(
                8, total_steps, "Ablation Study (5-fold, 20 epochs)", _run_ablation
            )
        except Exception as exc:
            print(f"\n[WARNING] Ablation study skipped: {exc}")
            timings["Ablation"] = 0.0
    else:
        print(f"\n[SKIPPED] Step 8: Ablation Study (add --ablation to enable)")

    # ── Experiment manifest ────────────────────────────────────────────────────
    try:
        from utils.integrity import save_manifest
        cfg_dict = vars(args)
        cfg_dict["pipeline"] = "fhs"
        save_manifest(str(RESULTS_DIR), cfg_dict)
        print("\n  results/experiment_manifest.json saved")
    except Exception as exc:
        print(f"\n  [WARNING] Manifest save failed: {exc}")

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - wall_start
    best_model, best_acc = read_best_fhs_model()

    print("\n" + "=" * 60)
    print("  STEP TIMING SUMMARY")
    print("=" * 60)
    for step_name, elapsed in timings.items():
        print(f"  {step_name:<35} {format_duration(elapsed):>12}")
    print("-" * 60)
    print(f"  {'TOTAL':<35} {format_duration(total_elapsed):>12}")
    print("=" * 60)

    print_final_box(total_elapsed, best_model, best_acc)

    # ── Save run log ───────────────────────────────────────────────────────────
    sys.stdout = tee.stream

    log_content = tee.getvalue()
    log_path_root = BASE_DIR / "fhs_run_log.txt"
    log_path_results = RESULTS_DIR / "fhs_run_log.txt"

    for log_path in [log_path_root, log_path_results]:
        log_path.write_text(log_content, encoding="utf-8")

    print(f"\nRun log saved to: {log_path_root}")
    print(f"All results in:   {RESULTS_DIR}")


if __name__ == "__main__":
    main()
