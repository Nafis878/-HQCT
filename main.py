"""
main.py — Full pipeline runner for the CKD Hybrid Quantum-Classical Transformer.

Usage:
  python main.py                  # Run full pipeline
  python main.py --skip-quantum   # Skip QSVM and hybrid transformer (fast mode)
  python main.py --skip-preprocessing  # Skip if .npy files already exist
  python main.py --epochs 20      # Override training epochs

All output is echoed to run_log.txt in the project root.

Step 7 of the QIP 2027 pipeline.
"""

import argparse
import io
import sys
import time
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hybrid Quantum-Classical Transformer Pipeline for CKD Classification"
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
    """Print a formatted step progress header."""
    ts = datetime.now().strftime("%H:%M:%S")
    bar = "=" * 42
    print(f"\n{bar}")
    print(f">> STEP {n}/{total}: {name} -- {ts}")
    print(f"{bar}")
    sys.stdout.flush()


def format_duration(seconds: float) -> str:
    """Format seconds as 'X min Y sec'."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m} min {s:02d} sec" if m > 0 else f"{s} sec"


def run_step(
    n: int, total: int, name: str, func, *args, **kwargs
) -> tuple:
    """
    Print step header, call func(*args, **kwargs), time it.
    Returns (result, elapsed_seconds).
    """
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


def read_best_model(results_dir: Path) -> tuple:
    """Read best model name and accuracy from results_table.csv."""
    import pandas as pd
    csv = results_dir / "results_table.csv"
    if not csv.exists():
        return "Unknown", 0.0
    df = pd.read_csv(csv)
    best = df.loc[df["F1"].idxmax()]
    return best["Model"], best["Accuracy"] * 100


def print_final_box(
    total_elapsed: float, best_model: str, best_acc: float
) -> None:
    """Print the final pipeline summary box."""
    duration = format_duration(total_elapsed)
    line1 = "PIPELINE COMPLETE"
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
    """Orchestrate the full CKD hybrid pipeline."""
    args = parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Install tee logger to capture all output
    tee = TeeLogger(sys.stdout)
    sys.stdout = tee

    run_start = datetime.now()
    wall_start = time.perf_counter()
    timings: dict = {}
    total_steps = 6

    print("=" * 60)
    print("  CKD HYBRID QUANTUM-CLASSICAL TRANSFORMER PIPELINE")
    print(f"  Started: {run_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Config : epochs={args.epochs}, batch={args.batch_size}, "
          f"skip_quantum={args.skip_quantum}")
    print("=" * 60)

    # ── Step 1: Preprocessing ──────────────────────────────────────────────────
    if args.skip_preprocessing:
        print("\n[SKIPPED] Step 1: Preprocessing (--skip-preprocessing flag)")
    else:
        from preprocessing import run_preprocessing
        _, timings["Preprocessing"] = run_step(
            1, total_steps, "Data Preprocessing", run_preprocessing
        )

    # ── Step 2: Classical TabTransformer ───────────────────────────────────────
    from models.tab_transformer import train_tab_transformer, DATA_DIR, MODELS_DIR
    _, timings["Classical TabTransformer"] = run_step(
        2, total_steps, "Classical TabTransformer Training",
        train_tab_transformer,
        data_dir=DATA_DIR,
        models_dir=MODELS_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # ── Step 3: Hybrid Quantum Transformer ─────────────────────────────────────
    if args.skip_quantum:
        print(f"\n[SKIPPED] Step 3: Hybrid Quantum Transformer (--skip-quantum flag)")
    else:
        from models.hybrid_quantum_transformer import train_hybrid_transformer
        from models.hybrid_quantum_transformer import DATA_DIR as HQT_DATA, MODELS_DIR as HQT_MODELS
        _, timings["Hybrid Quantum Transformer"] = run_step(
            3, total_steps, "Hybrid Quantum-Classical Transformer Training",
            train_hybrid_transformer,
            data_dir=HQT_DATA,
            models_dir=HQT_MODELS,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

    # ── Step 4: Baselines ──────────────────────────────────────────────────────
    from models.baselines import run_training as run_baselines
    _, timings["Baselines"] = run_step(
        4, total_steps, "Baselines (XGBoost + QSVM)",
        run_baselines, skip_quantum=args.skip_quantum
    )

    # ── Step 5: Evaluation ─────────────────────────────────────────────────────
    from evaluate import run_evaluation
    _, timings["Evaluation"] = run_step(
        5, total_steps, "Evaluation & Metrics",
        run_evaluation, skip_quantum=args.skip_quantum
    )

    # ── Step 6: Report ─────────────────────────────────────────────────────────
    from report import generate_report
    _, timings["Report"] = run_step(
        6, total_steps, "Paper-Ready Report Generation",
        generate_report
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - wall_start
    best_model, best_acc = read_best_model(RESULTS_DIR)

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
    sys.stdout = tee.stream  # restore

    log_content = tee.getvalue()
    log_path_root = BASE_DIR / "run_log.txt"
    log_path_results = RESULTS_DIR / "run_log.txt"

    for log_path in [log_path_root, log_path_results]:
        log_path.write_text(log_content, encoding="utf-8")

    print(f"\nRun log saved to: {log_path_root}")
    print(f"All results in:   {RESULTS_DIR}")


if __name__ == "__main__":
    main()
