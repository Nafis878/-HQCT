"""
scripts/sanity_check.py -- Verify pipeline integrity and result reproducibility.

Checks (PASS/FAIL):
  1. SHA-256 hashes in data_hashes.json match actual files
  2. Provenance records in provenance_log.json validate against model files
  3. CV results reported metrics are within ±0.5% of full_metrics CSVs
  4. McNemar contingency table a+b+c+d matches dataset size
  5. Statistical tests JSON parses and contains required keys
  6. All required result files exist

Run: python scripts/sanity_check.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

PASS = "✓ PASS"
FAIL = "✗ FAIL"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    print(f"  [{status}]  {label}")
    if not condition and detail:
        print(f"           Detail: {detail}")
    return condition


def run_checks() -> int:
    print("=" * 60)
    print("SANITY CHECK — HQCT Pipeline")
    print("=" * 60)
    failures = 0

    # ── Check 1: Required result files exist ──────────────────────────────────
    print("\n[1] Required output files")
    required = [
        RESULTS_DIR / "cv_results.csv",
        RESULTS_DIR / "fhs_cv_results.csv",
        RESULTS_DIR / "mcnemar_result.txt",
        RESULTS_DIR / "fhs_mcnemar_result.txt",
    ]
    for f in required:
        ok = f.exists()
        if not check(f.name, ok, f"Missing: {f}"):
            failures += 1

    # ── Check 2: Data hash integrity ──────────────────────────────────────────
    print("\n[2] SHA-256 data integrity")
    hash_path = RESULTS_DIR / "data_hashes.json"
    if not hash_path.exists():
        check("data_hashes.json exists", False, str(hash_path))
        failures += 1
    else:
        try:
            from utils.integrity import compute_sha256
            with open(hash_path) as f:
                hashes = json.load(f)
            for label, info in hashes.items():
                stored = info.get("sha256", "")
                filepath = info.get("filepath", "")
                if not filepath or not Path(filepath).exists():
                    check(f"Hash file exists ({label})", False, f"File not found: {filepath}")
                    failures += 1
                    continue
                actual = compute_sha256(filepath)
                ok = actual == stored
                if not check(f"SHA-256 match ({label})", ok,
                             f"Expected {stored[:16]}... got {actual[:16]}..."):
                    failures += 1
        except Exception as exc:
            check("data_hashes.json readable", False, str(exc))
            failures += 1

    # ── Check 3: Provenance records validate ──────────────────────────────────
    print("\n[3] Model provenance")
    prov_path = RESULTS_DIR / "provenance_log.json"
    if not prov_path.exists():
        check("provenance_log.json exists", False, str(prov_path))
        failures += 1
    else:
        try:
            from utils.integrity import compute_sha256
            with open(prov_path) as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = [records]

            for rec in records[:5]:  # Check first 5
                model_name = rec.get("model_path", "unknown")
                stored_sha = rec.get("model_sha256", "")
                model_dir = BASE_DIR / "models"
                model_file = model_dir / model_name
                if not model_file.exists():
                    check(f"Provenance model file ({model_name})", False,
                          f"File not found: {model_file}")
                    failures += 1
                    continue
                actual = compute_sha256(str(model_file))
                ok = actual == stored_sha
                if not check(f"Provenance SHA match ({model_name})", ok,
                             f"Hash mismatch — model file may have changed"):
                    failures += 1
        except Exception as exc:
            check("provenance_log.json valid", False, str(exc))
            failures += 1

    # ── Check 4: CV results within ±0.5% of full metrics ─────────────────────
    print("\n[4] CV result consistency (±0.5% tolerance)")
    for dataset, cv_csv, full_csv in [
        ("CKD", RESULTS_DIR / "cv_results.csv", RESULTS_DIR / "full_metrics_ckd.csv"),
        ("FHS", RESULTS_DIR / "fhs_cv_results.csv", RESULTS_DIR / "full_metrics_fhs.csv"),
    ]:
        if not cv_csv.exists() or not full_csv.exists():
            check(f"{dataset}: CV + full metrics both exist",
                  cv_csv.exists() and full_csv.exists())
            continue
        try:
            import pandas as pd
            cv_df = pd.read_csv(cv_csv)
            full_df = pd.read_csv(full_csv)
            for _, cv_row in cv_df.iterrows():
                model = cv_row["Model"]
                full_sub = full_df[full_df["model"] == model]
                if full_sub.empty:
                    continue
                reported_acc = cv_row.get("Accuracy", 0)
                computed_acc = full_sub["acc"].mean()
                diff = abs(reported_acc - computed_acc)
                ok = diff < 0.005
                if not check(f"{dataset}/{model}: Acc within ±0.5%", ok,
                             f"Reported={reported_acc:.4f} Computed={computed_acc:.4f} diff={diff:.4f}"):
                    failures += 1
        except Exception as exc:
            check(f"{dataset}: consistency check", False, str(exc))
            failures += 1

    # ── Check 5: McNemar contingency table size ───────────────────────────────
    print("\n[5] McNemar contingency table")
    for dataset, detail_json, y_file in [
        ("CKD", RESULTS_DIR / "mcnemar_detail.json", DATA_DIR / "y_full.npy"),
        ("FHS", RESULTS_DIR / "fhs_mcnemar_detail.json", DATA_DIR / "fhs_y_full.npy"),
    ]:
        if not detail_json.exists():
            check(f"{dataset}: mcnemar_detail.json exists", False)
            continue
        if not y_file.exists():
            check(f"{dataset}: y_full.npy exists", False)
            continue
        try:
            with open(detail_json) as f:
                detail = json.load(f)
            a = detail["a_both_correct"]
            b = detail["b_hqct_correct_xgb_wrong"]
            c = detail["c_hqct_wrong_xgb_correct"]
            d = detail["d_both_wrong"]
            total = a + b + c + d
            n_samples = int(np.load(y_file).shape[0])
            ok = total == n_samples
            if not check(f"{dataset}: a+b+c+d == n_samples", ok,
                         f"a+b+c+d={total}, n_samples={n_samples}"):
                failures += 1
        except Exception as exc:
            check(f"{dataset}: McNemar detail valid", False, str(exc))
            failures += 1

    # ── Check 6: Statistical tests JSON parses ────────────────────────────────
    print("\n[6] Statistical tests outputs")
    for dataset, stat_json in [
        ("CKD", RESULTS_DIR / "statistical_tests.json"),
        ("FHS", RESULTS_DIR / "fhs_statistical_tests.json"),
    ]:
        if not stat_json.exists():
            check(f"{dataset}: statistical_tests.json exists", False)
            continue
        try:
            with open(stat_json) as f:
                stats = json.load(f)
            has_wilcoxon = "wilcoxon_pairs" in stats or any("wilcoxon" in k for k in stats)
            if not check(f"{dataset}: Wilcoxon results present", has_wilcoxon):
                failures += 1
        except Exception as exc:
            check(f"{dataset}: statistical_tests.json parseable", False, str(exc))
            failures += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures == 0:
        print(f"ALL CHECKS PASSED ({PASS})")
    else:
        print(f"{failures} CHECK(S) FAILED ({FAIL})")
        print("Review the FAIL details above before submission.")
    print("=" * 60)
    return failures


def main():
    failures = run_checks()
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
