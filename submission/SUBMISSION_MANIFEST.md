# HQCT Submission Package Manifest
Generated: 2026-05-31 00:59
Target: Expert Systems with Applications (editorialmanager.com/eswa)
Manuscript type: Research Paper
Repository: https://github.com/Nafis878/-HQCT (tag v1.0-submission)

> Honest positioning: HybridQT leads on Cleveland (best MCC, tied-best accuracy,
> lowest variance) and is 2nd of 5 by F1 on the hard, imbalanced FHS; it is
> competitive but not best on the near-saturated CKD and on PIMA. All numbers
> trace to results/*.csv and *.json.

## figures/  (8 main figures; submit as separate files)
| File | Figure | Caption |
|------|--------|---------|
| fig1_architecture.pdf/.png | Fig. 1 | HQCT architecture (6-qubit HEA feed-forward) |
| fig2_roc_curves.pdf/.png | Fig. 2 | ROC curves (pooled out-of-fold), 4 datasets |
| fig3_pr_curves.pdf/.png | Fig. 3 | Multi-metric comparison, 4 datasets |
| fig4_critical_difference.pdf/.png | Fig. 4 | Model ranking by AUC, per dataset |
| fig5_quantum_scatter.pdf/.png | Fig. 5 | Measured expressibility vs entanglement |
| fig6_barren_plateau.pdf/.png | Fig. 6 | Trainability / convergence (illustrative) |
| fig7_shap_summary.pdf/.png | Fig. 7 | XGBoost TreeSHAP feature importance |
| fig8_calibration.pdf/.png | Fig. 8 | Reliability diagrams + ECE |
(Plus shap_summary_XGBoost_*, barren_plateau, calibration_* as supporting PDFs.)

## tables/
| File | Table | Content |
|------|-------|---------|
| table1_metrics.tex | Table 1 | Acc/F1/Sens/Spec/AUC/MCC/Kappa/Brier, 5 models x 4 datasets |
| table2_significance.tex | Table 2 | Pairwise Wilcoxon + Friedman per dataset |
| table3_quantum_circuit.tex | Table 3 | Expressibility, entanglement, params (3 configs) |
| table4_provenance.tex | Table 4 | Dataset SHA-256 + preprocessing |
| ablation_table.tex | Table 5 | 6-condition ablation |

## manuscript/   (LaTeX fragments — \input into your main .tex)
abstract.tex, methods_quantum.tex, results_table.tex, reproducibility.tex, fhs_context.tex

## supplementary/
full_metrics_{ckd,pima,cleveland,fhs}.csv, bootstrap_confidence_intervals.csv,
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
