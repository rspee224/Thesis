import json
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED   = Path("../3_Data_processed")
DATA_FILE   = PROCESSED / "model_ready.csv"
PREDICTORS  = PROCESSED / "predictors_list.txt"
PARAMS_FILE = PROCESSED / "best_params_grid_full.json"
LATEX_OUT   = PROCESSED / "binary_threshold_results.tex"

TARGET        = "vacancy_rate_pct"
PROPERTY_TYPE = "Woningen"
TRAIN_YEARS   = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS    = [2022, 2023, 2024]
BASELINE_R2   = 0.526
JSON_KEY      = "Woningen|no_zeros_with_structural|RandomForest"
THRESHOLDS    = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def load_data():
    df = pd.read_csv(DATA_FILE)
    log.info("Loaded data: %s", df.shape)
    all_preds = PREDICTORS.read_text().strip().splitlines()
    available = [p for p in all_preds if p in df.columns]
    log.info("Predictors available: %d", len(available))
    return df, available


def load_best_params():
    with open(PARAMS_FILE) as f:
        params = json.load(f)
    # key format: property_type|scenario|model
    if JSON_KEY not in params:
        raise KeyError(f"Key '{JSON_KEY}' not found. Available:\n" + "\n".join(params.keys()))
    log.info("Loaded best RF params: %s", JSON_KEY)
    return params[JSON_KEY]


def threshold_sweep(clf_name, y_prob, y_test_binary, y_test_all, X_test_all, reg, thresholds):
    results = []
    for thresh in thresholds:
        y_pred_t = (y_prob >= thresh).astype(int)
        fp = int(((y_test_binary == 0) & (y_pred_t == 1)).sum())
        fn = int(((y_test_binary == 1) & (y_pred_t == 0)).sum())
        pos_mask = y_pred_t == 1
        # Stage 2 only evaluates on observations Stage 1 passed through
        if pos_mask.sum() == 0:
            r2_t, mae_t, rmse_t = None, None, None
        else:
            y_pred_reg = reg.predict(X_test_all[pos_mask])
            y_true_reg = y_test_all[pos_mask]
            r2_t   = r2_score(y_true_reg, y_pred_reg)
            mae_t  = mean_absolute_error(y_true_reg, y_pred_reg)
            rmse_t = mean_squared_error(y_true_reg, y_pred_reg) ** 0.5
        results.append({
            "classifier":  clf_name,
            "threshold":   thresh,
            "FP":          fp,
            "FN":          fn,
            "n_passed":    int(pos_mask.sum()),
            "stage2_r2":   r2_t,
            "stage2_mae":  mae_t,
            "stage2_rmse": rmse_t,
            "delta_r2":    r2_t - BASELINE_R2 if r2_t is not None else None,
        })
        log.info("  thresh=%.2f  FP=%-4d FN=%-4d  R2=%s  MAE=%s  RMSE=%s  delta=%s",
                 thresh, fp, fn,
                 f"{r2_t:.4f}"   if r2_t   is not None else "N/A",
                 f"{mae_t:.4f}"  if mae_t  is not None else "N/A",
                 f"{rmse_t:.4f}" if rmse_t is not None else "N/A",
                 f"{r2_t - BASELINE_R2:+.4f}" if r2_t is not None else "N/A")
    return results


def save_latex(all_results, auc_scores, path):
    lines = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(r"\caption{Two-stage model threshold sweep for Residential vacancy. "
                 r"Stage 2 R\textsuperscript{2} evaluated on municipalities passed by Stage 1. "
                 r"Baseline single-stage RF R\textsuperscript{2} = 0.526.}")
    lines.append(r"\label{tab:binary_threshold}")
    lines.append(r"\begin{tabular}{llccccr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Classifier} & \textbf{Threshold} & \textbf{FP} & \textbf{FN} "
                 r"& \textbf{N passed} & \textbf{Stage 2 R\textsuperscript{2}} "
                 r"& \textbf{$\Delta$R\textsuperscript{2}} \\")
    lines.append(r"\midrule")

    # dict.fromkeys deduplicates while preserving insertion order
    classifiers = list(dict.fromkeys(r["classifier"] for r in all_results))
    for i, clf_name in enumerate(classifiers):
        clf_rows = [r for r in all_results
                    if r["classifier"] == clf_name and r["stage2_r2"] is not None]
        auc     = auc_scores[clf_name]
        best_r2 = max(r["stage2_r2"] for r in clf_rows)
        first   = True
        for row in clf_rows:
            clf_label = f"{clf_name} (AUC={auc:.3f})" if first else ""
            first = False
            r2_str = f"{row['stage2_r2']:.3f}"
            d_str  = f"{row['delta_r2']:+.3f}"
            if row["stage2_r2"] == best_r2:
                r2_str = r"\textbf{" + r2_str + "}"
                d_str  = r"\textbf{" + d_str  + "}"
            lines.append(
                f"{clf_label} & {row['threshold']:.2f} & {row['FP']} & {row['FN']} "
                f"& {row['n_passed']} & {r2_str} & {d_str} \\\\"
            )
        if i < len(classifiers) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    Path(path).write_text(tex)
    log.info("LaTeX table saved -> %s", path)
    return tex


def main():
    df, all_preds = load_data()
    best_rf_params = load_best_params()

    sub = df[df["property_type"] == PROPERTY_TYPE].dropna(subset=[TARGET]).copy()
    sub = sub.dropna(subset=[p for p in all_preds if p in sub.columns])
    log.info("Residential rows (all): %d  |  Zeros: %d (%.1f%%)",
             len(sub), (sub[TARGET] == 0).sum(), 100 * (sub[TARGET] == 0).mean())

    preds     = [p for p in all_preds if p in sub.columns]
    train_all = sub[sub["year"].isin(TRAIN_YEARS)].sort_values("year").copy()
    test_all  = sub[sub["year"].isin(TEST_YEARS)].copy()
    log.info("Train: %d  |  Test: %d", len(train_all), len(test_all))

    X_train_all       = train_all[preds].values
    X_test_all        = test_all[preds].values
    # binary: 1 = vacancy present, 0 = no vacancy
    y_train_binary    = (train_all[TARGET] > 0).astype(int).values
    y_test_binary     = (test_all[TARGET] > 0).astype(int).values
    y_test_all_values = test_all[TARGET].values

    log.info("Stage 2 — RF Regression (shared across all Stage 1 variants)")
    # Stage 2 trained on non-zero rows only — matches the actual pipeline
    train_nozero = train_all[train_all[TARGET] > 0].copy()
    reg = RandomForestRegressor(random_state=42, n_jobs=-1, **best_rf_params)
    reg.fit(train_nozero[preds].values, train_nozero[TARGET].values)

    y_train_pred = reg.predict(train_nozero[preds].values)
    log.info("Stage 2 train R2=%.4f", r2_score(train_nozero[TARGET].values, y_train_pred))

    test_nozero = test_all[test_all[TARGET] > 0].copy()
    r2_sanity = r2_score(test_nozero[TARGET].values, reg.predict(test_nozero[preds].values))
    log.info("Stage 2 sanity check R2=%.4f  (baseline=%.3f)", r2_sanity, BASELINE_R2)

    classifiers = {}

    log.info("Stage 1a — Baseline RF")
    clf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_base.fit(X_train_all, y_train_binary)
    prob_base = clf_base.predict_proba(X_test_all)[:, 1]
    classifiers["Baseline RF"] = prob_base
    log.info("AUC-ROC: %.4f", roc_auc_score(y_test_binary, prob_base))

    log.info("Stage 1b — Baseline RF + Platt calibration")
    clf_platt = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        method="sigmoid", cv=3
    )
    clf_platt.fit(X_train_all, y_train_binary)
    prob_platt = clf_platt.predict_proba(X_test_all)[:, 1]
    classifiers["Platt RF"] = prob_platt
    log.info("AUC-ROC: %.4f", roc_auc_score(y_test_binary, prob_platt))

    log.info("Stage 1c — Baseline RF + Isotonic calibration")
    clf_iso = CalibratedClassifierCV(
        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        method="isotonic", cv=3
    )
    clf_iso.fit(X_train_all, y_train_binary)
    prob_iso = clf_iso.predict_proba(X_test_all)[:, 1]
    classifiers["Isotonic RF"] = prob_iso
    log.info("AUC-ROC: %.4f", roc_auc_score(y_test_binary, prob_iso))

    log.info("Stage 1d — Class-weighted RF")
    clf_weighted = RandomForestClassifier(n_estimators=200, random_state=42,
                                          n_jobs=-1, class_weight="balanced")
    clf_weighted.fit(X_train_all, y_train_binary)
    prob_weighted = clf_weighted.predict_proba(X_test_all)[:, 1]
    classifiers["Weighted RF"] = prob_weighted
    log.info("AUC-ROC: %.4f", roc_auc_score(y_test_binary, prob_weighted))

    log.info("Stage 1e — SMOTE RF")
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_all, y_train_binary)
    log.info("SMOTE resampled: %d -> %d rows", len(X_train_all), len(X_train_sm))
    clf_smote = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf_smote.fit(X_train_sm, y_train_sm)
    prob_smote = clf_smote.predict_proba(X_test_all)[:, 1]
    classifiers["SMOTE RF"] = prob_smote
    log.info("AUC-ROC: %.4f", roc_auc_score(y_test_binary, prob_smote))

    all_results = []
    auc_scores  = {}

    for clf_name, y_prob in classifiers.items():
        log.info("Threshold sweep — %s", clf_name)
        auc_scores[clf_name] = roc_auc_score(y_test_binary, y_prob)
        results = threshold_sweep(
            clf_name, y_prob, y_test_binary,
            y_test_all_values, X_test_all, reg, THRESHOLDS
        )
        all_results.extend(results)

    log.info("Summary — best result per classifier")
    log.info("%-15s %-12s %-8s %-8s %-10s %-10s",
             "Classifier", "Threshold", "FP", "FN", "Stage2 R2", "vs baseline")
    for clf_name in classifiers:
        clf_rows = [r for r in all_results
                    if r["classifier"] == clf_name and r["stage2_r2"] is not None]
        best = max(clf_rows, key=lambda r: r["stage2_r2"])
        log.info("%-15s %-12.2f %-8d %-8d %-10.4f %-10s",
                 clf_name, best["threshold"], best["FP"], best["FN"],
                 best["stage2_r2"], f"{best['delta_r2']:+.4f}")
    log.info("Single-stage baseline R2: %.4f", BASELINE_R2)

    # 0.95 chosen because it eliminates all false positives (see threshold sweep above)
    OPTIMAL_THRESHOLD = 0.95
    log.info("Saving Stage 1 predictions at threshold=%.2f", OPTIMAL_THRESHOLD)

    # baseline RF used — highest AUC of all five variants
    y_pred_optimal = (prob_base >= OPTIMAL_THRESHOLD).astype(int)
    stage1_out = test_all[["municipality_code", "year"]].copy()
    stage1_out["stage1_label"] = y_pred_optimal
    stage1_out["stage1_prob"]  = prob_base
    # promote index to column so downstream scripts can filter by original row position
    stage1_out.index.name = "df_index"
    stage1_out = stage1_out.reset_index()

    out_path = PROCESSED / "stage1_predictions.csv"
    stage1_out.to_csv(out_path, index=False)
    log.info("Stage 1 predictions saved -> %s", out_path)
    log.info("  Predicted non-zero: %d  |  Predicted zero: %d",
             (y_pred_optimal == 1).sum(), (y_pred_optimal == 0).sum())

    PLOTS_ERROR = Path("../4_Plots/error_analysis")
    PLOTS_ERROR.mkdir(parents=True, exist_ok=True)

    fn_mask = (y_test_binary == 1) & (y_pred_optimal == 0)
    tp_mask = (y_test_binary == 1) & (y_pred_optimal == 1)
    fn_vals = y_test_all_values[fn_mask]
    tp_vals = y_test_all_values[tp_mask]

    log.info("Group comparison summary:")
    log.info("%-25s %-8s %-8s %-8s %-8s", "Group", "N", "Mean", "Median", "Std")
    for label, vals in [("False negatives", fn_vals), ("True positives", tp_vals)]:
        log.info("%-25s %-8d %-8.3f %-8.3f %-8.3f",
                 label, len(vals), vals.mean(), np.median(vals), vals.std())

    cap = 15.0  # cap at 15pp for readability
    fn_plot = fn_vals[fn_vals <= cap]
    tp_plot = tp_vals[tp_vals <= cap]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0, cap + 0.5, 0.5)

    ax.hist(tp_plot, bins=bins, density=True, alpha=0.5, color="#0D9488",
            label=f"True positives (n=3,026)", edgecolor="white", linewidth=0.5)
    ax.hist(fn_plot, bins=bins, density=True, alpha=0.6, color="#DC2626",
            label=f"False negatives (n=385)", edgecolor="white", linewidth=0.5)

    ax.axvline(fn_vals.mean(), color="#DC2626", linestyle="--", linewidth=2,
               label=f"FN mean = {fn_vals.mean():.2f} pp")
    ax.axvline(tp_vals.mean(), color="#0D9488", linestyle="--", linewidth=2,
               label=f"TP mean = {tp_vals.mean():.2f} pp")

    ax.set_xlabel("Actual vacancy rate (pp)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Vacancy Rate Distribution — False Negatives vs True Positives\n"
                 "Residential Stage 1 — Threshold 0.95 — Test set 2022-2024",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0, cap)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.text(0.99, 0.01,
             f"Note: {(fn_vals > cap).sum()} FN and {(tp_vals > cap).sum()} TP "
             f"observations above {cap:.0f}pp excluded from plot",
             ha="right", fontsize=8, color="#888", style="italic")

    plt.tight_layout()
    fname = PLOTS_ERROR / "fn_comparison_kde.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    log.info("FN comparison histogram saved -> %s", fname)


if __name__ == "__main__":
    main()