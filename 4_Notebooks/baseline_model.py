"""
Baseline Model Script
Trains and evaluates a baseline SVR model on the model-ready dataset.

Covers:
    1. Load model-ready data
    2. Train/test split (time-based: train 2015-2021, test 2022-2024)
    3. Separate models for Woningen and commercieel
    4. Baseline: mean predictor (DummyRegressor)
    5. SVR model with scaling pipeline
    6. Evaluation: MAE, RMSE, R²
    7. Residual and prediction plots
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths

PROCESSED = Path("../3_Data_processed")
PLOTS     = Path("../4_Plots/model")

# Constants

TARGET = "vacancy_rate_pct"

# Predictors are loaded dynamically from the file saved by pre_processing.py
# This ensures baseline always uses the same features as the model comparison
PREDICTORS_FILE = PROCESSED / "predictors_list.txt"

def load_predictors(df: pd.DataFrame) -> list[str]:
    """
    Load predictor list from file saved by pre_processing.py.
    Falls back to a hardcoded minimal list if the file doesn't exist.
    Filters to only columns that exist in the loaded dataframe.
    """
    if PREDICTORS_FILE.exists():
        predictors = PREDICTORS_FILE.read_text().strip().splitlines()
        log.info("Loaded %d predictors from %s", len(predictors), PREDICTORS_FILE)
    else:
        log.warning(
            "predictors_list.txt not found — falling back to hardcoded predictor list. "
            "Run pre_processing.py first to generate the full predictor list."
        )
        predictors = [
            "total_population", "share_working_age", "share_owner_occupied",
            "share_social_rental", "share_private_rental", "avg_property_value",
            "population_growth_per1000", "grey_pressure_pct", "ses_score",
            "structural_vacancy_count", "structural_vacancy_pct",
        ]

    # Filter to columns that actually exist in the dataframe
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        log.warning("  %d predictors not found in dataset and will be skipped: %s", len(missing), missing)
    available = [p for p in predictors if p in df.columns]
    log.info("  Using %d predictors.", len(available))
    return available

# Time-based split: train on past years, test on most recent years
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS  = [2022, 2023, 2024]

PROPERTY_TYPES = ["Woningen", "commercieel"]


# Helpers


def savefig(name: str) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


def evaluate(y_true: pd.Series, y_pred: np.ndarray, label: str, split: str = "test") -> dict:
    """Compute and log MAE, RMSE, and R² for a set of predictions."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    log.info("  %-30s [%-5s]  MAE=%.3f  RMSE=%.3f  R²=%.3f", label, split, mae, rmse, r2)
    return {"label": label, "split": split, "MAE": mae, "RMSE": rmse, "R2": r2}



# Load data


def load_data() -> pd.DataFrame:
    path = PROCESSED / "model_ready.csv"
    df   = pd.read_csv(path)
    log.info("Loaded model-ready data: %s  shape=%s", path, df.shape)
    return df

# Train / test split

def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split: train on TRAIN_YEARS, test on TEST_YEARS."""
    train = df[df["year"].isin(TRAIN_YEARS)].copy()
    test  = df[df["year"].isin(TEST_YEARS)].copy()
    log.info(
        "Split — train: %d rows (%s), test: %d rows (%s)",
        len(train), TRAIN_YEARS, len(test), TEST_YEARS,
    )
    return train, test

# Modelling

def build_svr() -> Pipeline:
    """SVR pipeline with standard scaling."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    ])


def run_models(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each Gebruiksfunctie, train a DummyRegressor (baseline) and an SVR,
    evaluate on the test set, and produce diagnostic plots.
    Returns a summary DataFrame with all metrics.
    """
    results   = []
    predictors = load_predictors(df)

    for func in PROPERTY_TYPES:
        log.info("── %s ──────────────────────────────────────", func)

        subset = df[df["property_type"] == func].copy()

        if len(subset) == 0:
            log.warning(
                "── %s — NO ROWS FOUND. Available property_type values: %s",
                func, df["property_type"].unique().tolist(),
            )
            continue

        train, test = split(subset)

        X_train = train[predictors]
        y_train = train[TARGET]
        X_test  = test[predictors]
        y_test  = test[TARGET]

        log.info("  Train shape: %s  |  Test shape: %s", X_train.shape, X_test.shape)

        # 1. Dummy baseline (always predicts the training mean)
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred_dummy_test  = dummy.predict(X_test)
        y_pred_dummy_train = dummy.predict(X_train)
        results.append(evaluate(y_test,  y_pred_dummy_test,  f"{func} — Dummy (mean)", split="test"))
        results.append(evaluate(y_train, y_pred_dummy_train, f"{func} — Dummy (mean)", split="train"))

        # 2. SVR — train on log1p(target), back-transform predictions
        svr = build_svr()
        log.info("  Training SVR ...")
        svr.fit(X_train, np.log1p(y_train))
        y_pred_svr_test  = np.expm1(svr.predict(X_test))
        y_pred_svr_train = np.expm1(svr.predict(X_train))
        results.append(evaluate(y_test,  y_pred_svr_test,  f"{func} — SVR (rbf, log target)", split="test"))
        results.append(evaluate(y_train, y_pred_svr_train, f"{func} — SVR (rbf, log target)", split="train"))

        # Log train vs test gap for overfitting diagnosis
        r2_test  = r2_score(y_test,  y_pred_svr_test)
        r2_train = r2_score(y_train, y_pred_svr_train)
        gap = r2_train - r2_test
        log.info(
            "  SVR overfitting check — Train R²=%.3f  Test R²=%.3f  Gap=%.3f%s",
            r2_train, r2_test, gap,
            " ⚠ possible overfitting" if gap > 0.1 else " ✓ generalizes well",
        )

        # Plots
        plot_predictions(y_test, y_pred_svr_test, y_pred_dummy_test, func)
        plot_residuals(y_test, y_pred_svr_test, func)

    return pd.DataFrame(results)


# Plots

def plot_predictions(
    y_true: pd.Series,
    y_pred_svr: np.ndarray,
    y_pred_dummy: np.ndarray,
    func: str,
) -> None:
    """Scatter plot of actual vs predicted values for SVR and dummy."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Actual vs Predicted — {func}", fontsize=13, fontweight="bold")

    for ax, preds, label, color in zip(
        axes,
        [y_pred_dummy, y_pred_svr],
        ["Dummy (mean)", "SVR (rbf)"],
        ["#AAAAAA", "#7F77DD"],
    ):
        ax.scatter(y_true, preds, alpha=0.3, s=8, color=color)
        lim = [min(y_true.min(), preds.min()), max(y_true.max(), preds.max())]
        ax.plot(lim, lim, color="#D85A30", linewidth=1.5, linestyle="--", label="Perfect fit")
        ax.set(xlabel="Actual", ylabel="Predicted", title=label)
        ax.legend(fontsize=8)

    plt.tight_layout()
    savefig(f"predictions_{func.lower()}.png")


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, func: str) -> None:
    """Residual plot and residual distribution for SVR."""
    residuals = y_true.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Residuals — SVR — {func}", fontsize=13, fontweight="bold")

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=8, color="#7F77DD")
    axes[0].axhline(0, color="#D85A30", linewidth=1.5, linestyle="--")
    axes[0].set(xlabel="Predicted", ylabel="Residual", title="Residuals vs Predicted")

    # Residual distribution
    axes[1].hist(residuals, bins=50, color="#5DCAA5", edgecolor="white")
    axes[1].axvline(0, color="#D85A30", linewidth=1.5, linestyle="--")
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual distribution")

    plt.tight_layout()
    savefig(f"residuals_{func.lower()}.png")



# Main

def main() -> None:
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130

    # 1. Load
    df = load_data()

    # 2. Run models and collect metrics
    log.info("Running models ...")
    summary = run_models(df)

    # 3. Print summary table
    log.info("\nResults summary:\n%s", summary.to_string(index=False))

    # 4. Save summary to CSV
    out = PROCESSED / "model_results.csv"
    summary.to_csv(out, index=False)
    log.info("Results saved -> %s", out)


if __name__ == "__main__":
    main()