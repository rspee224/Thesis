import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROCESSED   = Path("../3_Data_processed")
PLOTS       = Path("../4_Plots/spatial")
PLOTS_ERROR = Path("../4_Plots/error_analysis")
GPKG_PATH   = Path("../1_Data_raw/shapefiles/wijkenbuurten_2024_v2.gpkg")
ERROR_CSV   = PROCESSED / "error_analysis_results.csv"

TARGET      = "vacancy_rate_pct"
STRUCTURAL  = ["structural_vacancy_count", "structural_vacancy_pct"]
TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
TEST_YEARS  = [2022, 2023, 2024]

FINAL_MODELS = {
    "Woningen": {
        "model_name": "RandomForest", "zeros": False, "structural": True,
        "build": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=None, max_features=0.7,
            min_samples_leaf=1, min_samples_split=2,
            random_state=42, n_jobs=-1),
    },
    "commercieel": {
        "model_name": "XGBoost", "zeros": True, "structural": True,
        "build": lambda: XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=8,
            subsample=0.8, colsample_bytree=0.6, min_child_weight=5,
            reg_alpha=0.5, reg_lambda=5.0,
            random_state=42, verbosity=0),
    },
}


def savefig(name):
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOTS / name, bbox_inches="tight", dpi=150); plt.close()
    log.info("Saved: %s", PLOTS / name)


def load_predictors(df):
    all_p = (PROCESSED / "predictors_list.txt").read_text().strip().splitlines()
    avail = [p for p in all_p if p in df.columns]
    return [p for p in avail if p not in STRUCTURAL], [p for p in avail if p in STRUCTURAL]


def load_gemeente_gdf():
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"Place wijkenbuurten_2024_v2.gpkg in {GPKG_PATH.parent}")
    gdf = gpd.read_file(GPKG_PATH, layer="gemeenten")
    gdf = gdf[gdf["gemeentecode"].str.startswith("GM")].copy()
    before = len(gdf)
    gdf = gdf[gdf["water"] == "NEE"].copy()  # filter out water bodies
    log.info("Filtered out %d water bodies (%d land municipalities remain)", before - len(gdf), len(gdf))
    gdf["gem_code_int"] = gdf["gemeentecode"].str[2:].astype(int)
    log.info("Gemeente layer: %d municipalities", len(gdf))
    return gdf


def build_corop_gdf(gdf_gem, df):
    if "COROPcode" not in df.columns:
        log.warning("COROPcode not in dataset — skipping COROP maps"); return None
    lookup = df[["municipality_code", "COROPcode"]].drop_duplicates().copy()
    lookup["gem_code_int"] = lookup["municipality_code"].astype(int)
    gdf = gdf_gem.merge(lookup[["gem_code_int", "COROPcode"]], on="gem_code_int", how="left")
    # merge municipality polygons up to COROP level
    gdf_c = gdf.dropna(subset=["COROPcode"]).dissolve(by="COROPcode").reset_index()
    gdf_c["corop_int"] = gdf_c["COROPcode"].str.upper().str.replace("CR", "").astype(int)
    log.info("COROP geometry: %d regions", len(gdf_c)); return gdf_c


def compute_residuals(df, func, cfg, base_preds, struct_preds):
    sub = df[df["property_type"] == func].dropna(subset=[TARGET]).copy()
    if not cfg["zeros"]: sub = sub[sub[TARGET] > 0].copy()
    preds = base_preds + (struct_preds if cfg["structural"] else [])
    sub = sub.dropna(subset=preds)
    train, test = sub[sub["year"].isin(TRAIN_YEARS)].copy(), sub[sub["year"].isin(TEST_YEARS)].copy()
    log.info("  %s: train=%d test=%d", cfg["model_name"], len(train), len(test))
    # retrain from scratch so residuals match the exact final model config
    m = cfg["build"](); m.fit(train[preds], train[TARGET])
    yp = m.predict(test[preds])
    log.info("  R2=%.4f MAE=%.4f", r2_score(test[TARGET], yp), mean_absolute_error(test[TARGET], yp))
    r = test[["municipality_code", "municipality_name", "year"]].copy()
    r["residual"]  = test[TARGET].values - yp
    r["abs_error"] = np.abs(r["residual"])
    r["actual"]    = test[TARGET].values
    r["predicted"] = yp
    # column name varies between preprocessing runs
    for c in ["COROPcode", "corop_code"]:
        if c in test.columns: r["COROPcode"] = test[c].values; break
    return r


def agg_gemeente(results):
    a = results.groupby("municipality_code").agg(
        mean_residual=("residual", "mean"), mean_abs_error=("abs_error", "mean"), n=("residual", "count")
    ).reset_index()
    a["gem_code_int"] = a["municipality_code"].astype(int); return a


def agg_corop(results, df):
    if "COROPcode" not in results.columns:
        lk = df[["municipality_code", "COROPcode"]].drop_duplicates() if "COROPcode" in df.columns else None
        if lk is None: return None
        results = results.merge(lk, on="municipality_code", how="left")
    a = results.groupby("COROPcode").agg(
        mean_residual=("residual", "mean"), mean_abs_error=("abs_error", "mean"), n=("residual", "count")
    ).reset_index()
    a["corop_int"] = a["COROPcode"].str.upper().str.replace("CR", "").astype(int); return a


def choropleth(gdf, col, title, fname, cmap="RdYlGn", center_zero=True, clabel="", note="", stage1_zero_col=None):
    fig, ax = plt.subplots(figsize=(8, 10))
    vals = gdf[col].dropna()
    if vals.empty: plt.close(); return
    if center_zero:
        vmax = max(abs(vals.min()), abs(vals.max())) * 1.05
        # force colour midpoint to zero so green/red split at zero
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
    # Stage 1 predicted zeros — show in light blue
    if stage1_zero_col is not None and stage1_zero_col in gdf.columns:
        gdf[gdf[stage1_zero_col] == 1].plot(ax=ax, color="#BFD7ED", edgecolor="white", linewidth=0.2)
    gdf[gdf[col].isna()].plot(ax=ax, color="#DDDDDD", edgecolor="white", linewidth=0.2)
    gdf[gdf[col].notna()].plot(ax=ax, column=col, cmap=cmap, norm=norm, edgecolor="white", linewidth=0.2)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, pad=0.02, aspect=30)
    cb.set_label(clabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10); ax.set_axis_off()
    full_note = note
    if stage1_zero_col is not None:
        full_note += " · Light blue = Stage 1 predicted zero vacancy"
    if full_note: fig.text(0.5, 0.01, full_note, ha="center", fontsize=8, color="#666", style="italic")
    plt.tight_layout(); savefig(fname)


def analyse_urbanisation(results, df, func, model_name):
    urb_col = "urbanisation_level_enc"
    if urb_col not in df.columns:
        log.warning("urbanisation_level_enc not in dataset — skipping urbanisation analysis")
        return None

    urb_lookup = df[["municipality_code", "year", urb_col]].drop_duplicates()
    res = results.merge(urb_lookup, on=["municipality_code", "year"], how="left")

    rows = []
    for level in sorted(res[urb_col].dropna().unique()):
        sub = res[res[urb_col] == level]
        if len(sub) < 10:
            continue
        r2  = r2_score(sub["actual"], sub["predicted"]) if "actual" in sub.columns else None
        mae = sub["abs_error"].mean()
        rows.append({
            "property_type":      func,
            "model":              model_name,
            "urbanisation_level": int(level),
            "n":                  len(sub),
            "MAE":                round(mae, 4),
            "R2":                 round(r2, 4) if r2 is not None else None,
        })
        log.info("  Urbanisation %d: n=%-5d MAE=%.4f%s",
                 int(level), len(sub), mae,
                 f"  R2={r2:.4f}" if r2 is not None else "")
    return pd.DataFrame(rows)


def analyse_temporal(results, func, model_name):
    rows = []
    for year in sorted(results["year"].unique()):
        sub = results[results["year"] == year]
        mae = sub["abs_error"].mean()
        r2  = r2_score(sub["actual"], sub["predicted"]) if "actual" in sub.columns else None
        rows.append({
            "property_type": func,
            "model":         model_name,
            "year":          int(year),
            "n":             len(sub),
            "MAE":           round(mae, 4),
            "R2":            round(r2, 4) if r2 is not None else None,
        })
        log.info("  Year %d: n=%-5d MAE=%.4f%s",
                 int(year), len(sub), mae,
                 f"  R2={r2:.4f}" if r2 is not None else "")
    return pd.DataFrame(rows)


def plot_urbanisation(urb_df, func, model_name):
    PLOTS_ERROR.mkdir(parents=True, exist_ok=True)
    color  = "#0D9488" if func == "Woningen" else "#D97706"
    labels = {1: "Not\nurbanised", 2: "Slightly\nurbanised",
              3: "Moderately\nurbanised", 4: "Strongly\nurbanised",
              5: "Very strongly\nurbanised"}

    fig, ax = plt.subplots(figsize=(8, 5))
    levels = urb_df["urbanisation_level"].tolist()
    maes   = urb_df["MAE"].tolist()
    ax.bar([labels.get(l, str(l)) for l in levels], maes, color=color, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Mean Absolute Error (pp)", fontsize=11)
    ax.set_title(f"MAE by Urbanisation Level — {func} ({model_name})\nTest set 2022-2024",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Urbanisation level", fontsize=11)
    plt.tight_layout()
    fname = PLOTS_ERROR / f"mae_urbanisation_{func.lower()}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150); plt.close()
    log.info("Saved: %s", fname)


def plot_temporal(temp_df, func, model_name):
    PLOTS_ERROR.mkdir(parents=True, exist_ok=True)
    color = "#0D9488" if func == "Woningen" else "#D97706"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temp_df["year"], temp_df["MAE"], marker="o", color=color, linewidth=2, markersize=8)
    for _, row in temp_df.iterrows():
        ax.annotate(f"{row['MAE']:.3f}", (row["year"], row["MAE"]),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_ylabel("Mean Absolute Error (pp)", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title(f"MAE by Year — {func} ({model_name})\nTest set 2022-2024",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(temp_df["year"].tolist())
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fname = PLOTS_ERROR / f"mae_temporal_{func.lower()}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150); plt.close()
    log.info("Saved: %s", fname)


def plot_temporal_r2(temp_df, func, model_name):
    PLOTS_ERROR.mkdir(parents=True, exist_ok=True)
    if "R2" not in temp_df.columns or temp_df["R2"].isna().all():
        log.warning("No R2 data available for %s — skipping R2 temporal plot", func)
        return
    color = "#0D9488" if func == "Woningen" else "#D97706"

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(temp_df["year"], temp_df["R2"], marker="o", color=color, linewidth=2, markersize=8)
    for _, row in temp_df.iterrows():
        if row["R2"] is not None:
            ax.annotate(f"{row['R2']:.3f}", (row["year"], row["R2"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=10)
    ax.set_ylabel("R2", fontsize=11)
    ax.set_xlabel("Year", fontsize=11)
    ax.set_title(f"R2 by Year — {func} ({model_name})\nTest set 2022-2024",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(temp_df["year"].tolist())
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    fname = PLOTS_ERROR / f"r2_temporal_{func.lower()}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150); plt.close()
    log.info("Saved: %s", fname)


def main():
    df = pd.read_csv(PROCESSED / "model_ready.csv")
    log.info("Loaded: shape=%s", df.shape)
    bp, sp   = load_predictors(df)
    gdf_gem  = load_gemeente_gdf()
    gdf_cor  = build_corop_gdf(gdf_gem, df)

    stage1_file  = PROCESSED / "stage1_predictions.csv"
    stage1_preds = pd.read_csv(stage1_file) if stage1_file.exists() else None
    if stage1_preds is not None:
        log.info("Loaded Stage 1 predictions: %d rows", len(stage1_preds))
    else:
        log.warning("stage1_predictions.csv not found — using all non-zero test rows for Woningen")

    all_urb  = []
    all_temp = []

    for func, cfg in FINAL_MODELS.items():
        log.info("%s — %s", func, cfg["model_name"])
        res = compute_residuals(df, func, cfg, bp, sp)

        if func == "Woningen" and stage1_preds is not None:
            positive_idx = set(
                stage1_preds[stage1_preds["stage1_label"] == 1]["df_index"].values
            )
            zero_municipalities = set(
                stage1_preds[stage1_preds["stage1_label"] == 0]["municipality_code"].values
            )
            before = len(res)
            res    = res[res.index.isin(positive_idx)]
            # Stage 1 zeros = municipalities where ALL rows were predicted zero
            predicted_positive_municipalities = set(res["municipality_code"].values)
            stage1_zeros_gems = zero_municipalities - predicted_positive_municipalities
            stage1_zeros = pd.DataFrame({"municipality_code": list(stage1_zeros_gems)})
            stage1_zeros["stage1_zero"] = 1
            log.info("  Stage 1 filter: %d -> %d residual rows | %d municipalities predicted zero",
                     before, len(res), len(stage1_zeros))
        else:
            stage1_zeros = None

        ag    = agg_gemeente(res)
        gdf_m = gdf_gem.merge(ag, on="gem_code_int", how="left")

        if stage1_zeros is not None:
            stage1_zeros["gem_code_int"] = stage1_zeros["municipality_code"].astype(int)
            gdf_m = gdf_m.merge(stage1_zeros[["gem_code_int", "stage1_zero"]],
                                 on="gem_code_int", how="left")
        else:
            gdf_m["stage1_zero"] = np.nan

        choropleth(gdf_m, "mean_residual",
            f"Mean Residual — {func} ({cfg['model_name']})"
            f"\nActual minus Predicted (pp) · Test set 2022-2024",
            f"residual_map_{func.lower()}_gemeente.png",
            cmap="RdYlGn", center_zero=True, clabel="Mean residual (pp)",
            note="Green = model overpredicted · Red = model underpredicted",
            stage1_zero_col="stage1_zero" if func == "Woningen" else None)

        choropleth(gdf_m, "mean_abs_error",
            f"Mean Absolute Error — {func} ({cfg['model_name']})"
            f"\nAverage prediction error (pp) · Test set 2022-2024",
            f"mae_map_{func.lower()}_gemeente.png",
            cmap="YlOrRd", center_zero=False, clabel="MAE (pp)",
            note="Darker = larger prediction error",
            stage1_zero_col="stage1_zero" if func == "Woningen" else None)

        if gdf_cor is not None:
            ac = agg_corop(res, df)
            if ac is not None:
                gdf_c = gdf_cor.merge(ac, on="corop_int", how="left")

                choropleth(gdf_c, "mean_residual",
                    f"Mean Residual — {func} ({cfg['model_name']}) — COROP"
                    f"\nActual minus Predicted (pp) · Test set 2022-2024",
                    f"residual_map_{func.lower()}_corop.png",
                    cmap="RdYlGn", center_zero=True, clabel="Mean residual (pp)",
                    note="Green = overpredicted · Red = underpredicted")

                choropleth(gdf_c, "mean_abs_error",
                    f"Mean Absolute Error — {func} ({cfg['model_name']}) — COROP"
                    f"\nAverage prediction error (pp) · Test set 2022-2024",
                    f"mae_map_{func.lower()}_corop.png",
                    cmap="YlOrRd", center_zero=False, clabel="MAE (pp)",
                    note="Darker = larger error")

        log.info("\n--- Top 10 municipalities by MAE: %s ---", func)
        top10 = ag.nlargest(10, "mean_abs_error")[
            ["municipality_code", "mean_abs_error", "mean_residual", "n"]
        ].copy()
        name_lookup = res[["municipality_code", "municipality_name"]].drop_duplicates()
        top10 = top10.merge(name_lookup, on="municipality_code", how="left")
        log.info("%-12s %-30s %-10s %-12s %-6s", "Code", "Name", "MAE (pp)", "Residual (pp)", "N")
        log.info("-" * 72)
        for _, row in top10.iterrows():
            log.info("%-12s %-30s %-10.4f %-12.4f %-6d",
                     row["municipality_code"],
                     str(row.get("municipality_name", ""))[:28],
                     row["mean_abs_error"], row["mean_residual"], row["n"])
        top10.to_csv(PROCESSED / f"top10_mae_{func.lower()}.csv", index=False)
        log.info("Top 10 saved -> %s", PROCESSED / f"top10_mae_{func.lower()}.csv")

        log.info("\n--- Moran's I: %s ---", func)
        # libpysal and esda are optional — skip gracefully if not installed
        try:
            from libpysal.weights import Queen
            from esda.moran import Moran

            gdf_moran = gdf_m.dropna(subset=["mean_residual"]).copy()
            gdf_moran = gdf_moran[gdf_moran.geometry.is_valid]

            if len(gdf_moran) > 10:
                w = Queen.from_dataframe(gdf_moran, silence_warnings=True)
                w.transform = "r"
                mi = Moran(gdf_moran["mean_residual"].values, w, permutations=999)
                log.info("  Moran's I (residuals) = %.4f  p-value = %.4f  z-score = %.4f",
                         mi.I, mi.p_sim, mi.z_sim)
                log.info("  %s", "Significant spatial clustering" if mi.p_sim < 0.05
                         else "No significant spatial autocorrelation")

                gdf_mae = gdf_m.dropna(subset=["mean_abs_error"]).copy()
                gdf_mae = gdf_mae[gdf_mae.geometry.is_valid]
                w_mae = Queen.from_dataframe(gdf_mae, silence_warnings=True)
                w_mae.transform = "r"
                mi_mae = Moran(gdf_mae["mean_abs_error"].values, w_mae, permutations=999)
                log.info("  Moran's I (MAE)       = %.4f  p-value = %.4f  z-score = %.4f",
                         mi_mae.I, mi_mae.p_sim, mi_mae.z_sim)
            else:
                log.warning("  Too few municipalities for Moran's I — skipping")

        except Exception as e:
            log.warning("  Moran's I failed: %s", e)

        log.info("\n--- Urbanisation breakdown: %s ---", func)
        urb_df = analyse_urbanisation(res, df, func, cfg["model_name"])
        if urb_df is not None:
            all_urb.append(urb_df)
            plot_urbanisation(urb_df, func, cfg["model_name"])

        log.info("\n--- Temporal breakdown: %s ---", func)
        temp_df = analyse_temporal(res, func, cfg["model_name"])
        all_temp.append(temp_df)
        plot_temporal(temp_df, func, cfg["model_name"])
        plot_temporal_r2(temp_df, func, cfg["model_name"])

    if all_urb:
        pd.concat(all_urb).to_csv(PROCESSED / "error_urbanisation.csv", index=False)
        log.info("Urbanisation results -> %s", PROCESSED / "error_urbanisation.csv")
    pd.concat(all_temp).to_csv(PROCESSED / "error_temporal.csv", index=False)
    log.info("Temporal results -> %s", PROCESSED / "error_temporal.csv")
    log.info("Done — all outputs saved")


if __name__ == "__main__":
    main()