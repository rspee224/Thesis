"""
Data Overview Script
====================
Covers:
    1. SES (Socio-Economic Status) scores
    2. Population density per municipality
    3. Municipality harmonization (fusies & splits)
    4. Regional key figures (regionale kerncijfers)
    5. Leegstandsmonitor (vacancy monitor)
    6. Merging labels and predictors
    7. EDA (exploratory data analysis)

"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


# Logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths

RAW       = Path("../1_Data_raw")
INTERIM   = Path("../2_Data_intermediate")
PROCESSED = Path("../3_Data_processed")
PLOTS     = Path("plots")

COROP_KERNCIJFERS_FILE = "kerncijfers_corop.csv"

# Column constants

BEV_ABSOLUTE_COLS = [
    "BevolkingAanHetBeginVanDePeriode_1",
    "LevendGeborenKinderen_2",
    "Overledenen_3",
    "TotaleVestiging_4",
    "VestigingVanuitEenAndereGemeente_5",
    "Immigratie_6",
    "TotaalVertrekInclAdmCorrecties_7",
    "VertrekNaarAndereGemeente_8",
    "EmigratieInclusiefAdmCorrecties_9",
    "OverigeCorrecties_10",
    "Bevolkingsgroei_11",
    "BevolkingsgroeiSinds1Januari_13",
    "BevolkingAanHetEindeVanDePeriode_15",
]
BEV_RELATIVE_COLS = [
    "BevolkingsgroeiRelatief_12",
    "BevolkingsgroeiSinds1JanuariRela_14",
]

KERN_ABSOLUTE_COLS = [
    "TotaleBevolking_1",
    "k_20Tot25Jaar_8",
    "k_25Tot45Jaar_9",
    "k_45Tot65Jaar_10",
    "k_65Tot80Jaar_11",
    "k_80JaarOfOuder_12",
    "LevendGeborenKinderen_58",
    "Overledenen_60",
    "Bevolkingsgroei_79",
    "Land_249",
    # New municipality-level features
    "Werkloosheid_159",          # unemployment (absolute count)
    "Migratiesaldo_76",          # net migration (absolute count)
    "BedrijfsvestigingenTotaal_168",  # total businesses (absolute count)
]
KERN_RELATIVE_COLS = [
    "Koopwoningen_94",
    "HuurwoningenVanWoningcorporatie_95",
    "HuurwoningenVanOverigeVerhuurders_96",
    "GemiddeldeWOZWaardeVanWoningen_98",
    # New municipality-level features
    "Bevolkingsdichtheid_57",         # population density (relative)
    "AfstandTotTreinstation_238",     # distance to train station (relative)
    "GemiddeldeHuishoudensgrootte_89", # avg household size (relative)
]

TARGET = "vacancy_rate_pct"

# Mapping from raw Dutch/CBS column names to clean English names
COLUMN_RENAME = {
    "Jaar":                                                    "year",
    "Gemeentecode":                                            "municipality_code",
    "GemeenteCode":                                            "municipality_code",
    "Gemeentenaam":                                            "municipality_name",
    "Stedelijkheid":                                           "urbanisation_level",
    "urbanisation_level_enc":                                  "urbanisation_level_enc",
    "Gebruiksfunctie":                                         "property_type",
    "Onderzoekspopulatie":                                     "research_population",
    "Leegstand (aantal)":                                      "vacancy_count",
    "Leegstandspercentage_calc":                               "vacancy_rate",
    "Leegstandspercentage_calc_pct":                           "vacancy_rate_pct",
    "TotaleBevolking_1":                                       "total_population",
    "aandeel_20_65":                                           "share_working_age",
    "aandeel_65plus":                                          "share_elderly",
    "Koopwoningen_94":                                         "share_owner_occupied",
    "HuurwoningenVanWoningcorporatie_95":                      "share_social_rental",
    "HuurwoningenVanOverigeVerhuurders_96":                    "share_private_rental",
    "GemiddeldeWOZWaardeVanWoningen_98":                       "avg_property_value",
    "Bevolkingsgroei_per1000_CBS":                             "population_growth_per1000",
    "GrijzeDruk_pct":                                          "grey_pressure_pct",
    "SES_score":                                               "ses_score",
    "Leegstand, jaar eerder ook al leeg (aantal)":             "structural_vacancy_count",
    "structurele_leegstand_pct":                               "structural_vacancy_pct",
    # New municipality-level features from kerncijfers
    "Werkloosheid_159":                                        "unemployment",
    "Migratiesaldo_76":                                        "net_migration",
    "BedrijfsvestigingenTotaal_168":                           "total_businesses",
    "Bevolkingsdichtheid_57":                                  "population_density",
    "AfstandTotTreinstation_238":                              "dist_train_station",
    "GemiddeldeHuishoudensgrootte_89":                         "avg_household_size",
    # COROP-level features
    "corop_unemployment":                                      "corop_unemployment",
    "corop_population_density":                                "corop_population_density",
    "corop_net_migration":                                     "corop_net_migration",
    "corop_total_businesses":                                  "corop_total_businesses",
    "corop_dist_train_station":                                "corop_dist_train_station",
    "corop_avg_household_size":                                "corop_avg_household_size",
    "corop_total_jobs":                                        "corop_total_jobs",
}

# Urbanisation level ordinal mapping (Niet stedelijk=1 ... Zeer sterk stedelijk=5)
URBANISATION_ORDER = {
    "Niet stedelijk":       1,
    "Weinig stedelijk":     2,
    "Matig stedelijk":      3,
    "Sterk stedelijk":      4,
    "Zeer sterk stedelijk": 5,
}

# COROP-level columns to extract from kerncijfers_corop.csv
COROP_COLS = {
    "Werkloosheid_159":            "corop_unemployment",
    "Bevolkingsdichtheid_57":      "corop_population_density",
    "Migratiesaldo_76":            "corop_net_migration",
    "BedrijfsvestigingenTotaal_168": "corop_total_businesses",
    "AfstandTotTreinstation_238":  "corop_dist_train_station",
    "GemiddeldeHuishoudensgrootte_89": "corop_avg_household_size",
    "TotaalBanen_116":             "corop_total_jobs",
}

PREDICTORS = [
    # Municipality-level demographic and housing features
    "total_population",
    "share_working_age",
    "share_owner_occupied",
    "share_social_rental",
    "share_private_rental",
    "avg_property_value",
    "population_growth_per1000",
    "grey_pressure_pct",
    "ses_score",
    # New municipality-level features
    "unemployment",
    "net_migration",
    "total_businesses",
    "population_density",
    "dist_train_station",
    "avg_household_size",
    # Urbanisation level (ordinal 1-5)
    "urbanisation_level_enc",
    # Structural vacancy features from Leegstandsmonitor
    "structural_vacancy_count",
    "structural_vacancy_pct",
    "vacancy_rate_lag1",
    # COROP-level regional economic features
    "corop_unemployment",
    "corop_population_density",
    "corop_net_migration",
    "corop_total_businesses",
    "corop_dist_train_station",
    "corop_avg_household_size",
    "corop_total_jobs",
]

# COROP dummy columns are added dynamically during preprocessing
# and appended to PREDICTORS in prepare_modelling_dataset()

# Minimum number of objects required for a vacancy rate to be meaningful
MIN_ONDERZOEKSPOPULATIE = 25

SUM_COLS = [
    "Totale voorraad",
    "Onderzoekspopulatie",
    "Leegstand (aantal)",
    "Leegstand, jaar eerder ook al leeg (aantal)",   # structural vacancy count
]

GROUP_KEYS = ["Jaar", "Gemeentecode", "Gemeentenaam", "Stedelijkheid", "COROPcode"]

COMMERCIEEL_FUNCTIES = [
    "Kantoren", "Winkels", "Industrie", "Logies", "Bijeenkomsten",
    "Gezondheidszorg", "Onderwijs", "Sport",
    "Niet-woning met meerdere functies", "Overige",
]

# Municipality harmonization data

FUSIES: List[Dict[str, int]] = [
    # 2016
    {"oude": 1921, "nieuwe": 1940, "jaar": 2016},  # De Friese Meren -> De Fryske Marren
    {"oude": 241,  "nieuwe": 1945, "jaar": 2016},  # Groesbeek -> Berg en Dal
    {"oude": 381,  "nieuwe": 1942, "jaar": 2016},  # Gooise Meeren
    {"oude": 424,  "nieuwe": 1942, "jaar": 2016},
    {"oude": 425,  "nieuwe": 1942, "jaar": 2016},
    {"oude": 478,  "nieuwe": 385,  "jaar": 2016},  # Edam-Volendam
    # 2017
    {"oude": 844,  "nieuwe": 1948, "jaar": 2017},
    {"oude": 846,  "nieuwe": 1948, "jaar": 2017},
    {"oude": 860,  "nieuwe": 1948, "jaar": 2017},  # Meijerstad
    # 2018
    {"oude": 7,    "nieuwe": 1950, "jaar": 2018},
    {"oude": 48,   "nieuwe": 1950, "jaar": 2018},  # Westerwolde
    {"oude": 18,   "nieuwe": 1952, "jaar": 2018},
    {"oude": 40,   "nieuwe": 1952, "jaar": 2018},
    {"oude": 1987, "nieuwe": 1952, "jaar": 2018},  # Midden-Groningen
    {"oude": 63,   "nieuwe": 1949, "jaar": 2018},
    {"oude": 70,   "nieuwe": 1949, "jaar": 2018},
    {"oude": 1908, "nieuwe": 1949, "jaar": 2018},  # Waadhoeke
    {"oude": 81,   "nieuwe": 80,   "jaar": 2018},  # Leeuwarderadeel -> Leeuwarden
    {"oude": 196,  "nieuwe": 299,  "jaar": 2018},  # Rijnwaarden -> Zevenaar
    # 2019
    {"oude": 5,    "nieuwe": 1966, "jaar": 2019},
    {"oude": 1651, "nieuwe": 1966, "jaar": 2019},
    {"oude": 1663, "nieuwe": 1966, "jaar": 2019},
    {"oude": 53,   "nieuwe": 1966, "jaar": 2019},  # Het Hogeland
    {"oude": 9,    "nieuwe": 14,   "jaar": 2019},  # Ten Boer -> Groningen
    {"oude": 17,   "nieuwe": 14,   "jaar": 2019},  # Haren -> Groningen
    {"oude": 15,   "nieuwe": 1969, "jaar": 2019},
    {"oude": 22,   "nieuwe": 1969, "jaar": 2019},
    {"oude": 25,   "nieuwe": 1969, "jaar": 2019},
    {"oude": 56,   "nieuwe": 1969, "jaar": 2019},  # Westerkwartier
    {"oude": 58,   "nieuwe": 1970, "jaar": 2019},
    {"oude": 79,   "nieuwe": 1970, "jaar": 2019},
    {"oude": 1722, "nieuwe": 1970, "jaar": 2019},  # Noardeast-Fryslan
    {"oude": 236,  "nieuwe": 1960, "jaar": 2019},
    {"oude": 304,  "nieuwe": 1960, "jaar": 2019},
    {"oude": 733,  "nieuwe": 1960, "jaar": 2019},  # West Betuwe
    {"oude": 393,  "nieuwe": 394,  "jaar": 2019},  # Haarlemmerliede -> Haarlemmermeer
    {"oude": 545,  "nieuwe": 1961, "jaar": 2019},
    {"oude": 620,  "nieuwe": 1961, "jaar": 2019},
    {"oude": 707,  "nieuwe": 1961, "jaar": 2019},  # Vijfheerenlanden
    {"oude": 576,  "nieuwe": 575,  "jaar": 2019},  # Noordwijkerhout -> Noordwijk
    {"oude": 584,  "nieuwe": 1963, "jaar": 2019},
    {"oude": 585,  "nieuwe": 1963, "jaar": 2019},
    {"oude": 588,  "nieuwe": 1963, "jaar": 2019},
    {"oude": 611,  "nieuwe": 1963, "jaar": 2019},
    {"oude": 617,  "nieuwe": 1963, "jaar": 2019},  # Hoeksche Waard
    {"oude": 689,  "nieuwe": 1978, "jaar": 2019},
    {"oude": 1927, "nieuwe": 1978, "jaar": 2019},  # Molenlanden
    {"oude": 738,  "nieuwe": 1959, "jaar": 2019},
    {"oude": 870,  "nieuwe": 1959, "jaar": 2019},
    {"oude": 874,  "nieuwe": 1959, "jaar": 2019},  # Altena
    {"oude": 881,  "nieuwe": 1954, "jaar": 2019},
    {"oude": 951,  "nieuwe": 1954, "jaar": 2019},
    {"oude": 962,  "nieuwe": 1954, "jaar": 2019},  # Beekdaelen
    # 2021
    {"oude": 3,    "nieuwe": 1979, "jaar": 2021},
    {"oude": 10,   "nieuwe": 1979, "jaar": 2021},
    {"oude": 24,   "nieuwe": 1979, "jaar": 2021},  # Eemsdelta
    # 2022
    {"oude": 370,  "nieuwe": 439,  "jaar": 2022},  # Beemster -> Purmerend
    {"oude": 398,  "nieuwe": 1980, "jaar": 2022},
    {"oude": 416,  "nieuwe": 1980, "jaar": 2022},  # Dijk en Waard
    {"oude": 1685, "nieuwe": 1991, "jaar": 2022},
    {"oude": 856,  "nieuwe": 1991, "jaar": 2022},  # Maashorst
    {"oude": 756,  "nieuwe": 1982, "jaar": 2022},
    {"oude": 1684, "nieuwe": 1982, "jaar": 2022},
    {"oude": 786,  "nieuwe": 1982, "jaar": 2022},
    {"oude": 815,  "nieuwe": 1982, "jaar": 2022},
    {"oude": 1702, "nieuwe": 1982, "jaar": 2022},  # Land van Cuijk
    {"oude": 457,  "nieuwe": 363,  "jaar": 2022},  # Weesp -> Amsterdam
    # 2023
    {"oude": 501,  "nieuwe": 1992, "jaar": 2023},
    {"oude": 530,  "nieuwe": 1992, "jaar": 2023},
    {"oude": 614,  "nieuwe": 1992, "jaar": 2023},  # Voorne aan Zee
]

# Weights ideally based on population counts; otherwise village counts.
SPLITS: Dict[int, Dict[int, float]] = {
    # Haaren (2021)
    788: {756: 0.134, 824: 0.357, 855: 0.125, 865: 0.384},
    # Littenseradiel (village count weights: 15/10/4)
    140: {1900: 15/29, 80: 10/29, 1949: 4/29},
}

# Harmonization helpers

def _resolve_all_sources_to_final(fusies: Iterable[Dict[str, Any]]) -> Dict[int, int]:
    """Map every historical municipality code directly to its 2024 final code."""
    parent = {int(f["oude"]): int(f["nieuwe"]) for f in fusies}

    def find_final(x: int) -> int:
        path: List[int] = []
        while x in parent:
            path.append(x)
            x = parent[x]
            if x in path:
                break  # cycle guard
        return x

    return {oude: find_final(oude) for oude in parent}


def _build_direct_final_map(fusies: Iterable[Dict[str, Any]]) -> Dict[int, Tuple[int, int]]:
    """Map every historical code to (final_code, earliest_merger_year)."""
    parent: Dict[int, Tuple[int, int]] = {
        int(f["oude"]): (int(f["nieuwe"]), int(f["jaar"])) for f in fusies
    }
    final_map: Dict[int, Tuple[int, int]] = {}
    for oude in parent:
        cur, min_year = oude, None
        visited: set = set()
        while cur in parent and cur not in visited:
            visited.add(cur)
            nxt, jaar = parent[cur]
            min_year = jaar if min_year is None else min(min_year, jaar)
            cur = nxt
        final_map[oude] = (cur, min_year if min_year is not None else 9999)
    return final_map


def harmonize_municipalities(
    df: pd.DataFrame,
    code_col: str = "GemeenteCode",
    year_col: str = "Jaar",
    absolute_cols: Optional[List[str]] = None,
    relative_cols: Optional[List[str]] = None,
    fusies: Optional[Iterable[Dict[str, Any]]] = None,
    splits: Optional[Dict[int, Dict[int, float]]] = None,
    new_code_col: str = "GemeenteCode_harmonized",
    backward: bool = True,
) -> pd.DataFrame:
    """
    Harmonize municipality codes to their 2024 equivalents.

    backward=True  - map all historical codes to the 2024 final code regardless of year.
    backward=False - apply mergers only from the merger year onwards.
    Splits are applied first (weighted); absolute_cols are scaled, relative_cols are not.
    """
    df = df.copy()
    absolute_cols = absolute_cols or []
    relative_cols = relative_cols or []
    fusies        = list(fusies or FUSIES)
    splits        = splits or SPLITS

    df[code_col] = df[code_col].astype(int)

    # 1. Apply splits
    rows = []
    for _, row in df.iterrows():
        code = int(row[code_col])
        if code in splits:
            weights = splits[code]
            total_w = sum(weights.values())
            if not (0.999 <= total_w <= 1.001):
                weights = {k: v / total_w for k, v in weights.items()}
            for new_code, weight in weights.items():
                new_row = row.copy()
                new_row[code_col] = int(new_code)
                for col in absolute_cols:
                    if col in new_row:
                        new_row[col] = new_row[col] * weight
                rows.append(new_row)
        else:
            rows.append(row)
    df_expanded = pd.DataFrame(rows)

    # 2. Build and apply mapping
    final_map           = _resolve_all_sources_to_final(fusies)
    final_map_with_year = _build_direct_final_map(fusies)

    if backward:
        df_expanded[new_code_col] = df_expanded[code_col].apply(
            lambda c: final_map.get(int(c), int(c))
        )
    else:
        def map_row(code: int, jaar: int) -> int:
            if code in final_map_with_year:
                finale, ingang = final_map_with_year[code]
                if ingang is None or jaar >= ingang:
                    return finale
            return code

        df_expanded[new_code_col] = df_expanded.apply(
            lambda r: map_row(int(r[code_col]), int(r[year_col])), axis=1
        )

    return df_expanded


def aggregate_panel(
    df: pd.DataFrame,
    code_col: str = "GemeenteCode_harmonized",
    year_col: str = "Jaar",
    absolute_cols: Optional[List[str]] = None,
    relative_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    absolute_cols = absolute_cols or []
    relative_cols = relative_cols or []
    agg_dict = {
        **{col: "sum"  for col in absolute_cols},
        **{col: "mean" for col in relative_cols},
    }
    return df.groupby([code_col, year_col], as_index=False).agg(agg_dict)


def validate_panel(
    df: pd.DataFrame,
    expected_per_year: int = 342,
    code_col: str = "GemeenteCode_harmonized",
    year_col: str = "Jaar",
) -> pd.Series:
    """Log the number of unique municipalities per year and flag deviations."""
    counts = df.groupby(year_col)[code_col].nunique().sort_index()
    deviations = counts[counts != expected_per_year]
    if deviations.empty:
        log.info("Panel validation passed: %d municipalities per year.", expected_per_year)
    else:
        log.warning("Panel validation - unexpected counts:\n%s", deviations)
    return counts

# Utility helpers

def clean_to_float(x: Any) -> float:
    """Robustly convert any value to float, returning NaN on failure."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (list, tuple, set, np.ndarray)):
        x = list(x)[0] if len(x) else np.nan
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, str):
        s = x.strip()
        if s in {"", "-", "n.v.t.", "NA", "N/A"}:
            return np.nan
        s = s.replace(",", ".").replace(" ", "")
        m = re.search(r"-?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def savefig(name: str) -> None:
    """Save the current figure to PLOTS/ and close it."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    log.info("Saved plot: %s", path)


def to_int64(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.strip().replace("", pd.NA), errors="coerce"
    ).astype("Int64")

# Section functions

def load_ses() -> pd.DataFrame:
    """Load, clean and extrapolate SES scores to 2024."""
    log.info("Loading SES scores ...")
    files = list((RAW / "SES_WOA scores").glob("*.csv"))
    ses = pd.concat([pd.read_csv(f, sep=";") for f in files], ignore_index=True)

    ses_gm = ses[ses["WijkenEnBuurten"].str.startswith("GM")].copy()
    ses_gm["Jaar"]         = ses_gm["Perioden"].str[:4].astype(int)
    ses_gm["SES_score"]    = ses_gm["GemiddeldeScore_29"]
    ses_gm["GemeenteCode"] = ses_gm["WijkenEnBuurten"].str[2:].astype(int)

    ses_sorted = (
        ses_gm[["GemeenteCode", "Jaar", "SES_score"]]
        .sort_values(["Jaar", "GemeenteCode"], ascending=[False, True])
    )

    # Linear extrapolation to 2024
    ses_2023 = ses_sorted[ses_sorted["Jaar"] == 2023].copy()
    ses_2022 = ses_sorted[ses_sorted["Jaar"] == 2022].copy()
    merged = ses_2023.merge(
        ses_2022[["GemeenteCode", "SES_score"]],
        on="GemeenteCode",
        suffixes=("_2023", "_2022"),
    )
    merged["SES_score"] = merged["SES_score_2023"] + (merged["SES_score_2023"] - merged["SES_score_2022"])
    ses_2024 = merged[["GemeenteCode", "SES_score"]].copy()
    ses_2024["Jaar"] = 2024

    ses_final = (
        pd.concat([ses_sorted, ses_2024], ignore_index=True)
        .sort_values(["GemeenteCode", "Jaar"])
        .loc[lambda d: d["Jaar"] != 2014]
        .copy()
    )
    ses_final["SES_score"] = ses_final["SES_score"].round(3)

    out = INTERIM / "ses_woa_clean.csv"
    ses_final.to_csv(out, index=False)
    log.info("SES scores saved -> %s  (%d rows)", out, len(ses_final))
    return ses_final


def load_bevolking() -> pd.DataFrame:
    """Load and prepare population data for harmonization."""
    log.info("Loading population data ...")
    bev = pd.read_csv(
        RAW / "Bevolkingsontwikkeling_per_gemeente_2015_2024.csv",
        sep=";", decimal=",", dtype=str,
    )
    bev["Jaar"]         = bev["Perioden"].str[:4].astype(int)
    bev["GemeenteCode"] = bev["RegioS"].str[2:].astype(int)

    bev_temp = bev.drop(columns=["Perioden", "RegioS", "ID"]).dropna()

    for col in BEV_ABSOLUTE_COLS + BEV_RELATIVE_COLS:
        bev_temp[col] = bev_temp[col].apply(clean_to_float)
    bev_temp["GemeenteCode"] = bev_temp["GemeenteCode"].astype(int)

    log.info("Population data loaded: %d rows.", len(bev_temp))
    return bev_temp


def build_bev_panel(bev_temp: pd.DataFrame) -> pd.DataFrame:
    """Harmonize population data and return a validated panel."""
    log.info("Harmonizing population panel ...")
    harmonized = harmonize_municipalities(
        bev_temp,
        absolute_cols=BEV_ABSOLUTE_COLS,
        relative_cols=BEV_RELATIVE_COLS,
        backward=True,
    )
    panel = aggregate_panel(harmonized, absolute_cols=BEV_ABSOLUTE_COLS, relative_cols=BEV_RELATIVE_COLS)
    validate_panel(panel)
    return panel


def load_kerncijfers() -> pd.DataFrame:
    """Load, harmonize and enrich regional key figures."""
    log.info("Loading regional key figures ...")
    df = pd.concat(
        [
            pd.read_csv(RAW / "regionale_kerncijfers_2014_2017.csv", sep=";"),
            pd.read_csv(RAW / "regionale_kerncijfers_2018_2021.csv", sep=";"),
            pd.read_csv(RAW / "regionale_kerncijfers_2022_2024.csv", sep=";"),
        ],
        ignore_index=True,
    ).sort_values("ID")

    df = df.assign(
        Jaar        = df["Perioden"].str[:4].astype(int),
        GemeenteCode = df["RegioS"].str[2:].astype(int),
    )[["Jaar", "GemeenteCode"] + KERN_ABSOLUTE_COLS + KERN_RELATIVE_COLS].copy()

    harmonized = harmonize_municipalities(
        df,
        absolute_cols=KERN_ABSOLUTE_COLS,
        relative_cols=KERN_RELATIVE_COLS,
        backward=True,
    )
    panel = (
        aggregate_panel(harmonized, absolute_cols=KERN_ABSOLUTE_COLS, relative_cols=KERN_RELATIVE_COLS)
        .dropna()
        .loc[lambda d: d["Jaar"] != 2014]
        .reset_index(drop=True)
    )
    validate_panel(panel)

    # Diagnose any year with fewer than expected municipalities
    all_codes = set(panel["GemeenteCode_harmonized"].dropna().unique())
    for jaar, grp in panel.groupby("Jaar"):
        codes_this_year = set(grp["GemeenteCode_harmonized"].dropna())
        missing = sorted(all_codes - codes_this_year)
        if missing:
            log.warning("Jaar %s: %d missing municipality codes: %s", int(jaar), len(missing), missing)

    # Derived features
    panel["Bevolkingsgroei_per1000_CBS"] = (
        1000 * panel["Bevolkingsgroei_79"] / panel["TotaleBevolking_1"].replace(0, np.nan)
    ).round(3)

    panel["pop_65plus"] = panel[["k_65Tot80Jaar_11", "k_80JaarOfOuder_12"]].sum(axis=1, min_count=1)
    panel["pop_20_65"]  = panel[["k_20Tot25Jaar_8", "k_25Tot45Jaar_9", "k_45Tot65Jaar_10"]].sum(axis=1, min_count=1)
    panel["GrijzeDruk_pct"] = (
        100 * panel["pop_65plus"] / panel["pop_20_65"].replace(0, np.nan)
    ).round(2)

    # Replace raw age counts with population share ratios to avoid multicollinearity
    pop = panel["TotaleBevolking_1"].replace(0, np.nan)
    panel["aandeel_65plus"] = (panel["pop_65plus"] / pop).round(4)
    panel["aandeel_20_65"]  = (panel["pop_20_65"]  / pop).round(4)

    age_cols_to_drop = [
        "k_20Tot25Jaar_8", "k_25Tot45Jaar_9", "k_45Tot65Jaar_10",
        "k_65Tot80Jaar_11", "k_80JaarOfOuder_12", "pop_65plus", "pop_20_65",
    ]
    panel = panel.drop(columns=age_cols_to_drop)
    log.info("Replaced raw age columns with aandeel_65plus and aandeel_20_65.")

    panel["Jaar"] = panel["Jaar"].astype(int)
    panel.rename(columns={"GemeenteCode_harmonized": "GemeenteCode"}, inplace=True)

    log.info("Kerncijfers panel ready: %d rows.", len(panel))
    return panel


def merge_ses_into_predictors(panel: pd.DataFrame, ses_final: pd.DataFrame) -> pd.DataFrame:
    """Merge SES scores into the kerncijfers panel and save predictors."""
    log.info("Merging SES scores into predictors ...")

    panel["GemeenteCode"]     = to_int64(panel["GemeenteCode"])
    panel["Jaar"]             = to_int64(panel["Jaar"].astype(str).str[:4])
    ses_final["GemeenteCode"] = to_int64(ses_final["GemeenteCode"])
    ses_final["Jaar"]         = to_int64(ses_final["Jaar"].astype(str).str[:4])

    ses_agg    = ses_final.groupby(["GemeenteCode", "Jaar"], as_index=False)["SES_score"].mean()
    predictors = panel.merge(ses_agg, on=["GemeenteCode", "Jaar"], how="left")

    out = INTERIM / "predictors.csv"
    predictors.to_csv(out, index=False)
    log.info("Predictors saved -> %s  (%d rows)", out, len(predictors))
    return predictors


def load_kerncijfers_corop() -> pd.DataFrame:
    """
    Load COROP-level regional key figures and return a clean panel.

    Joins onto the main dataset via COROPcode + Jaar to provide
    regional economic context not available at municipality level.
    """
    log.info("Loading COROP-level kerncijfers ...")
    path = RAW / COROP_KERNCIJFERS_FILE
    df   = pd.read_csv(path, sep=";", encoding="latin-1")

    # Parse year from Perioden (e.g. '2015JJ00' -> 2015)
    df["Jaar"] = df["Perioden"].str[:4].astype(int)

    # COROPcode is stored in KoppelvariabeleRegioCode_321
    df["COROPcode"] = df["KoppelvariabeleRegioCode_321"].str.strip()

    # Select and rename relevant columns
    cols_needed = ["COROPcode", "Jaar"] + list(COROP_COLS.keys())
    df = df[cols_needed].copy()
    df = df.rename(columns=COROP_COLS)

    log.info(
        "COROP kerncijfers loaded: %d rows | %d regions | %d-%d",
        len(df), df["COROPcode"].nunique(), df["Jaar"].min(), df["Jaar"].max(),
    )

    # Log NaN coverage
    for col in COROP_COLS.values():
        pct = df[col].notna().mean() * 100
        log.info("  %-35s  %.1f%% non-null", col, pct)

    return df


def load_leegstandsmonitor() -> pd.DataFrame:
    """Load, filter and enrich the vacancy monitor data."""
    log.info("Loading leegstandsmonitor ...")
    df = pd.read_excel(RAW / "Leegstandsmonitor_2015_2024_tabel5.xlsx")
    df.columns = df.columns.str.strip()

    # Keep only municipal rows and clean the code column
    code_col = "Gemeentecode"
    df = df[df[code_col].notna() & (df[code_col].astype(str).str.strip() != "")]
    df[code_col] = pd.to_numeric(
        df[code_col].astype(str).str.strip().str[2:].replace("", pd.NA),
        errors="coerce",
    ).astype("Int64")
    df = df.reset_index(drop=True)

    # Drop geographic grouping columns but KEEP COROPcode for regional feature join
    df = df.drop(columns=["Regio", "Gebiedstype", "Provinciecode", "Provincienaam", "COROPnaam"])

    # Keep count-based rows only
    df = df.loc[
        ~df["Eenheid"].astype(str).str.lower().str.contains("oppervlakte", na=False)
    ].reset_index(drop=True)

    validate_panel(df, 342, "Gemeentecode", "Jaar")

    for col in SUM_COLS:
        df[col] = pd.to_numeric(df[col].replace(".", pd.NA), errors="coerce")

    # Aggregate all commercial functions into a single 'commercieel' row
    df_comm = (
        df[df["Gebruiksfunctie"].isin(COMMERCIEEL_FUNCTIES)]
        .groupby(GROUP_KEYS, observed=True)[SUM_COLS]
        .sum(min_count=1)
        .reset_index()
        .assign(Gebruiksfunctie="commercieel", Eenheid="Aantal")
    )
    for col in df.columns:
        if col not in df_comm.columns:
            df_comm[col] = pd.NA
    df = pd.concat([df, df_comm[df.columns]], ignore_index=True)

    # Filter to relevant functions and compute vacancy rates
    df = (
        df[df["Gebruiksfunctie"].isin(["commercieel", "Woningen"])]
        .sort_values(["Gemeentecode", "Jaar"])
        .reset_index(drop=True)
    )
    df["Onderzoekspopulatie"] = pd.to_numeric(df["Onderzoekspopulatie"], errors="coerce")
    df["Leegstand (aantal)"]  = pd.to_numeric(df["Leegstand (aantal)"],  errors="coerce")
    df["Leegstand, jaar eerder ook al leeg (aantal)"] = pd.to_numeric(
        df["Leegstand, jaar eerder ook al leeg (aantal)"], errors="coerce"
    )
    df["Leegstand, jaar eerder ook al leeg (percentage t.o.v. totale leegstand)"] = pd.to_numeric(
        df["Leegstand, jaar eerder ook al leeg (percentage t.o.v. totale leegstand)"], errors="coerce"
    )

    df["Leegstandspercentage_calc"]     = df["Leegstand (aantal)"] / df["Onderzoekspopulatie"]
    df["Leegstandspercentage_calc_pct"] = df["Leegstandspercentage_calc"] * 100

    # Structural vacancy: persistent vacancy as share of total stock
    # (different from the existing percentage which is share of total vacancy)
    df["structurele_leegstand_pct"] = (
        df["Leegstand, jaar eerder ook al leeg (aantal)"] / df["Onderzoekspopulatie"] * 100
    ).round(3)

    # Investigate 100% vacancy outlier
    rows_100 = df[df["Leegstandspercentage_calc_pct"] == 100]
    if not rows_100.empty:
        log.warning(
            "Found %d row(s) with 100%% vacancy:\n%s",
            len(rows_100),
            rows_100[["Gemeentenaam", "Jaar", "Gebruiksfunctie", "Onderzoekspopulatie", "Leegstand (aantal)"]].to_string(),
        )

    # Investigate Alphen-Chaam as a potential small-sample outlier
    alphen = df[df["Gemeentenaam"] == "Alphen-Chaam"]
    if not alphen.empty:
        log.info(
            "Alphen-Chaam breakdown:\n%s",
            alphen[["Jaar", "Gebruiksfunctie", "Onderzoekspopulatie", "Leegstand (aantal)", "Leegstandspercentage_calc_pct"]].to_string(),
        )

    # Drop rows where Onderzoekspopulatie is too small to produce a meaningful rate
    before = len(df)
    df = df[df["Onderzoekspopulatie"] >= MIN_ONDERZOEKSPOPULATIE].reset_index(drop=True)
    log.info(
        "Dropped %d rows with Onderzoekspopulatie < %d  (%d rows remaining).",
        before - len(df), MIN_ONDERZOEKSPOPULATIE, len(df),
    )

    out = INTERIM / "labels.csv"
    df.to_csv(out, index=False)
    log.info("Labels saved -> %s  (%d rows)", out, len(df))
    return df


def merge_labels_and_predictors() -> pd.DataFrame:
    """Merge labels and predictors into the final processed dataset."""
    log.info("Merging labels and predictors ...")
    labels     = pd.read_csv(INTERIM / "labels.csv")
    predictors = pd.read_csv(INTERIM / "predictors.csv")

    merged = labels.merge(
        predictors,
        left_on=["Gemeentecode", "Jaar"],
        right_on=["GemeenteCode", "Jaar"],
        how="left",
    )

    out = PROCESSED / "merged.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    log.info("Merged dataset saved -> %s  shape=%s", out, merged.shape)
    return merged


def prepare_modelling_dataset(merged: pd.DataFrame, corop_kern: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset and produce a model-ready file.

    New in this version:
        - Urbanisation level encoded as ordinal 1-5
        - COROP-level economic features joined via COROPcode + Jaar
        - COROP region one-hot encoded into 40 dummies

    Checks performed (all logged):
        1. NaN target rows        — dropped
        2. Zero target rows       — logged for awareness, kept
        3. NaN in feature columns — logged per column, rows dropped
        4. Duplicate rows         — dropped
        5. Row counts per year and property type — logged as sanity check
    """
    log.info("Preparing model-ready dataset ...")
    df = merged.copy()
    start_rows = len(df)

    # Use the raw column name before renaming
    raw_target     = "Leegstandspercentage_calc_pct"
    raw_predictors = [
        "TotaleBevolking_1",
        "aandeel_20_65",
        "Koopwoningen_94",
        "HuurwoningenVanWoningcorporatie_95",
        "HuurwoningenVanOverigeVerhuurders_96",
        "GemiddeldeWOZWaardeVanWoningen_98",
        "Bevolkingsgroei_per1000_CBS",
        "GrijzeDruk_pct",
        "SES_score",
        "Leegstand, jaar eerder ook al leeg (aantal)",
        "structurele_leegstand_pct",
        # New municipality-level features
        "Werkloosheid_159",
        "Migratiesaldo_76",
        "BedrijfsvestigingenTotaal_168",
        "Bevolkingsdichtheid_57",
        "AfstandTotTreinstation_238",
        "GemiddeldeHuishoudensgrootte_89",
    ]

    # ── A. Urbanisation level → ordinal encoding ─────────────────────────────
    df["urbanisation_level_enc"] = df["Stedelijkheid"].map(URBANISATION_ORDER)
    missing_urban = df["urbanisation_level_enc"].isna().sum()
    if missing_urban > 0:
        log.warning("  %d rows with unrecognised urbanisation level — set to NaN.", missing_urban)
    log.info("  Urbanisation level encoded (1=Niet stedelijk ... 5=Zeer sterk stedelijk).")

    # ── B. Join COROP-level features ─────────────────────────────────────────
    # Ensure COROPcode is clean string for join
    df["COROPcode"] = df["COROPcode"].astype(str).str.strip()
    corop_kern["COROPcode"] = corop_kern["COROPcode"].astype(str).str.strip()
    corop_kern["Jaar"]      = corop_kern["Jaar"].astype(int)
    df["Jaar"]              = df["Jaar"].astype(int)

    before_join = len(df)
    df = df.merge(corop_kern, on=["COROPcode", "Jaar"], how="left")
    log.info(
        "  COROP features joined: %d rows before, %d after (delta=%d).",
        before_join, len(df), len(df) - before_join,
    )

    # Check COROP join coverage
    for col in COROP_COLS.values():
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            log.warning("  %d NaN rows in COROP feature '%s' after join.", n_missing, col)

    # ── C. COROP one-hot encoding (40 regional dummies) ──────────────────────
    df["COROPcode"] = df["COROPcode"].astype(str).str.strip()
    corop_dummies   = pd.get_dummies(df["COROPcode"], prefix="corop", drop_first=True, dtype=float)
    corop_dummy_cols = corop_dummies.columns.tolist()
    df = pd.concat([df, corop_dummies], axis=1)
    log.info("  COROP one-hot encoded: %d dummy columns added.", len(corop_dummy_cols))

    # ── D. Zero-fill where research was conducted ─────────────────────────────
    has_research = df["Onderzoekspopulatie"].notna() & (df["Onderzoekspopulatie"] > 0)
    cols_to_zero_fill = [
        raw_target,
        "Leegstand, jaar eerder ook al leeg (aantal)",
        "structurele_leegstand_pct",
    ]
    for col in cols_to_zero_fill:
        if col in df.columns:
            mask   = has_research & df[col].isna()
            filled = mask.sum()
            if filled > 0:
                df.loc[mask, col] = 0.0
                log.info("  Zero-filled %d NaN rows in '%s' where research_population > 0.", filled, col)

    # ── 1. Drop rows where target is still NaN after zero-fill ───────────────
    nan_target = df[raw_target].isna().sum()
    df = df.dropna(subset=[raw_target])
    log.info("  NaN target rows dropped: %d", nan_target)

    # ── 2. Log zero-target rows (kept) ───────────────────────────────────────
    zero_target = (df[raw_target] == 0).sum()
    log.info("  Zero target rows (kept): %d", zero_target)

    # ── 3. Check and drop NaN predictor rows ─────────────────────────────────
    all_raw_predictors = raw_predictors + ["urbanisation_level_enc"] + list(COROP_COLS.values())
    missing_per_col = df[all_raw_predictors].isna().sum()
    missing_per_col = missing_per_col[missing_per_col > 0]
    if not missing_per_col.empty:
        log.warning("  NaN values found in predictor columns:\n%s", missing_per_col.to_string())
    before = len(df)
    df = df.dropna(subset=all_raw_predictors)
    log.info("  Rows dropped due to NaN predictors: %d", before - len(df))

    # ── 4. Drop duplicate rows ───────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates()
    log.info("  Duplicate rows dropped: %d", before - len(df))

    # ── 4b. Lagged vacancy rate ───────────────────────────────────────────────
    # Sort by municipality, property type, and year so shift(1) looks back
    # exactly one year within each municipality × property type group.
    # 2015 rows get NaN (no prior year) and are dropped here.
    df = df.sort_values(["Gemeentecode", "Gebruiksfunctie", "Jaar"]).copy()
    df["vacancy_rate_lag1"] = (
        df.groupby(["Gemeentecode", "Gebruiksfunctie"])[raw_target].shift(1)
    )
    before_lag = len(df)
    df = df.dropna(subset=["vacancy_rate_lag1"])
    log.info(
        "  Lagged vacancy added. Dropped %d rows (2015, no prior year).",
        before_lag - len(df),
    )

    # ── 5. Sanity check ──────────────────────────────────────────────────────
    counts_jaar = df.groupby("Jaar").size()
    counts_func = df.groupby("Gebruiksfunctie").size()
    log.info("  Rows per year:\n%s", counts_jaar.to_string())
    log.info("  Rows per property type:\n%s", counts_func.to_string())

    log.info(
        "Model-ready dataset: %d -> %d rows (dropped %d total).",
        start_rows, len(df), start_rows - len(df),
    )

    # ── Rename all columns to clean English names ────────────────────────────
    df = df.rename(columns=COLUMN_RENAME)

    # Save updated PREDICTORS list including COROP dummies to a text file
    # so model scripts can load it without hardcoding
    all_predictors = PREDICTORS + corop_dummy_cols
    pred_path = PROCESSED / "predictors_list.txt"
    pred_path.write_text("\n".join(all_predictors))
    log.info("  Full predictor list (%d features) saved -> %s", len(all_predictors), pred_path)

    out = PROCESSED / "model_ready.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Model-ready dataset saved -> %s", out)
    return df, all_predictors


def run_eda(df_raw: pd.DataFrame) -> None:
    """Run all EDA plots and save them to the plots folder."""
    log.info("Running EDA ...")
    df = df_raw.dropna(subset=[TARGET, "total_population"]).copy()
    log.info(
        "EDA dataset: %d rows | %d-%d | %d municipalities",
        len(df), df["year"].min(), df["year"].max(), df["municipality_name"].nunique(),
    )

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.dpi"] = 130

    # 1. Target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Target distribution", fontsize=13, fontweight="bold")
    axes[0].hist(df[TARGET], bins=60, color="#5DCAA5", edgecolor="white")
    axes[0].set(xlabel="Leegstand %", ylabel="Count", title="Raw")
    axes[1].hist(np.log1p(df[TARGET]), bins=60, color="#7F77DD", edgecolor="white")
    axes[1].set(xlabel="log1p(Leegstand %)", title="Log-transformed")
    plt.tight_layout()
    savefig("eda_01_target_distribution.png")

    # 2. Target by property type
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Vacancy % by property type", fontsize=13, fontweight="bold")
    for func, color in zip(["Woningen", "commercieel"], ["#5DCAA5", "#D85A30"]):
        ax.hist(df[df["property_type"] == func][TARGET], bins=40, alpha=0.6,
                label=func, color=color, edgecolor="white")
    ax.set(xlabel="Vacancy %", ylabel="Count")
    ax.legend()
    plt.tight_layout()
    savefig("eda_02_by_gebruiksfunctie.png")

    # 3. Target by urbanisation level
    order = ["Niet stedelijk", "Weinig stedelijk", "Matig stedelijk", "Sterk stedelijk", "Zeer sterk stedelijk"]
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Vacancy % by urbanisation level", fontsize=13, fontweight="bold")
    sns.boxplot(data=df, x="urbanisation_level", y=TARGET, hue="urbanisation_level", order=order, palette="Blues", legend=False, ax=ax, fliersize=2)
    ax.set(xlabel="", ylabel="Vacancy %")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    savefig("eda_03_by_stedelijkheid.png")

    # 4. Trend over time
    yearly = df.groupby("year")[TARGET].agg(["mean", "median"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Vacancy trend over time", fontsize=13, fontweight="bold")
    axes[0].plot(yearly.index, yearly["mean"],   marker="o", label="Mean",   color="#7F77DD")
    axes[0].plot(yearly.index, yearly["median"], marker="s", label="Median", color="#1D9E75", linestyle="--")
    axes[0].set(xlabel="Year", ylabel="Vacancy %", title="Overall (all property types)")
    axes[0].legend()
    for func, color in zip(["Woningen", "commercieel"], ["#5DCAA5", "#D85A30"]):
        sub = df[df["property_type"] == func].groupby("year")[TARGET].mean()
        axes[1].plot(sub.index, sub.values, marker="o", label=func, color=color)
    axes[1].set(xlabel="Year", ylabel="Vacancy %", title="Split by property type")
    axes[1].legend()
    plt.tight_layout()
    savefig("eda_04_trend_over_time.png")

    # 5. Correlation with target
    corr = df[PREDICTORS + [TARGET]].corr()[TARGET].drop(TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Predictor correlations with target", fontsize=13, fontweight="bold")
    ax.barh(corr.index, corr.values, color=["#D85A30" if v > 0 else "#378ADD" for v in corr.values])
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Pearson r")
    plt.tight_layout()
    savefig("eda_05_correlations.png")

    # 6. Scatter: top correlated predictors vs target
    top6 = corr.head(3).index.tolist() + corr.tail(3).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Scatter: predictors vs vacancy %", fontsize=13, fontweight="bold")
    for ax, col in zip(axes.flat, top6):
        sample = df[[col, TARGET]].dropna().sample(min(2000, len(df)), random_state=42)
        ax.scatter(sample[col], sample[TARGET], alpha=0.2, s=8, color="#7F77DD")
        m, b, *_ = stats.linregress(sample[col], sample[TARGET])
        xr = np.array([sample[col].min(), sample[col].max()])
        ax.plot(xr, m * xr + b, color="#D85A30", linewidth=1.5)
        ax.set(xlabel=col, ylabel="Vacancy %", title=f"r = {corr[col]:.3f}")
        ax.xaxis.label.set_size(9)
        ax.title.set_size(10)
    plt.tight_layout()
    savefig("eda_06_scatterplots.png")

    # 7. Predictor correlation heatmap
    corr_matrix = df[PREDICTORS].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.suptitle("Predictor correlation heatmap", fontsize=13, fontweight="bold")
    sns.heatmap(
        corr_matrix, mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
        annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        linewidths=0.4, annot_kws={"size": 7}, ax=ax,
    )
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    savefig("eda_07_heatmap.png")

    # 8. Top & bottom municipalities
    mean_by_gemeente = df.groupby("municipality_name")[TARGET].mean()
    top10 = mean_by_gemeente.sort_values(ascending=False).head(10)
    bot10 = mean_by_gemeente.sort_values().head(10)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Municipalities by average vacancy %", fontsize=13, fontweight="bold")
    axes[0].barh(top10.index[::-1], top10.values[::-1], color="#D85A30")
    axes[0].set(xlabel="Mean vacancy %", title="Highest 10")
    axes[1].barh(bot10.index[::-1], bot10.values[::-1], color="#5DCAA5")
    axes[1].set(xlabel="Mean vacancy %", title="Lowest 10")
    plt.tight_layout()
    savefig("eda_08_municipalities.png")

    # 9. Outlier detection (IQR)
    Q1, Q3    = df[TARGET].quantile([0.25, 0.75])
    iqr_limit = Q3 + 1.5 * (Q3 - Q1)
    outliers  = df[df[TARGET] > iqr_limit]
    log.info("Outliers (IQR): %d rows (%.1f%%)", len(outliers), 100 * len(outliers) / len(df))
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle("Outliers in target (IQR method)", fontsize=13, fontweight="bold")
    ax.scatter(df.index, df[TARGET], s=4, alpha=0.3, color="#7F77DD", label="Normal")
    ax.scatter(outliers.index, outliers[TARGET], s=10, alpha=0.7,
               color="#D85A30", label=f"Outliers (n={len(outliers)})")
    ax.axhline(iqr_limit, color="#D85A30", linestyle="--", linewidth=1)
    ax.set_ylabel("Vacancy %")
    ax.legend()
    plt.tight_layout()
    savefig("eda_09_outliers.png")

    # 10. Missing data heatmap by year
    missing_by_year = df.groupby("year")[PREDICTORS].apply(lambda g: g.isnull().mean() * 100)
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Missing data % by year (predictors)", fontsize=13, fontweight="bold")
    sns.heatmap(missing_by_year.T, annot=True, fmt=".1f", cmap="Oranges",
                linewidths=0.3, annot_kws={"size": 7}, ax=ax)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    savefig("eda_10_missing_by_year.png")

    log.info("EDA complete - all plots saved to %s/", PLOTS)

# Main

def main() -> None:
    # 1. SES scores
    ses_final = load_ses()

    # 2 & 3. Population data + harmonization
    bev_temp = load_bevolking()
    build_bev_panel(bev_temp)

    # 4. Regional key figures + derived features
    kern_panel = load_kerncijfers()

    # Merge SES into predictors
    merge_ses_into_predictors(kern_panel, ses_final)

    # 4b. COROP-level regional key figures
    corop_kern = load_kerncijfers_corop()

    # 5. Leegstandsmonitor (labels)
    load_leegstandsmonitor()

    # 6. Merge labels + predictors
    merged = merge_labels_and_predictors()

    # 7. Prepare model-ready dataset (now includes COROP features + urbanisation encoding)
    model_df, all_predictors = prepare_modelling_dataset(merged, corop_kern)

    log.info("Final predictor list (%d features):\n  %s", len(all_predictors), "\n  ".join(all_predictors))

    # 8. EDA
    run_eda(model_df)


if __name__ == "__main__":
    main()