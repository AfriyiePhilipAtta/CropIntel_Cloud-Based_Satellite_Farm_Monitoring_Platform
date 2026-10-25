"""Gap-fill missing Sentinel-2 NDVI readings using Sentinel-1 radar, for the
real CropIntel acquisition-history files.

WHAT THIS IS AND IS NOT
------------------------
This module is additive and read-only with respect to the pipeline's real
observation records. It never writes to s2_ndvi_history.csv or
s1_radar_history.csv -- transform_data.py already documents both of those as
records of real, dated satellite acquisitions only, and field_report.py's
Field Trends chart currently relies on that guarantee (it breaks the plotted
line across any gap, on purpose, so a cloud gap is never shown as a smooth
slope). This module writes its predictions to a separate file,
s2_ndvi_gap_filled.csv, so that guarantee stays true for the two files that
already exist.

Every value in the output file is a MODEL PREDICTION, not a measurement.
Nothing here is wired into decision_engine.py. It must not be allowed to
change action_priority, overall_confidence_client, or any other
client-facing decision field -- it exists only to optionally draw extra
(visually distinct) points on the Field Trends chart. This mirrors the
existing RVI_SIGNAL_ENABLED pattern in config.py / decision_engine.py:
default OFF, additive only, gated behind one flag.

METHOD
------
Random forest (classical ML, not deep learning), trained per farm on:
  - the two nearest real NDVI readings around the target date
    (last_clear_ndvi / next_clear_ndvi and how many days away each is)
  - day-of-year seasonality (doy_sin / doy_cos)
  - the Sentinel-1 VV/VH backscatter closest in time to the target date
This is the same classical Sentinel-1/2 fusion idea discussed earlier for
this project, following Lasko (2022). I do not have a verified full
citation (journal, DOI) for that reference in front of me right now --
confirm it before it appears anywhere client-facing.

BEFORE ENABLING config.NDVI_GAP_FILL_ENABLED
---------------------------------------------
Run validate_against_held_out_readings() on THIS pipeline's real
s2_ndvi_history.csv / s1_radar_history.csv. If the random forest does not
clearly beat the plain linear-interpolation baseline on real farms, do not
turn the flag on -- that would mean a model guess is not actually better
than a straight line between two real readings for this data, which is
exactly the outcome the starter version of this script hit on synthetic
data. This has NOT yet been run against real CropIntel data (none was
available when this file was written) -- treat every number this script
produces as unverified until you have run it yourself.

REAL-SCHEMA DIFFERENCES FROM THE FIRST DRAFT OF THIS SCRIPT
-------------------------------------------------------------
- Column names follow transform_data.py's real history files: farm_id /
  acquisition_date / ndvi / ndmi / ndre (s2_ndvi_history.csv) and farm_id /
  acquisition_date / vv_db / vh_db / rvi / relative_orbit / orbit_pass
  (s1_radar_history.csv) -- not the field_id / vv / vh used in the first
  draft.
- Sentinel-1 and Sentinel-2 acquisitions do NOT fall on the same calendar
  dates in reality (the earlier synthetic placeholder data assumed a single
  shared date axis, which real acquisitions never give you). This version
  matches each S1 date to the nearest S2 date within a tolerance window
  using pandas.merge_asof, instead of assuming both instruments were read
  on the same day.
- There is no ndvi_texture feature here, because extract_data.py does not
  currently compute one. Leaving it out rather than inventing a stand-in.
- Only NDVI is gap-filled. NDMI and NDRE are left alone deliberately: the
  physical relationship between C-band VV/VH backscatter and NDVI (canopy
  structure) is far better established in the remote-sensing literature
  than the relationship between backscatter and NDMI/NDRE (moisture and
  chlorophyll/nitrogen content). Extending this to NDMI/NDRE would need its
  own separate validation, not an assumption that whatever works for NDVI
  transfers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from config import PROCESSED_DIR

try:
    # Reused from the existing radar-continuity feature as a starting point
    # for "how old a radar reading can be and still count as current" --
    # NOT validated for this specific use. If S1_GAP_REFERENCE_MAX_DAYS ever
    # changes meaning in config.py, revisit this import; consider giving gap
    # filling its own named constant (e.g. NDVI_GAP_FILL_RADAR_TOLERANCE_DAYS)
    # in config.py instead of borrowing this one, once validated.
    from config import S1_GAP_REFERENCE_MAX_DAYS as RADAR_MATCH_TOLERANCE_DAYS
except ImportError:
    RADAR_MATCH_TOLERANCE_DAYS = 12

S2_HISTORY_FILE = PROCESSED_DIR / "s2_ndvi_history.csv"
S1_HISTORY_FILE = PROCESSED_DIR / "s1_radar_history.csv"
OUTPUT_FILE = PROCESSED_DIR / "s2_ndvi_gap_filled.csv"

FEATURE_COLUMNS = [
    "vv_db", "vh_db", "doy_sin", "doy_cos",
    "last_clear_ndvi", "days_since_last_clear",
    "next_clear_ndvi", "days_to_next_clear",
]


def load_history(farm_id: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pipeline's real S2 and S1 acquisition-history files.

    Raises FileNotFoundError with a clear message if transform_data.py has
    not been run yet (these files are built by _append_s2_acquisition /
    _append_s1_acquisition / _merge_s2_history_from_raw /
    _merge_s1_history_from_raw inside transform_data.py).
    """
    if not S2_HISTORY_FILE.exists() or not S1_HISTORY_FILE.exists():
        raise FileNotFoundError(
            "s2_ndvi_history.csv / s1_radar_history.csv not found under "
            f"{PROCESSED_DIR}. Run transform_data.py at least once first."
        )
    s2 = pd.read_csv(S2_HISTORY_FILE, parse_dates=["acquisition_date"])
    s1 = pd.read_csv(S1_HISTORY_FILE, parse_dates=["acquisition_date"])
    if farm_id is not None:
        s2 = s2[s2["farm_id"] == farm_id].copy()
        s1 = s1[s1["farm_id"] == farm_id].copy()
    return (
        s2.sort_values(["farm_id", "acquisition_date"]).reset_index(drop=True),
        s1.sort_values(["farm_id", "acquisition_date"]).reset_index(drop=True),
    )


def build_daily_grid(
    s2: pd.DataFrame, s1: pd.DataFrame, tolerance_days: int = RADAR_MATCH_TOLERANCE_DAYS
) -> pd.DataFrame:
    """One row per real S1 acquisition, per farm, carrying the NDVI reading
    (if any) taken within `tolerance_days` of that same date.

    Rows where "ndvi" is NaN are candidate gap-fill targets: CropIntel has a
    radar reading close to that date but no matching optical reading close
    enough to trust as-is. Rows where "ndvi" is present are used for
    validation (there IS a real answer to check the model against).
    """
    frames = []
    for farm_id, s1_farm in s1.groupby("farm_id"):
        s2_farm = s2.loc[s2["farm_id"] == farm_id, ["acquisition_date", "ndvi"]]
        merged = pd.merge_asof(
            s1_farm[["acquisition_date", "vv_db", "vh_db"]].sort_values("acquisition_date"),
            s2_farm.sort_values("acquisition_date"),
            on="acquisition_date",
            direction="nearest",
            tolerance=pd.Timedelta(days=tolerance_days),
        )
        merged.insert(0, "farm_id", farm_id)
        frames.append(merged)
    daily = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["farm_id", "acquisition_date", "vv_db", "vh_db", "ndvi"]
    )
    return daily.rename(columns={"acquisition_date": "date"})


def build_features(daily: pd.DataFrame, s2_for_features: pd.DataFrame) -> pd.DataFrame:
    """Attach seasonality plus nearest-real-NDVI features to each row.

    `s2_for_features` is passed in separately from the ground truth so that
    validate_against_held_out_readings() can hide some real readings from
    the feature-building step without also hiding them from the eventual
    accuracy check -- using the same list for both would let the model see
    a held-out answer disguised as a "nearest clear reading" feature.
    """
    daily = daily.sort_values(["farm_id", "date"]).reset_index(drop=True)
    doy = pd.to_datetime(daily["date"]).dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    pieces = []
    for farm_id, group in daily.groupby("farm_id"):
        group = group.sort_values("date").reset_index(drop=True)
        clear = (
            s2_for_features.loc[s2_for_features["farm_id"] == farm_id, ["acquisition_date", "ndvi"]]
            .dropna(subset=["ndvi"])
            .sort_values("acquisition_date")
        )

        if clear.empty:
            group["last_clear_ndvi"] = np.nan
            group["days_since_last_clear"] = np.nan
            group["next_clear_ndvi"] = np.nan
            group["days_to_next_clear"] = np.nan
            pieces.append(group)
            continue

        last = pd.merge_asof(
            group[["date"]], clear, left_on="date", right_on="acquisition_date", direction="backward"
        )
        nxt = pd.merge_asof(
            group[["date"]], clear, left_on="date", right_on="acquisition_date", direction="forward"
        )

        group["last_clear_ndvi"] = last["ndvi"].to_numpy()
        group["days_since_last_clear"] = (
            (group["date"] - last["acquisition_date"]).dt.days.to_numpy()
        )
        group["next_clear_ndvi"] = nxt["ndvi"].to_numpy()
        group["days_to_next_clear"] = (
            (nxt["acquisition_date"] - group["date"]).dt.days.to_numpy()
        )
        pieces.append(group)

    return pd.concat(pieces, ignore_index=True)


def validate_against_held_out_readings(
    s2: pd.DataFrame, s1: pd.DataFrame, hide_fraction: float = 0.2, seed: int = 0
) -> tuple[dict, RandomForestRegressor]:
    """Hide some real NDVI readings, predict them, and compare against a
    plain linear-interpolation baseline. This is the step that decides
    whether NDVI_GAP_FILL_ENABLED should ever be turned on -- do not skip
    it and do not trust this script's predictions without running it.

    Returns (metrics, fitted_model). metrics includes both models' MAE and
    R^2, plus a warning message when the random forest does not clearly
    beat interpolation.
    """
    rng = np.random.default_rng(seed)
    daily = build_daily_grid(s2, s1)
    known = daily.dropna(subset=["ndvi"]).reset_index(drop=True)

    if len(known) < 20:
        raise ValueError(
            f"Only {len(known)} S1 readings have a matching real NDVI reading "
            "close enough in time to validate against. Gather more history "
            "before trusting this method for these farms."
        )

    hide_mask = rng.random(len(known)) < hide_fraction
    hidden = known[hide_mask].copy()
    visible_dates = known.loc[~hide_mask, ["farm_id", "date", "ndvi"]].rename(
        columns={"date": "acquisition_date"}
    )
    # s2 rows the model is allowed to see when building features: every real
    # S2 reading EXCEPT the ones deliberately hidden for this test.
    hidden_keys = set(zip(hidden["farm_id"], hidden["date"]))
    s2_visible = s2[
        ~s2.apply(lambda r: (r["farm_id"], r["acquisition_date"]) in hidden_keys, axis=1)
    ]

    feat = build_features(daily, s2_visible)
    train = feat.dropna(subset=FEATURE_COLUMNS + ["ndvi"])
    train = train[~train.set_index(["farm_id", "date"]).index.isin(hidden.set_index(["farm_id", "date"]).index)]

    test = feat.merge(hidden[["farm_id", "date"]], on=["farm_id", "date"], how="inner").dropna(
        subset=FEATURE_COLUMNS
    )

    if train.empty or test.empty:
        raise ValueError(
            "Not enough rows with every feature present to run this validation. "
            "This usually means too few farms have both S1 and S2 history yet."
        )

    model = RandomForestRegressor(n_estimators=400, random_state=seed)
    model.fit(train[FEATURE_COLUMNS], train["ndvi"])
    predicted = model.predict(test[FEATURE_COLUMNS])
    actual = test["ndvi"].to_numpy()

    baseline = test[["last_clear_ndvi", "next_clear_ndvi"]].mean(axis=1).to_numpy()

    metrics = {
        "n_validation_points": int(len(test)),
        "random_forest_mae": float(mean_absolute_error(actual, predicted)),
        "random_forest_r2": float(r2_score(actual, predicted)) if len(test) > 1 else float("nan"),
        "interpolation_mae": float(mean_absolute_error(actual, baseline)),
        "interpolation_r2": float(r2_score(actual, baseline)) if len(test) > 1 else float("nan"),
    }
    if metrics["random_forest_mae"] >= metrics["interpolation_mae"]:
        metrics["warning"] = (
            "Random forest did not beat plain linear interpolation on this "
            "pipeline's real data. Do not enable NDVI_GAP_FILL_ENABLED until "
            "this changes -- a model guess that is no better than a straight "
            "line is not worth the added complexity or the risk of a "
            "confidently-wrong point on a farmer-facing chart."
        )
    return metrics, model


def fill_gaps(s2: pd.DataFrame, s1: pd.DataFrame, model: RandomForestRegressor) -> pd.DataFrame:
    """Predict NDVI only for real S1 dates that have no matching NDVI reading.

    Returns farm_id, date, ndvi_gap_filled, source ("gap_filled"), for
    writing to s2_ndvi_gap_filled.csv. Never touches a date that already has
    a real NDVI reading.
    """
    daily = build_daily_grid(s2, s1)
    feat = build_features(daily, s2)
    targets = feat[feat["ndvi"].isna()].dropna(subset=FEATURE_COLUMNS).copy()
    if targets.empty:
        return pd.DataFrame(columns=["farm_id", "date", "ndvi_gap_filled", "source"])
    targets["ndvi_gap_filled"] = model.predict(targets[FEATURE_COLUMNS])
    targets["source"] = "gap_filled"
    return targets[["farm_id", "date", "ndvi_gap_filled", "source"]].reset_index(drop=True)


def run(hide_fraction: float = 0.2, seed: int = 0) -> pd.DataFrame | None:
    """Validate, print the honest result, and only write output if the
    random forest actually beat interpolation. Intended to be called from
    transform_data.py's main(), gated behind config.NDVI_GAP_FILL_ENABLED.
    """
    s2, s1 = load_history()
    metrics, model = validate_against_held_out_readings(s2, s1, hide_fraction, seed)
    print("NDVI gap-fill validation:", metrics)
    if "warning" in metrics:
        print(metrics["warning"])
        return None

    filled = fill_gaps(s2, s1, model)
    filled.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {len(filled)} gap-filled NDVI points to {OUTPUT_FILE}")
    return filled


if __name__ == "__main__":
    run()
