"""
Transform raw CropIntel data into crop-condition and Water & Climate indicators.

This module keeps the original crop-condition early-warning status and adds a
separate Water & Climate interpretation layer. The water/climate indicators are
screening tools, not automatic agronomic diagnoses or irrigation prescriptions.

Execution steps
---------------
1. Read all current, historical, forecast, climate and management tables.
2. Standardize farm identifiers so the tables can be matched reliably.
3. For each farm, compare recent and previous NDVI, NDMI and NDRE windows.
4. Read the NDVI raster and calculate the low-condition share and hectares.
5. Summarize forecast rainfall without changing missing data to zero.
6. Combine Sentinel-2 NDMI with rainfall, forecast and climate background.
7. Add management context without filling unrecorded events.
8. Run separate crop-condition, nutrition and irrigation screening rules.
9. Merge all results into one farm record.
10. Save ``cropintel_monitoring.csv`` for the dashboard modules.
"""

from __future__ import annotations

from datetime import timedelta
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask

from config import (
    ABSOLUTE_NDVI_FLOOR,
    CROP_WATER_PROFILES,
    HEAT_EXPOSURE_HIGH_DAYS_30D,
    HEAT_EXPOSURE_MONITOR_DAYS_30D,
    LOW_AREA_FIELD_CHECK_SHARE,
    NDVI_LOW_BREAK,
    LOW_AREA_MONITOR_SHARE,
    NDRE_ASSESS_DROP,
    NDRE_MONITOR_DROP,
    NDMI_FIELD_CHECK_DROP,
    NDMI_MONITOR_DROP,
    NUTRIENT_LOW_AREA_ASSESS_SHARE,
    NUTRIENT_LOW_AREA_MONITOR_SHARE,
    LOW_FORECAST_RAIN_7D_MM,
    NDVI_FIELD_CHECK_DROP,
    NDVI_GAP_FILL_ENABLED,
    NDVI_MONITOR_DROP,
    NDVI_SLOPE_MONITOR,
    PROCESSED_DIR,
    RAW_DIR,
    RELATIVE_NDVI_DROP,
    S1_MIN_BASELINE_SCENES,
    S1_VH_CHANGE_THRESHOLD_DB,
    S1_VV_CHANGE_THRESHOLD_DB,
    S2_MIN_VALID_COVERAGE,
    WATER_SUPPLY_RATIO_HIGH_RISK,
    WATER_SUPPLY_RATIO_MONITOR,
    WEEKLY_FRESHNESS_DAYS,
)

from management_loader import load_management_records, summarize_management_records
from decision_engine import build_decision_fields
from farm_loader import load_farms


def safe_float(value):
    """Convert one value to float and use NaN for invalid or missing input."""
    try:
        return np.nan if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return np.nan


def observation_age_days(observation_date, run_datetime):
    """Return whole calendar days between an acquisition and the report."""
    observation = pd.to_datetime(observation_date, errors="coerce", utc=True)
    report = pd.to_datetime(run_datetime, errors="coerce", utc=True)
    if pd.isna(observation) or pd.isna(report):
        return np.nan
    age = int((report.normalize() - observation.normalize()).days)
    return age if age >= 0 else np.nan


def satellite_freshness_metrics(farm):
    """Classify optical freshness and conservative Sentinel-1 change evidence.

    Sentinel-1 is used as a separate structural-change signal. It never fills
    or predicts NDVI, NDMI or NDRE. A radar comparison is accepted only when
    the newest acquisition is recent, both VV and VH changes are available,
    and the same-orbit baseline contains enough earlier scenes.
    """
    run_datetime = farm.get("run_datetime")
    s2_age = observation_age_days(farm.get("latest_s2_date"), run_datetime)
    s1_age = observation_age_days(farm.get("latest_s1_date"), run_datetime)
    latest_s2_coverage = safe_float(farm.get("s2_latest_usable_coverage"))
    s2_fresh = (
        pd.notna(s2_age)
        and s2_age <= WEEKLY_FRESHNESS_DAYS
        and pd.notna(latest_s2_coverage)
        and latest_s2_coverage >= S2_MIN_VALID_COVERAGE
    )
    s1_fresh = pd.notna(s1_age) and s1_age <= WEEKLY_FRESHNESS_DAYS

    baseline_count = safe_float(farm.get("s1_baseline_scene_count"))
    vv_change = safe_float(farm.get("vv_change_db"))
    vh_change = safe_float(farm.get("vh_change_db"))
    baseline_ready = (
        pd.notna(baseline_count)
        and baseline_count >= S1_MIN_BASELINE_SCENES
        and pd.notna(vv_change)
        and pd.notna(vh_change)
    )

    unusual = np.nan
    if pd.isna(s1_age):
        radar_status = "NO_DATA"
    elif not s1_fresh:
        radar_status = "STALE"
    elif not baseline_ready:
        radar_status = "INSUFFICIENT_BASELINE"
    else:
        unusual = bool(
            abs(vv_change) >= S1_VV_CHANGE_THRESHOLD_DB
            or abs(vh_change) >= S1_VH_CHANGE_THRESHOLD_DB
        )
        radar_status = "UNUSUAL_CHANGE" if unusual else "NO_UNUSUAL_CHANGE"

    if s2_fresh:
        weekly_status = "FRESH_OPTICAL"
    elif s1_fresh and radar_status in {"UNUSUAL_CHANGE", "NO_UNUSUAL_CHANGE"}:
        weekly_status = "FRESH_RADAR"
    elif s1_fresh:
        weekly_status = "FRESH_RADAR_WITHOUT_BASELINE"
    else:
        weekly_status = "NO_FRESH_SATELLITE"

    return {
        "s2_age_days": s2_age,
        "s1_age_days": s1_age,
        "s2_is_fresh": bool(s2_fresh),
        "s1_is_fresh": bool(s1_fresh),
        "radar_baseline_ready": bool(baseline_ready),
        "radar_unusual_change": unusual,
        "radar_signal_status": radar_status,
        "weekly_satellite_status": weekly_status,
        "radar_vv_change_threshold_db": S1_VV_CHANGE_THRESHOLD_DB,
        "radar_vh_change_threshold_db": S1_VH_CHANGE_THRESHOLD_DB,
        "radar_signal_note": (
            "Sentinel-1 change compares the newest VV/VH values with a "
            "same-orbit, same-pass historical median. It is a change screen, "
            "not an NDVI estimate or diagnosis."
        ),
    }


def recent_slope(values):
    """Estimate the linear slope of the last four valid observations."""
    series = pd.Series(values, dtype="float64").dropna()
    if len(series) < 3:
        return np.nan
    y = series.tail(4).to_numpy()
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def temporal_metrics(history):
    """Compare recent and previous NDVI windows and calculate the recent trend.

    Only windows meeting S2_MIN_VALID_COVERAGE are used -- the same bar
    choose_current_s2() applies to the headline condition score. Without
    this, a window with a tiny cloud-free sliver of the farm could be
    averaged into the change figure with no less weight than a properly
    clear window, even though it wouldn't have qualified as a usable
    "current" reading on its own. This keeps the score and its change
    figure answerable to one evidentiary standard instead of two.
    """
    history = history.copy()
    history["window_end"] = pd.to_datetime(history["window_end"])
    history = history.sort_values("window_end")
    if "s2_valid_coverage" in history.columns:
        coverage = pd.to_numeric(history["s2_valid_coverage"], errors="coerce")
        history = history[coverage >= S2_MIN_VALID_COVERAGE]
    valid = history.dropna(subset=["ndvi"])

    recent = valid.tail(3)
    previous = valid.iloc[-6:-3] if len(valid) >= 6 else valid.iloc[0:0]
    recent_mean = recent["ndvi"].mean() if len(recent) else np.nan
    previous_mean = previous["ndvi"].mean() if len(previous) else np.nan
    change = (
        recent_mean - previous_mean
        if pd.notna(recent_mean) and pd.notna(previous_mean)
        else np.nan
    )
    return {
        "ndvi_recent_30d_mean": recent_mean,
        "ndvi_previous_30d_mean": previous_mean,
        "ndvi_30d_change": change,
        "ndvi_recent_slope": recent_slope(valid["ndvi"]),
    }



def index_temporal_metrics(history, column, prefix):
    """Calculate recent/previous 30-day change and short slope for an index.

    Applies the same S2_MIN_VALID_COVERAGE filter as temporal_metrics()
    above, for the same reason: this pipeline should not let a lightly-clear
    partial window quietly stand in for a properly clear one just because it
    landed in the averaging window rather than the headline score.
    """
    if history.empty or column not in history.columns:
        return {
            f"{prefix}_recent_30d_mean": np.nan,
            f"{prefix}_previous_30d_mean": np.nan,
            f"{prefix}_30d_change": np.nan,
            f"{prefix}_recent_slope": np.nan,
        }

    df = history.copy()
    df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")
    df[column] = pd.to_numeric(df[column], errors="coerce")
    if "s2_valid_coverage" in df.columns:
        coverage = pd.to_numeric(df["s2_valid_coverage"], errors="coerce")
        df = df[coverage >= S2_MIN_VALID_COVERAGE]
    valid = df.sort_values("window_end").dropna(subset=[column])
    recent = valid.tail(3)
    previous = valid.iloc[-6:-3] if len(valid) >= 6 else valid.iloc[0:0]
    recent_mean = recent[column].mean() if len(recent) else np.nan
    previous_mean = previous[column].mean() if len(previous) else np.nan
    change = (
        recent_mean - previous_mean
        if pd.notna(recent_mean) and pd.notna(previous_mean)
        else np.nan
    )
    return {
        f"{prefix}_recent_30d_mean": recent_mean,
        f"{prefix}_previous_30d_mean": previous_mean,
        f"{prefix}_30d_change": change,
        f"{prefix}_recent_slope": recent_slope(valid[column]),
    }


def low_condition_metrics(ndvi_raster, farm_boundary):
    """Measure pixels below the field-relative NDVI condition threshold.

    The actual farm polygon is applied before any statistics are calculated.
    This is essential because Earth Engine GeoTIFF downloads can store pixels
    outside a clipped polygon as numeric zeros without marking them as NoData.
    Those background zeros must not be counted as poor crop condition.

    Returns the field median, threshold, affected share, affected hectares and
    valid mapped hectares. Pixel areas are approximate screening results, not
    diagnoses or cadastral area measurements.
    """
    empty_result = {
        "ndvi_field_median": np.nan,
        "low_condition_threshold": np.nan,
        "low_condition_share": np.nan,
        "low_condition_area_ha": np.nan,
        "valid_area_ha": np.nan,
        "map_low_share": np.nan,
        "map_low_area_ha": np.nan,
    }

    if not ndvi_raster or pd.isna(ndvi_raster):
        return empty_result

    if farm_boundary is None or farm_boundary.empty:
        raise ValueError(
            "A farm boundary is required to calculate low-condition area"
        )

    with rasterio.open(ndvi_raster) as src:
        if src.crs is None:
            raise ValueError(f"NDVI raster has no CRS: {ndvi_raster}")
        if src.crs.is_geographic:
            raise ValueError(
                f"NDVI raster must use a projected CRS for area calculation: "
                f"{ndvi_raster}"
            )

        data = src.read(1).astype("float64")

        # Start with physically possible, finite NDVI values. A legitimate
        # in-field NDVI value of zero remains valid; only the polygon decides
        # whether a pixel belongs to this farm.
        valid = np.isfinite(data) & (data >= -1.0) & (data <= 1.0)
        if src.nodata is not None:
            valid &= data != src.nodata

        boundary = farm_boundary.copy()
        if boundary.crs is None:
            raise ValueError("Farm boundary has no CRS")
        boundary = boundary.to_crs(src.crs)
        geometries = [
            geometry.__geo_interface__
            for geometry in boundary.geometry
            if geometry is not None and not geometry.is_empty
        ]
        if not geometries:
            raise ValueError("Farm boundary contains no usable geometry")

        inside_farm = geometry_mask(
            geometries,
            out_shape=data.shape,
            transform=src.transform,
            invert=True,
        )
        valid &= inside_farm
        values = data[valid]

        if len(values) == 0:
            return empty_result

        median = float(np.nanmedian(values))
        threshold = max(ABSOLUTE_NDVI_FLOOR, median - RELATIVE_NDVI_DROP)
        low_pixels = int(np.sum(values < threshold))
        # Fixed-break count: same NDVI_LOW_BREAK the field map uses for red,
        # so the reported hectares match what the farmer sees on the map.
        map_low_pixels = int(np.sum(values < NDVI_LOW_BREAK))
        valid_pixels = len(values)
        transform = src.transform
        pixel_area_m2 = abs(transform.a * transform.e - transform.b * transform.d)

        return {
            "ndvi_field_median": median,
            "low_condition_threshold": threshold,
            "low_condition_share": low_pixels / valid_pixels,
            "low_condition_area_ha": low_pixels * pixel_area_m2 / 10_000,
            "valid_area_ha": valid_pixels * pixel_area_m2 / 10_000,
            "map_low_share": map_low_pixels / valid_pixels,
            "map_low_area_ha": map_low_pixels * pixel_area_m2 / 10_000,
        }


def ecmwf_forecast_metrics(ecmwf_forecast, minimum_rain_mm=1.0):
    """Summarize the ECMWF daily rainfall forecast, alongside GFS's own.

    Kept separate from forecast_metrics() (the GFS summary) rather than
    merged into it, since GFS's function handles 6-hour-period partial-day
    exclusion that doesn't apply to ECMWF's already-daily series - forcing
    them into one function risked breaking the existing GFS logic for a
    difference that doesn't actually apply here.
    """
    empty_result = {
        "ecmwf_forecast_rain_7d_mm": np.nan,
        "ecmwf_forecast_wettest_day": None,
        "ecmwf_forecast_wettest_day_mm": np.nan,
        "ecmwf_likely_rain_days": None,
    }
    if ecmwf_forecast.empty:
        return empty_result

    df = ecmwf_forecast.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df["ecmwf_forecast_rain_mm"] = pd.to_numeric(
        df["ecmwf_forecast_rain_mm"], errors="coerce"
    )
    df = df.dropna(subset=["forecast_date"]).sort_values("forecast_date").head(7)
    if df.empty or not df["ecmwf_forecast_rain_mm"].notna().any():
        return empty_result

    valid = df.dropna(subset=["ecmwf_forecast_rain_mm"])
    wet_idx = valid["ecmwf_forecast_rain_mm"].idxmax()

    # Same threshold as GFS's likely_rain_days, for a fair side-by-side
    # comparison - not a separately-tuned number. No partial-day exclusion
    # needed here: every row is already a complete 24-hour forecast day,
    # unlike GFS's 6-hour-period accumulation.
    likely_days = valid[valid["ecmwf_forecast_rain_mm"] >= float(minimum_rain_mm)]
    if likely_days.empty:
        ecmwf_likely_rain_days = "None in the available forecast days"
    else:
        ecmwf_likely_rain_days = ", ".join(
            date.strftime("%d %b") for date in likely_days["forecast_date"]
        )

    return {
        "ecmwf_forecast_rain_7d_mm": valid["ecmwf_forecast_rain_mm"].sum(),
        "ecmwf_forecast_wettest_day": valid.loc[wet_idx, "forecast_date"].strftime("%d %b"),
        "ecmwf_forecast_wettest_day_mm": valid.loc[wet_idx, "ecmwf_forecast_rain_mm"],
        "ecmwf_likely_rain_days": ecmwf_likely_rain_days,
    }


def forecast_metrics(forecast, minimum_rain_mm=1.0):
    """
    Summarize provider-neutral daily rainfall without creating false zeros.

    If every daily value is missing, CropIntel returns N/A fields rather than
    reporting 0 mm rainfall.
    """
    empty_result = {
        "forecast_source": None,
        "forecast_issue_time": None,
        "gfs_creation_time": None,
        "forecast_rain_3d_mm": np.nan,
        "forecast_rain_7d_mm": np.nan,
        "rain_forecast_3d_mm": np.nan,
        "rain_forecast_7d_mm": np.nan,
        "forecast_wettest_day": None,
        "forecast_wettest_day_mm": np.nan,
        "forecast_likely_rain_days": None,
        "forecast_partial_days_excluded": False,
    }
    if forecast.empty:
        return empty_result

    df = forecast.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
    df["forecast_rain_mm"] = pd.to_numeric(
        df["forecast_rain_mm"],
        errors="coerce",
    )
    if "contributing_6h_periods" in df.columns:
        df["contributing_6h_periods"] = pd.to_numeric(
            df["contributing_6h_periods"],
            errors="coerce",
        )
    df = df.dropna(subset=["forecast_date"]).sort_values("forecast_date").head(7)

    if df.empty or not df["forecast_rain_mm"].notna().any():
        return empty_result

    valid = df.dropna(subset=["forecast_rain_mm"])
    wet_idx = valid["forecast_rain_mm"].idxmax()

    # Keep the exact forecast issue and client-facing rain-day list in the
    # processed monitoring row. Dashboards can then render a self-contained
    # snapshot instead of reopening a raw CSV that may have been overwritten.
    forecast_issue_time = None
    for issue_column in ("forecast_issue_time", "gfs_creation_time"):
        if issue_column in df.columns and df[issue_column].notna().any():
            forecast_issue_time = df[issue_column].dropna().iloc[0]
            break

    likely_rain_days = None
    partial_days_excluded = False
    if "contributing_6h_periods" in df.columns:
        coverage = df["contributing_6h_periods"]
        partial_days_excluded = bool(
            coverage.isna().any() or coverage.lt(4).any()
        )
        full_days = df[coverage.ge(4)]
        likely_days = full_days[
            full_days["forecast_rain_mm"] >= float(minimum_rain_mm)
        ]

        if likely_days.empty:
            likely_rain_days = "None in the available full forecast days"
        else:
            likely_rain_days = ", ".join(
                date.strftime("%d %b")
                for date in likely_days["forecast_date"]
            )

        # Previously this appended " (edge days excluded; partial data)" to
        # likely_rain_days itself, which put raw pipeline/debug language in
        # front of the client. The same fact is already captured cleanly in
        # forecast_partial_days_excluded (below) for any internal or future
        # dashboard use -- the display string no longer needs to carry it.

    rain_3d = df.head(3)["forecast_rain_mm"].sum(min_count=1)
    rain_7d = df["forecast_rain_mm"].sum(min_count=1)
    return {
        "forecast_source": (
            df["forecast_source"].dropna().iloc[0]
            if "forecast_source" in df.columns
            and df["forecast_source"].notna().any()
            else None
        ),
        "forecast_issue_time": forecast_issue_time,
        "gfs_creation_time": forecast_issue_time,
        "forecast_rain_3d_mm": rain_3d,
        "forecast_rain_7d_mm": rain_7d,
        # Compatibility aliases for existing dashboards.
        "rain_forecast_3d_mm": rain_3d,
        "rain_forecast_7d_mm": rain_7d,
        "forecast_wettest_day": valid.loc[wet_idx, "forecast_date"].date().isoformat(),
        "forecast_wettest_day_mm": valid.loc[wet_idx, "forecast_rain_mm"],
        "forecast_likely_rain_days": likely_rain_days,
        "forecast_partial_days_excluded": partial_days_excluded,
    }


def status_for(ndvi_change, slope, low_share):
    """Classify crop condition as field check, monitor or normal.

    A field check is triggered when either the NDVI decline reaches the severe
    threshold or the low-condition share reaches its severe threshold. The
    monitor class uses the less severe change, trend and area thresholds. This
    is screening logic rather than a diagnosis of stress, pests or deficiency.
    """
    ndvi_change, slope, low_share = map(safe_float, [ndvi_change, slope, low_share])
    if (
        (pd.notna(ndvi_change) and ndvi_change <= NDVI_FIELD_CHECK_DROP)
        or (pd.notna(low_share) and low_share >= LOW_AREA_FIELD_CHECK_SHARE)
    ):
        return "FIELD CHECK RECOMMENDED"
    if (
        (pd.notna(ndvi_change) and ndvi_change <= NDVI_MONITOR_DROP)
        or (pd.notna(slope) and slope <= NDVI_SLOPE_MONITOR)
        or (pd.notna(low_share) and low_share >= LOW_AREA_MONITOR_SHARE)
    ):
        return "MONITOR"
    return "NORMAL"


# -----------------------------------------------------------------------------
# Crop-water functions
# -----------------------------------------------------------------------------

def _normalise_crop_name(value) -> str:
    """Standardize crop names so aliases can be matched consistently."""
    if value is None or pd.isna(value):
        return ""
    value = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return " ".join(value.split())


def crop_profile(crop_type):
    """Return an indicative CropIntel Kc profile or None when crop is unknown."""
    crop = _normalise_crop_name(crop_type)
    if not crop:
        return None
    for key, profile in CROP_WATER_PROFILES.items():
        aliases = [_normalise_crop_name(a) for a in profile.get("aliases", [])]
        if crop == key or crop in aliases:
            return profile
    return None


def crop_stage_and_kc(crop_type, planting_date, target_date):
    """Determine simplified crop stage and stage Kc for one date."""
    profile = crop_profile(crop_type)
    if profile is None:
        return "UNKNOWN CROP", np.nan, np.nan

    planting = pd.to_datetime(planting_date, errors="coerce")
    target = pd.to_datetime(target_date, errors="coerce")
    if pd.isna(planting) or pd.isna(target):
        return "PLANTING DATE NEEDED", np.nan, np.nan

    days = int((target.normalize() - planting.normalize()).days)
    if days < 0:
        return "PRE-PLANT", np.nan, days

    stages = ["INITIAL", "DEVELOPMENT", "MID-SEASON", "LATE-SEASON"]
    cumulative = 0
    for stage, length, kc in zip(
        stages, profile["stage_lengths_days"], profile["kc"]
    ):
        cumulative += int(length)
        if days < cumulative:
            return stage, float(kc), days

    return "POST-SEASON", np.nan, days


def _sum_last_valid(df, column, n):
    """Sum the most recent ``n`` valid values from a DataFrame column."""
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna().tail(n)
    return values.sum() if len(values) else np.nan


def _mean_last_valid(df, column, n):
    """Average the most recent ``n`` valid values from a DataFrame column."""
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna().tail(n)
    return values.mean() if len(values) else np.nan


def crop_water_requirement_from_windows(climate_history, crop_type, planting_date, n_windows):
    """
    Estimate crop water requirement by applying stage Kc to each 10-day ET0 window.

    This is a simplified Kc × ET0 implementation. If crop or planting date is
    missing, N/A is returned instead of applying a generic coefficient.
    """
    if climate_history.empty:
        return np.nan

    df = climate_history.copy()
    df["window_start"] = pd.to_datetime(df["window_start"], errors="coerce")
    df["window_end"] = pd.to_datetime(df["window_end"], errors="coerce")
    df["et0_mm"] = pd.to_numeric(df["et0_mm"], errors="coerce")
    df = df.sort_values("window_end").dropna(subset=["et0_mm"]).tail(n_windows)
    if df.empty:
        return np.nan

    total = 0.0
    used = 0
    for _, row in df.iterrows():
        midpoint = row["window_start"] + (row["window_end"] - row["window_start"]) / 2
        _, kc, _ = crop_stage_and_kc(crop_type, planting_date, midpoint)
        if pd.isna(kc):
            continue
        total += float(row["et0_mm"]) * float(kc)
        used += 1

    return total if used else np.nan


def heat_exposure_risk(days):
    """Classify 30-day heat exposure from the configured day thresholds."""
    days = safe_float(days)
    if pd.isna(days):
        return "NO DATA"
    if days >= HEAT_EXPOSURE_HIGH_DAYS_30D:
        return "HIGH"
    if days >= HEAT_EXPOSURE_MONITOR_DAYS_30D:
        return "MONITOR"
    return "LOW"


def water_stress_risk(ndmi_change, rainfall_ratio_30d, rain_forecast_7d):
    """Classify water concern with Sentinel-2 NDMI as the primary evidence.

    CHIRPS rainfall, forecast rainfall and ERA5-derived ET0 are supporting
    context. No source is interpreted as direct field soil-water measurement.
    """
    ndmi_change = safe_float(ndmi_change)
    ratio = safe_float(rainfall_ratio_30d)
    forecast = safe_float(rain_forecast_7d)

    if pd.isna(ndmi_change):
        return "INSUFFICIENT SENTINEL WATER-STATUS DATA"

    dry_context = pd.notna(ratio) and ratio < WATER_SUPPLY_RATIO_MONITOR
    very_dry_context = pd.notna(ratio) and ratio < WATER_SUPPLY_RATIO_HIGH_RISK
    low_forecast = pd.notna(forecast) and forecast < LOW_FORECAST_RAIN_7D_MM

    if ndmi_change <= NDMI_FIELD_CHECK_DROP and (very_dry_context or low_forecast):
        return "HIGH"
    if ndmi_change <= NDMI_MONITOR_DROP or dry_context:
        return "MODERATE"
    return "LOW"


def field_water_attention(water_risk):
    """Return a monitoring action without prescribing irrigation."""
    if water_risk == "HIGH":
        return "FIELD WATER CHECK REQUIRED"
    if water_risk == "MODERATE":
        return "MONITOR WATER STATUS"
    if water_risk == "LOW":
        return "NO IMMEDIATE WATER-STRESS SIGNAL"
    return "SENTINEL DATA NEEDED"


def water_climate_metrics(
    farm,
    climate_current,
    climate_history,
    observed_history,
    future,
    ndmi,
):
    """Combine Sentinel water evidence with rainfall and climate context.

    NDMI change is the primary monitoring signal. CHIRPS, forecast rainfall and
    ERA5-derived temperature/ET0 provide supporting context. ERA5 soil-water
    fields are not carried into the processed monitoring table.
    """
    crop_type = farm.get("crop_type", "")
    planting_date = farm.get("planting_date", "")

    current_date = climate_current.get("era5_latest_date")
    stage, current_kc, days_after_planting = crop_stage_and_kc(
        crop_type, planting_date, current_date
    )

    et0_30 = safe_float(climate_current.get("climate_30d_et0_mm"))
    et0_90 = safe_float(climate_current.get("climate_90d_et0_mm"))
    chirps_rain_30 = safe_float(farm.get("rain_previous_30d_mm"))
    chirps_rain_90 = _sum_last_valid(
        observed_history.sort_values("window_end"), "rain_mm", 9
    )

    etc_30 = crop_water_requirement_from_windows(
        climate_history, crop_type, planting_date, 3
    )
    etc_90 = crop_water_requirement_from_windows(
        climate_history, crop_type, planting_date, 9
    )

    climate_balance_30 = (
        chirps_rain_30 - et0_30
        if pd.notna(chirps_rain_30) and pd.notna(et0_30)
        else np.nan
    )
    climate_balance_90 = (
        chirps_rain_90 - et0_90
        if pd.notna(chirps_rain_90) and pd.notna(et0_90)
        else np.nan
    )
    crop_balance_30 = (
        chirps_rain_30 - etc_30
        if pd.notna(chirps_rain_30) and pd.notna(etc_30)
        else np.nan
    )
    crop_balance_90 = (
        chirps_rain_90 - etc_90
        if pd.notna(chirps_rain_90) and pd.notna(etc_90)
        else np.nan
    )

    climate_ratio_30 = (
        chirps_rain_30 / et0_30
        if pd.notna(chirps_rain_30) and pd.notna(et0_30) and et0_30 > 0
        else np.nan
    )
    climate_ratio_90 = (
        chirps_rain_90 / et0_90
        if pd.notna(chirps_rain_90) and pd.notna(et0_90) and et0_90 > 0
        else np.nan
    )
    crop_ratio_30 = (
        chirps_rain_30 / etc_30
        if pd.notna(chirps_rain_30) and pd.notna(etc_30) and etc_30 > 0
        else np.nan
    )

    deficit_30 = max(0.0, -crop_balance_30) if pd.notna(crop_balance_30) else np.nan
    heat_days = safe_float(climate_current.get("climate_30d_heat_exposure_days"))
    ndmi_change = safe_float(ndmi.get("ndmi_30d_change"))
    risk = water_stress_risk(
        ndmi_change, climate_ratio_30, future.get("rain_forecast_7d_mm")
    )

    return {
        "era5_latest_date": current_date,
        "temperature_mean_30d_c": safe_float(
            climate_current.get("climate_30d_temperature_mean_c")
        ),
        "temperature_max_30d_c": safe_float(
            climate_current.get("climate_30d_temperature_max_c")
        ),
        "heat_exposure_days_30d": heat_days,
        "heat_exposure_risk": heat_exposure_risk(heat_days),
        "reference_et0_30d_mm": et0_30,
        "reference_et0_90d_mm": et0_90,
        "era5_model_et_30d_mm": safe_float(
            climate_current.get("climate_30d_era5_model_et_mm")
        ),
        "climate_background_source": "ERA5-Land",
        "field_water_primary_source": "Sentinel-2 NDMI",
        **ndmi,
        "crop_stage": stage,
        "days_after_planting": days_after_planting,
        "crop_coefficient_kc": current_kc,
        "crop_water_requirement_30d_mm": etc_30,
        "crop_water_requirement_90d_mm": etc_90,
        "chirps_rain_90d_mm": chirps_rain_90,
        "climatic_water_balance_30d_mm": climate_balance_30,
        "climatic_water_balance_90d_mm": climate_balance_90,
        "crop_water_balance_30d_mm": crop_balance_30,
        "crop_water_balance_90d_mm": crop_balance_90,
        "climate_water_supply_ratio_30d": climate_ratio_30,
        "climate_water_supply_ratio_90d": climate_ratio_90,
        "crop_water_supply_ratio_30d": crop_ratio_30,
        "estimated_unmet_crop_water_demand_30d_mm": deficit_30,
        "water_stress_risk": risk,
        # Compatibility alias for existing dashboards; this is no longer an
        # ERA5 root-zone drought classification.
        "drought_screening": risk,
        "irrigation_attention": field_water_attention(risk),
        "irrigation_note": (
            "Monitoring only. Confirm conditions in the field before changing "
            "irrigation; CropIntel does not prescribe irrigation depth."
        ),
    }



def estimate_next_7d_crop_water_need(water_metrics):
    """
    Estimate near-term ETc from recent ET0 climatology and the current Kc.

    The prototype does not forecast full Penman-Monteith ET0. It uses
    the recent 30-day ERA5-Land ET0 daily rate as a screening estimate for the
    next seven days. This must be labelled as an estimate, not an exact demand.
    """
    et0_30 = safe_float(water_metrics.get("reference_et0_30d_mm"))
    kc = safe_float(water_metrics.get("crop_coefficient_kc"))
    if pd.isna(et0_30) or pd.isna(kc):
        return np.nan, np.nan
    et0_7 = et0_30 / 30.0 * 7.0
    return et0_7, et0_7 * kc


def irrigation_screening(water_metrics, future, management):
    """Create a conservative field-water monitoring summary."""
    et0_7, etc_7 = estimate_next_7d_crop_water_need(water_metrics)
    forecast_7 = safe_float(future.get("rain_forecast_7d_mm"))
    deficit = (
        max(0.0, etc_7 - forecast_7)
        if pd.notna(etc_7) and pd.notna(forecast_7)
        else np.nan
    )
    risk = str(water_metrics.get("water_stress_risk", "NO DATA"))

    if pd.isna(etc_7):
        status = "CROP/STAGE DATA NEEDED"
        recommendation = "Add crop type and planting date to improve water-context monitoring."
    elif pd.isna(forecast_7):
        status = "RAINFALL FORECAST DATA NEEDED"
        recommendation = (
            "The 7-day rainfall forecast is unavailable, so CropIntel will not "
            "estimate a near-term rainfall-demand gap."
        )
    elif risk == "HIGH":
        status = "FIELD WATER CHECK REQUIRED"
        recommendation = (
            "Inspect the highlighted area and check crop and soil conditions. "
            "Use field findings and farm records before changing irrigation."
        )
    elif risk == "MODERATE":
        status = "MONITOR WATER STATUS"
        recommendation = (
            "Monitor crop water patterns and review recent and expected rain."
        )
    else:
        status = "NO IMMEDIATE IRRIGATION SIGNAL"
        recommendation = "Continue routine monitoring."

    return {
        "reference_et0_next_7d_est_mm": et0_7,
        "crop_water_requirement_next_7d_est_mm": etc_7,
        "forecast_rain_next_7d_mm": forecast_7,
        "estimated_water_deficit_next_7d_mm": deficit,
        "irrigation_screening_status": status,
        "irrigation_recommendation": recommendation,
        "irrigation_record_status": management.get("irrigation_record_status", "NO RECORD"),
        "last_irrigation_date": management.get("last_irrigation_date"),
        "last_irrigation_mm": management.get("last_irrigation_mm", np.nan),
        "irrigation_screening_note": (
            "CropIntel provides monitoring and field prioritisation only. It does "
            "not prescribe irrigation timing or depth."
        ),
    }


def nutrition_screening(farm, temporal, ndre, spatial, water, management):
    """
    Screen for potential nutrient stress without diagnosing a nutrient deficiency.

    Sentinel red-edge/vegetation signals are non-specific. Water stress, disease,
    rooting problems, management, and other factors can create similar patterns.
    Recorded fertilizer history is treated as contextual evidence only.
    """
    ndre_change = safe_float(ndre.get("ndre_30d_change"))
    low_share = safe_float(spatial.get("low_condition_share"))
    ndvi_change = safe_float(temporal.get("ndvi_30d_change"))
    crop_balance = safe_float(water.get("crop_water_balance_30d_mm"))
    drought = str(water.get("drought_screening", "NO DATA"))

    score = 0
    evidence = []
    if pd.notna(ndre_change):
        if ndre_change <= NDRE_ASSESS_DROP:
            score += 2
            evidence.append("the crop nutrition condition declined substantially")
        elif ndre_change <= NDRE_MONITOR_DROP:
            score += 1
            evidence.append("the crop nutrition condition is declining")
    if pd.notna(low_share):
        if low_share >= NUTRIENT_LOW_AREA_ASSESS_SHARE:
            score += 2
            evidence.append("a large relative low-condition area is present")
        elif low_share >= NUTRIENT_LOW_AREA_MONITOR_SHARE:
            score += 1
            evidence.append("a relative low-condition area is present")
    if pd.notna(ndvi_change) and ndvi_change < 0:
        score += 1
        evidence.append("the overall crop condition is below the previous period")

    water_confounded = (
        (pd.notna(crop_balance) and crop_balance < -20)
        or drought.startswith("HIGH")
    )

    if score >= 4:
        risk = "HIGH"
    elif score >= 2:
        risk = "MODERATE"
    elif score == 1:
        risk = "MONITOR"
    else:
        risk = "LOW"

    fertilizer_status = management.get("fertilizer_record_status", "NO RECORD")
    recorded_n = safe_float(management.get("recorded_n_applied_kg_ha"))
    stage = str(water.get("crop_stage", "UNKNOWN"))

    if risk in {"HIGH", "MODERATE"}:
        recommendation = (
            "Inspect the highlighted area and compare crop colour and growth. "
            "Check fertilizer records, leaves and soil before changing fertilizer use."
        )
    elif risk == "MONITOR":
        recommendation = "Monitor the zone and review management records before the next application."
    else:
        recommendation = "No strong crop nutrition concern was detected in this update."

    if water_confounded:
        confidence = "LOW - WATER CONDITIONS MAY ALSO EXPLAIN THE PATTERN"
    elif risk in {"HIGH", "MODERATE"} and fertilizer_status == "AVAILABLE":
        confidence = "MEDIUM - FARM RECORDS ARE AVAILABLE"
    else:
        confidence = "LOW - FIELD CONFIRMATION IS NEEDED"

    if fertilizer_status == "AVAILABLE" and pd.notna(recorded_n):
        nitrogen_context = (
            f"Recorded seasonal nitrogen application: {recorded_n:.1f} kg/ha. "
            "This record does not by itself confirm a shortage."
        )
    else:
        nitrogen_context = "No recorded nitrogen fertilizer application is available."

    return {
        **ndre,
        "nutrient_stress_screening": risk,
        "nutrient_screening_confidence": confidence,
        "nutrition_recommendation": recommendation,
        "nutrition_screening_basis": "; ".join(evidence) if evidence else "No strong crop difference detected",
        "nitrogen_context": nitrogen_context,
        "nitrogen_assessment_status": (
            "REVIEW N STATUS" if risk in {"HIGH", "MODERATE"} and stage in {"DEVELOPMENT", "MID-SEASON"}
            else "NO SPECIFIC N CLAIM"
        ),
        "recorded_n_applied_kg_ha": management.get("recorded_n_applied_kg_ha", np.nan),
        "recorded_p2o5_applied_kg_ha": management.get("recorded_p2o5_applied_kg_ha", np.nan),
        "recorded_k2o_applied_kg_ha": management.get("recorded_k2o_applied_kg_ha", np.nan),
        "fertilizer_record_status": fertilizer_status,
        "last_fertilizer_date": management.get("last_fertilizer_date"),
        "last_fertilizer_product": management.get("last_fertilizer_product"),
        "nutrition_note": (
            "Crop imagery cannot confirm a nutrient shortage on its own. Use farm "
            "records and field, leaf or soil checks to confirm the cause."
        ),
    }


def main():
    """Build the final monitoring table for every extracted farm.

    Processing order: load raw tables, isolate each farm, calculate crop,
    spatial, forecast, climate and management metrics, assign screening
    statuses, and save ``cropintel_monitoring.csv``.
    """
    # Step 1: Load outputs from the three extraction modules.
    current = pd.read_csv(RAW_DIR / "current_metrics.csv")
    history = pd.read_csv(RAW_DIR / "timeseries.csv")
    forecast = pd.read_csv(RAW_DIR / "rain_forecast_daily.csv")
    ecmwf_forecast_path = RAW_DIR / "rain_forecast_daily_ecmwf.csv"
    ecmwf_forecast = (
        pd.read_csv(ecmwf_forecast_path)
        if ecmwf_forecast_path.exists()
        else pd.DataFrame(columns=["farm_id", "forecast_date", "ecmwf_forecast_rain_mm"])
    )
    climate_current = pd.read_csv(RAW_DIR / "climate_current.csv")
    climate_history = pd.read_csv(RAW_DIR / "climate_timeseries.csv")
    management_records = load_management_records()
    farm_boundaries = load_farms()

    # Step 2: Standardize identifiers before joining farm-specific records.
    for df in (current, history, forecast, climate_current, climate_history):
        df["farm_id"] = df["farm_id"].astype(str)
    if not ecmwf_forecast.empty:
        ecmwf_forecast["farm_id"] = ecmwf_forecast["farm_id"].astype(str)
    farm_boundaries["farm_id"] = farm_boundaries["farm_id"].astype(str)

    # Step 3: Calculate every indicator separately for each farm.
    results = []
    for _, farm in current.iterrows():
        farm_id = str(farm["farm_id"])
        farm_history = history[history["farm_id"] == farm_id].copy()
        farm_climate_history = climate_history[
            climate_history["farm_id"] == farm_id
        ].copy()
        farm_climate_current = climate_current[
            climate_current["farm_id"] == farm_id
        ]

        climate_row = (
            farm_climate_current.iloc[0].to_dict()
            if not farm_climate_current.empty
            else {}
        )

        # Crop, spatial, weather, management and decision-support indicators.
        temporal = temporal_metrics(farm_history)
        ndmi = index_temporal_metrics(farm_history, "ndmi", "ndmi")
        ndre = index_temporal_metrics(farm_history, "ndre", "ndre")
        # Radar vegetation index trend. Computed with the same window logic
        # as the optical indices so it is directly comparable during review.
        # This is recorded for calibration and agronomic inspection only:
        # no threshold is applied and no client-facing label reads it,
        # because RVI has not been validated against field observations for
        # this crop or region (see prepare_sentinel1_rvi in extract_data.py).
        rvi = index_temporal_metrics(farm_history, "rvi", "rvi")
        farm_boundary = farm_boundaries[
            farm_boundaries["farm_id"] == farm_id
        ]
        spatial = low_condition_metrics(
            farm.get("ndvi_raster", ""),
            farm_boundary,
        )
        future = forecast_metrics(forecast[forecast["farm_id"] == farm_id].copy())
        future_ecmwf = ecmwf_forecast_metrics(
            ecmwf_forecast[ecmwf_forecast["farm_id"] == farm_id].copy()
            if not ecmwf_forecast.empty else ecmwf_forecast
        )
        management = summarize_management_records(management_records, farm_id)
        water = water_climate_metrics(
            farm,
            climate_row,
            farm_climate_history,
            farm_history,
            future,
            ndmi,
        )
        irrigation = irrigation_screening(water, future, management)
        nutrition = nutrition_screening(
            farm, temporal, ndre, spatial, water, management
        )
        status = status_for(
            temporal["ndvi_30d_change"],
            temporal["ndvi_recent_slope"],
            spatial["low_condition_share"],
        )
        freshness = satellite_freshness_metrics(farm)

        # Combine all outputs, then add a separate client-language layer. Raw
        # measurements remain available for audit and agronomic review, while
        # dashboards consume the clear monitoring labels.
        combined = {
            **farm.to_dict(),
            **temporal,
            **spatial,
            **future,
            **future_ecmwf,
            **water,
            **management,
            **nutrition,
            **irrigation,
            **freshness,
            **rvi,
            "status": status,
            "total_area_ha": (
                farm_boundary["total_area_ha"].iloc[0]
                if not farm_boundary.empty else np.nan
            ),
        }
        combined.update(build_decision_fields(combined))
        results.append(combined)

    # Step 4: Write the single processed table consumed by all dashboards.
    # This snapshot write is unchanged from before - it always reflects only
    # the most recent run, and every existing consumer of this file keeps
    # working exactly as it did.
    results_df = pd.DataFrame(results)
    results_df.to_csv(PROCESSED_DIR / "cropintel_monitoring.csv", index=False)

    # Step 5: Append this run's results to a separate, persistent history
    # file. Unlike the snapshot above, this file is never overwritten -
    # each weekly run adds to it, so past readings are preserved for the
    # time-series charts. Deduplicated on (farm_id, run_datetime): if this
    # exact run is re-executed (e.g. after a DAG retry), the retry's row
    # replaces the earlier attempt for that same run rather than creating a
    # duplicate entry for the same week.
    _append_to_history(results_df)

    # Step 6: Append this run's found S2/S1 acquisitions to acquisition-keyed
    # history, separate from the run-keyed history above. Deduplicated on the
    # real acquisition date (not the run date): if the same underlying
    # satellite pass is still the "current" one several weeks in a row (a
    # long cloud gap, for example), it collapses to a single entry instead of
    # being re-recorded every week. A genuinely new acquisition found in a
    # later run adds a new point. This is what the NDVI/radar time-series
    # charts should read from - it reflects real satellite passes, not the
    # pipeline's run schedule.
    for combined in results:
        _append_s2_acquisition(combined)
        _append_s1_acquisition(combined)

    # Bootstrap/refresh radar acquisition history from the real acquisition
    # metadata saved in raw timeseries.csv. This means the Field Trends page
    # does not have to wait for several future weekly DAG runs before it can
    # show a multi-observation radar continuity series.
    _merge_s2_history_from_raw(history)
    _merge_s1_history_from_raw(history)

    # Step 7 (optional): predict NDVI for real S1 dates that have no matching
    # S2 reading close enough in time, using gap_fill_ndvi.py. Off by
    # default (NDVI_GAP_FILL_ENABLED, config.py). Imported here rather than
    # at module level so this file has no hard dependency on scikit-learn
    # when the flag is off. Writes only to s2_ndvi_gap_filled.csv -- never
    # to s2_ndvi_history.csv above, and never into anything decision_engine
    # reads, and it self-validates against real held-out readings every run,
    # refusing to write output if it doesn't beat plain interpolation. See
    # gap_fill_ndvi.py's module docstring before ever enabling this flag.
    if NDVI_GAP_FILL_ENABLED:
        from gap_fill_ndvi import run as run_ndvi_gap_fill
        run_ndvi_gap_fill()


def _append_s2_acquisition(combined: dict) -> None:
    """Record this farm's currently-found S2 acquisition, deduped by date.

    Reuses whatever choose_current_s2 already found this run (current_ndvi,
    latest_s2_date) rather than issuing a new Earth Engine query - this file
    accumulates real, distinct acquisition dates across repeated runs.
    """
    farm_id = combined.get("farm_id")
    acquisition_date = combined.get("latest_s2_date")
    ndvi = combined.get("current_ndvi")
    if not farm_id or acquisition_date is None or pd.isna(ndvi):
        return

    row = pd.DataFrame([{
        "farm_id": farm_id,
        "acquisition_date": acquisition_date,
        "ndvi": ndvi,
        "ndmi": combined.get("current_ndmi"),
        "ndre": combined.get("current_ndre"),
        "s2_valid_coverage": combined.get("s2_latest_usable_coverage"),
    }])
    _append_acquisition_history(
        PROCESSED_DIR / "s2_ndvi_history.csv", row,
        dedup_keys=["farm_id", "acquisition_date"],
    )


def _append_s1_acquisition(combined: dict) -> None:
    """Record this farm's currently-found S1 acquisition, deduped by date.

    Includes relative orbit and pass direction alongside the VV/VH values,
    since a comparison between two S1 readings is only valid when both share
    the same viewing geometry - a consumer of this history must filter on
    those fields before comparing rows, not just sort by date.
    """
    farm_id = combined.get("farm_id")
    acquisition_date = combined.get("latest_s1_date")
    vv = combined.get("current_vv_db")
    vh = combined.get("current_vh_db")
    if not farm_id or acquisition_date is None or pd.isna(vv) or pd.isna(vh):
        return

    row = pd.DataFrame([{
        "farm_id": farm_id,
        "acquisition_date": acquisition_date,
        "vv_db": vv,
        "vh_db": vh,
        "rvi": combined.get("current_rvi"),
        "relative_orbit": combined.get("s1_relative_orbit"),
        "orbit_pass": combined.get("s1_orbit_pass"),
    }])
    _append_acquisition_history(
        PROCESSED_DIR / "s1_radar_history.csv", row,
        dedup_keys=["farm_id", "acquisition_date", "relative_orbit", "orbit_pass"],
    )


def _merge_s2_history_from_raw(history: pd.DataFrame) -> None:
    """Persist real clear S2 acquisitions discovered in historical windows."""
    required = {
        "farm_id", "s2_acquisition_date", "s2_acq_ndvi",
        "s2_acq_ndmi", "s2_acq_ndre", "s2_acq_valid_coverage",
    }
    if history.empty or not required.issubset(history.columns):
        return
    raw = history[list(required)].copy().rename(columns={
        "s2_acquisition_date": "acquisition_date",
        "s2_acq_ndvi": "ndvi",
        "s2_acq_ndmi": "ndmi",
        "s2_acq_ndre": "ndre",
        "s2_acq_valid_coverage": "s2_valid_coverage",
    })
    for col in ("ndvi", "ndmi", "ndre", "s2_valid_coverage"):
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["farm_id", "acquisition_date", "ndvi"])
    if raw.empty:
        return
    _append_acquisition_history(
        PROCESSED_DIR / "s2_ndvi_history.csv", raw,
        dedup_keys=["farm_id", "acquisition_date"],
    )


def _merge_s1_history_from_raw(history: pd.DataFrame) -> None:
    """Persist real S1 acquisitions discovered inside historical windows.

    ``extract_data.time_series_for_farm`` stores the newest real Sentinel-1
    acquisition inside each historical window together with orbit/pass
    metadata. This function promotes those rows into the persistent acquisition
    history used by the dashboard. Rows without an acquisition date or usable
    VV/VH values are ignored.
    """
    required = {
        "farm_id", "s1_acquisition_date", "vv_db", "vh_db",
        "s1_relative_orbit", "s1_orbit_pass",
    }
    if history.empty or not required.issubset(history.columns):
        return

    raw = history[list(required | {"rvi"})].copy() if "rvi" in history.columns else history[list(required)].copy()
    raw = raw.rename(columns={
        "s1_acquisition_date": "acquisition_date",
        "s1_relative_orbit": "relative_orbit",
        "s1_orbit_pass": "orbit_pass",
    })
    raw["vv_db"] = pd.to_numeric(raw["vv_db"], errors="coerce")
    raw["vh_db"] = pd.to_numeric(raw["vh_db"], errors="coerce")
    raw = raw.dropna(subset=["farm_id", "acquisition_date", "vv_db", "vh_db"])
    if raw.empty:
        return
    if "rvi" not in raw.columns:
        raw["rvi"] = np.nan
    raw = raw[[
        "farm_id", "acquisition_date", "vv_db", "vh_db", "rvi",
        "relative_orbit", "orbit_pass",
    ]]
    _append_acquisition_history(
        PROCESSED_DIR / "s1_radar_history.csv", raw,
        dedup_keys=["farm_id", "acquisition_date", "relative_orbit", "orbit_pass"],
    )


def _append_acquisition_history(path, new_row: pd.DataFrame, dedup_keys: list) -> None:
    """Shared append-and-dedup logic for the acquisition-keyed history files."""
    if path.exists():
        existing = pd.read_csv(path)
        combined_df = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined_df = new_row
    combined_df = combined_df.drop_duplicates(subset=dedup_keys, keep="last")
    combined_df = combined_df.sort_values(dedup_keys)
    combined_df.to_csv(path, index=False)


def _append_to_history(results_df: pd.DataFrame) -> None:
    """Append this run's results to the persistent time-series history file.

    Deduplicates on (farm_id, run_datetime) so a re-run for the same week
    replaces that week's row instead of creating a duplicate, while genuinely
    new weekly runs accumulate as new rows. This is additive-only: it never
    touches cropintel_monitoring.csv.
    """
    history_path = PROCESSED_DIR / "cropintel_monitoring_history.csv"
    if history_path.exists():
        existing = pd.read_csv(history_path)
        combined = pd.concat([existing, results_df], ignore_index=True)
    else:
        combined = results_df
    combined = combined.drop_duplicates(
        subset=["farm_id", "run_datetime"], keep="last"
    )
    combined = combined.sort_values(["farm_id", "run_datetime"])
    combined.to_csv(history_path, index=False)


if __name__ == "__main__":
    main()