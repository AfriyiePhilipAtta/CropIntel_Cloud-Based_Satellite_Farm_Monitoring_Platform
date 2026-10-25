"""Validate required CropIntel pipeline outputs and scientific ranges."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR
from farm_loader import load_farms


RAW_FILES = [
    "current_metrics.csv",
    "timeseries.csv",
    "rain_observed_daily.csv",
    "rain_forecast_6h.csv",
    "rain_forecast_daily.csv",
    "climate_current.csv",
    "climate_timeseries.csv",
]


def required(path: Path) -> pd.DataFrame:
    """Read a required non-empty CSV output."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty output: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Output has no rows: {path}")
    return frame


def _check_farms(frame: pd.DataFrame, farm_ids: set[str], name: str) -> None:
    """Require every configured farm ID in an output table."""
    if "farm_id" not in frame.columns:
        raise ValueError(f"{name} has no farm_id column")
    missing = farm_ids - set(frame["farm_id"].astype(str))
    if missing:
        raise ValueError(f"{name} is missing farms: {sorted(missing)}")


def main(stage: str) -> None:
    """Validate raw extraction tables or the processed monitoring table."""
    farm_ids = set(load_farms()["farm_id"].astype(str))

    if stage == "raw":
        frames = {name: required(RAW_DIR / name) for name in RAW_FILES}
        for name, frame in frames.items():
            _check_farms(frame, farm_ids, name)

        current = frames["current_metrics.csv"]
        satellite_freshness_columns = {
            "s2_latest_usable_coverage",
            "latest_s1_date",
            "s1_relative_orbit",
            "s1_orbit_pass",
            "current_vv_db",
            "current_vh_db",
            "s1_baseline_scene_count",
            "baseline_vv_db",
            "baseline_vh_db",
            "vv_change_db",
            "vh_change_db",
        }
        missing_satellite_fields = satellite_freshness_columns - set(current.columns)
        if missing_satellite_fields:
            raise ValueError(
                "current_metrics.csv is missing satellite freshness fields: "
                f"{sorted(missing_satellite_fields)}"
            )

        for metric in [
            "current_ndvi",
            "current_ndmi",
            "current_ndre",
            "current_evi",
        ]:
            if metric in current.columns:
                values = pd.to_numeric(current[metric], errors="coerce").dropna()
                if not values.between(-1, 1).all():
                    raise ValueError(f"{metric} contains values outside -1 to 1")

        run_dates = pd.to_datetime(current["run_datetime"], errors="coerce", utc=True)
        for field in ("latest_s2_date", "latest_s1_date"):
            acquisition_dates = pd.to_datetime(
                current[field], errors="coerce", utc=True
            )
            future = acquisition_dates.notna() & run_dates.notna() & (
                acquisition_dates.dt.normalize() > run_dates.dt.normalize()
            )
            if future.any():
                farms = current.loc[future, "farm_id"].astype(str).tolist()
                raise ValueError(f"{field} is after run_datetime for farms: {farms}")
    else:
        processed = required(PROCESSED_DIR / "cropintel_monitoring.csv")
        _check_farms(processed, farm_ids, "cropintel_monitoring.csv")

        freshness_columns = {
            "s2_age_days",
            "s1_age_days",
            "s2_is_fresh",
            "s1_is_fresh",
            "radar_baseline_ready",
            "radar_signal_status",
            "weekly_satellite_status",
            "overall_confidence_client",
            "satellite_update_client",
        }
        missing_freshness = freshness_columns - set(processed.columns)
        if missing_freshness:
            raise ValueError(
                "cropintel_monitoring.csv is missing freshness fields: "
                f"{sorted(missing_freshness)}"
            )

        allowed_radar = {
            "NO_DATA",
            "STALE",
            "INSUFFICIENT_BASELINE",
            "NO_UNUSUAL_CHANGE",
            "UNUSUAL_CHANGE",
        }
        invalid_radar = set(processed["radar_signal_status"].dropna()) - allowed_radar
        if invalid_radar:
            raise ValueError(f"Unknown radar_signal_status values: {sorted(invalid_radar)}")

        no_fresh = processed["weekly_satellite_status"] == "NO_FRESH_SATELLITE"
        wrong_confidence = no_fresh & (
            processed["overall_confidence_client"].astype(str) != "Low"
        )
        if wrong_confidence.any():
            farms = processed.loc[wrong_confidence, "farm_id"].astype(str).tolist()
            raise ValueError(
                "Confidence must be Low when neither satellite source is fresh; "
                f"affected farms: {farms}"
            )

    print(f"Validated {stage} outputs for {len(farm_ids)} farm(s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["raw", "processed"])
    main(parser.parse_args().stage)
