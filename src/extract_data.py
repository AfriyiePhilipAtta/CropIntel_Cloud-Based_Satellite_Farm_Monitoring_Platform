"""
CropIntel observed-data extraction.

Every farm is loaded from a polygon file in POLYGON_DIR and processed
independently.

Resolution rule
---------------
Sentinel-2 and Sentinel-1 are reduced and exported through the exact farm
boundary. GPM rainfall and SMAP surface soil moisture are much coarser than most
farm parcels, so CropIntel samples a small local context area around the farm
centroid. Those coarse values are regional/local environmental context, not
field-resolution measurements.

Execution steps
---------------
1. Receive a UTC monitoring date from Airflow or the command line.
2. Connect to Earth Engine and discover all configured farm polygons.
3. Mask Sentinel-2 cloud/cirrus pixels and calculate NDVI, NDMI, NDRE and EVI.
4. Collect Sentinel-1 VV and VH radar observations for complementary context.
5. Select the latest usable optical composite for each farm.
6. Reduce current index and radar values over the exact farm boundary.
7. Sample GPM rainfall and SMAP moisture over documented coarse context areas.
8. Export farm-clipped NDVI, NDMI, NDRE and RGB GeoTIFFs.
9. Repeat the reductions in fixed historical windows to build time series.
10. Save current metrics and time-series CSV files for transformation.
"""

# ============================================================
# STEP 1: IMPORTS AND CONFIGURATION
# Load Python libraries, Earth Engine, project settings, farm-loading tools
# and shared helper functions used throughout the extraction workflow.
# ============================================================
from __future__ import annotations

import argparse
from datetime import timedelta

import ee
import numpy as np
import pandas as pd

from config import (
    GEE_PROJECT_ID,
    GPM_CONTEXT_RADIUS_METERS,
    GPM_SCALE_METERS,
    RASTER_NODATA_VALUE,
    RAINFALL_TREND_DAYS,
    RAW_DIR,
    S2_SOURCE,
    S1_BASELINE_DAYS,
    S1_BASELINE_GAP_DAYS,
    S1_GAP_REFERENCE_MAX_DAYS,
    S1_SEARCH_DAYS,
    S1_VH_CHANGE_THRESHOLD_DB,
    S1_VV_CHANGE_THRESHOLD_DB,
    S2_FALLBACK_WINDOWS_DAYS,
    S2_MIN_VALID_COVERAGE,
    SMAP_CONTEXT_RADIUS_METERS,
    SMAP_SCALE_METERS,
    TIME_SERIES_DAYS,
    TIME_SERIES_WINDOW_DAYS,
)
from clear_view_state import (
    load_s2_clear_view_state,
    resolve_persisted_clear_view,
    save_s2_clear_view_state,
)
from farm_loader import load_farms
from history_cache import WindowCache, geometry_key, iter_windows
from gee_helpers import (
    coarse_context_geometry,
    download_geotiff,
    dt_string,
    ee_geometry_from_shapely,
    initialise_earth_engine,
    millis_to_datetime,
    parse_utc_datetime,
    utm_crs_for_geometry,
)

# ============================================================
# STEP 2: DEFINE EARTH ENGINE DATA SOURCES
# These collection IDs identify the optical, radar, rainfall and soil-moisture
# datasets queried later. The state file stores each farm's best clear S2 view.
# ============================================================
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"
GPM_COLLECTION = "NASA/GPM_L3/IMERG_V07"
SMAP_COLLECTION = "NASA/SMAP/SPL3SMP_E/006"

# The clear-view state file and its forward-only rule now live in
# clear_view_state.py, shared with extract_s2_regional.py.

# Columns extract_s2_regional.py supplies when S2_SOURCE is "stac".
S2_CURRENT_FIELDS = [
    "s2_scene_count", "s2_lookback_days", "s2_valid_coverage",
    "s2_latest_usable_coverage", "latest_s2_date", "current_ndvi",
    "current_ndmi", "current_ndre", "current_evi", "ndvi_raster",
    "ndre_raster", "ndmi_raster", "rgb_raster", "raster_crs",
    "s2_candidate_date_this_run", "s2_candidate_coverage_this_run",
    "s2_clear_view_regressed_this_run", "s2_source", "s2_offset_rule",
]
S2_WINDOW_FIELDS = [
    "ndvi", "ndmi", "ndre", "evi", "s2_valid_coverage", "s2_acquisition_date",
    "s2_acq_ndvi", "s2_acq_ndmi", "s2_acq_ndre", "s2_acq_valid_coverage",
    "s2_scene_count",
]


# ============================================================
# STEP 3: PREPARE SENTINEL-2
# Mask unusable SCL classes, scale reflectance, calculate NDVI/NDMI/NDRE/EVI,
# and keep the original image metadata needed for later date/quality checks.
# ============================================================
def prepare_sentinel2(image: ee.Image) -> ee.Image:
    """Mask clouds/shadows/snow, scale reflectance, and add crop indices."""
    scl = image.select("SCL")
    clear = (
        scl.neq(1)
        .And(scl.neq(3))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    sr = (
        image.select(["B2", "B3", "B4", "B5", "B8", "B8A", "B11"])
        .multiply(0.0001)
        .updateMask(clear)
    )

    ndvi = sr.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = sr.normalizedDifference(["B8", "B11"]).rename("NDMI")
    ndre = sr.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    evi = sr.expression(
        "2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))",
        {"nir": sr.select("B8"), "red": sr.select("B4"), "blue": sr.select("B2")},
    ).rename("EVI")

    return sr.addBands([ndvi, ndmi, ndre, evi]).copyProperties(
        image, image.propertyNames()
    )


# STEP 4: BUILD THE SENTINEL-2 COLLECTION
# Keep scenes that intersect the farm and requested dates, reject very cloudy
# scenes at metadata level, then apply the Sentinel-2 preparation above.
def sentinel2_collection(farm_geometry, start_dt, end_dt):
    """Select only Sentinel-2 scenes intersecting the current farm."""
    return (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(farm_geometry)
        .filterDate(dt_string(start_dt), dt_string(end_dt))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        .map(prepare_sentinel2)
    )


# ============================================================
# STEP 5: PREPARE SENTINEL-1 RADAR
# Convert VV/VH from dB back to linear power before calculating RVI.
# RVI is retained as a screening signal, not treated as a replacement for NDVI.
# ============================================================
def prepare_sentinel1_rvi(image: ee.Image) -> ee.Image:
    """Add a dual-pol Radar Vegetation Index band to one Sentinel-1 scene.

    RVI = 4*VH / (VV + VH), following Trudel et al. (2012)'s dual-pol
    adaptation of Kim & van Zyl (2009)'s quad-pol RVI, as applied to
    Sentinel-1 VV/VH by Nasirzadehdizaji et al. (2019).

    UNIT WARNING: COPERNICUS/S1_GRD bands are stored in dB (10*log10(power)),
    per the Earth Engine dataset documentation. RVI is defined on linear
    power, so VV and VH are converted back to linear before the ratio is
    computed. Applying this ratio directly to dB values would silently
    produce a meaningless number, because dB is logarithmic, not linear.

    CALIBRATION STATUS: this is an uncalibrated prototype screening signal.
    It is NOT a validated substitute for an optical crop index. Radar
    backscatter also responds to soil moisture, surface roughness, flooding
    and tillage, so RVI is most likely to be confounded during exactly the
    wet conditions that make it necessary. Published correlations with crop
    biophysical variables come from other crops and climates and should not
    be assumed to transfer to this deployment without local field checks.
    """
    vv_linear = ee.Image(10).pow(image.select("VV").divide(10))
    vh_linear = ee.Image(10).pow(image.select("VH").divide(10))
    rvi = (
        vh_linear.multiply(4)
        .divide(vv_linear.add(vh_linear))
        .rename("RVI")
    )
    return image.addBands(rvi)


# STEP 6: BUILD THE SENTINEL-1 COLLECTION
# Select IW-mode scenes with both VV and VH over the farm and requested dates,
# then attach the RVI band to every scene.
def sentinel1_collection(farm_geometry, start_dt, end_dt):
    """Select Sentinel-1 VV/VH scenes for this farm and add the RVI band."""
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(farm_geometry)
        .filterDate(dt_string(start_dt), dt_string(end_dt))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .map(prepare_sentinel1_rvi)
    )


# ============================================================
# STEP 7: PREPARE SMAP SOIL MOISTURE
# Combine available AM and PM surface-soil-moisture retrievals. If only one
# retrieval exists, keep that one; if both exist, use their mean.
# ============================================================
def prepare_smap(image: ee.Image) -> ee.Image:
    """Combine available AM/PM SMAP surface-soil-moisture retrievals."""
    am = image.select("soil_moisture_am")
    pm = image.select("soil_moisture_pm")
    count = am.mask().unmask(0).add(pm.mask().unmask(0))
    combined = (
        am.unmask(0)
        .add(pm.unmask(0))
        .divide(count)
        .updateMask(count.gt(0))
        .rename("soil_moisture")
    )
    return combined.copyProperties(image, image.propertyNames())


# STEP 8: BUILD THE SMAP COLLECTION
# Filter SMAP to the supplied coarse context geometry and time period, then
# standardize each image with prepare_smap().
def smap_collection(farm_geometry, start_dt, end_dt):
    """Return the SMAP collection for the farm's coarse context geometry."""
    return (
        ee.ImageCollection(SMAP_COLLECTION)
        .filterBounds(farm_geometry)
        .filterDate(dt_string(start_dt), dt_string(end_dt))
        .map(prepare_smap)
    )


# ============================================================
# STEP 9: CHECK SENTINEL-2 FARM COVERAGE
# Measure how much of the farm still has valid NDVI after masking. This avoids
# accepting a scene that exists in the collection but is cloudy over the farm.
# ============================================================
def sentinel2_valid_coverage(composite, farm_geometry):
    """Return the fraction of the farm containing valid cloud-masked NDVI.

    A Sentinel-2 collection can contain scenes while every pixel over a small
    farm is masked by cloud or shadow. Checking this fraction prevents such a
    scene from being exported as an all-zero condition map.
    """
    # Calculate the area of pixels that remain valid after the NDVI mask.
    valid_area = (
        ee.Image.pixelArea()
        .rename("valid_area_m2")
        .updateMask(composite.select("NDVI").mask())
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=farm_geometry,
            scale=10,
            bestEffort=True,
            maxPixels=1_000_000_000,
        )
        .get("valid_area_m2")
        .getInfo()
    )
    # Divide valid pixel area by total farm area to obtain a 0-1 coverage fraction.
    farm_area = farm_geometry.area(maxError=10).getInfo()
    if valid_area is None or farm_area is None or float(farm_area) <= 0:
        return 0.0
    return max(0.0, min(1.0, float(valid_area) / float(farm_area)))


# STEP 10: FIND THE LATEST USABLE SENTINEL-2 SCENE
# Calculate valid farm coverage for each scene, keep scenes meeting the minimum
# threshold, and return the newest usable acquisition and its coverage.
def latest_usable_s2_observation(collection, farm_geometry):
    """Return the latest scene covering enough of this farm after masking.

    The collection's newest timestamp alone is not sufficient because its
    pixels over the farm may all be masked by cloud or shadow. Freshness is
    therefore based only on a scene whose valid farm coverage reaches the same
    threshold used to accept the optical composite.
    """
    # Attach a valid-coverage fraction to every scene so scenes can be filtered.
    farm_area = ee.Number(farm_geometry.area(maxError=10))

    def add_coverage(image):
        valid_area_value = (
            ee.Image.pixelArea()
            .rename("valid_area_m2")
            .updateMask(image.select("NDVI").mask())
            .reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=farm_geometry,
                scale=10,
                bestEffort=True,
                maxPixels=1_000_000_000,
            )
            .get("valid_area_m2")
        )
        valid_area = ee.Number(
            ee.Algorithms.If(valid_area_value, valid_area_value, 0)
        )
        return image.set("farm_valid_coverage", valid_area.divide(farm_area))

    # Prefer scenes meeting the configured minimum coverage; retain partial
    # coverage only for transparency when no scene reaches the threshold.
    annotated = collection.map(add_coverage)
    usable = annotated.filter(
        ee.Filter.gte("farm_valid_coverage", S2_MIN_VALID_COVERAGE)
    )
    if int(usable.size().getInfo() or 0) == 0:
        usable = annotated.filter(ee.Filter.gt("farm_valid_coverage", 0))
        if int(usable.size().getInfo() or 0) == 0:
            return None, 0.0

    latest = ee.Image(usable.sort("system:time_start", False).first())
    latest_dt = millis_to_datetime(latest.get("system:time_start").getInfo())
    latest_coverage = latest.get("farm_valid_coverage").getInfo()
    return (
        latest_dt.date().isoformat() if latest_dt else None,
        float(latest_coverage) if latest_coverage is not None else 0.0,
    )


# STEP 11: CHOOSE THE CURRENT OPTICAL COMPOSITE
# Search progressively wider fallback windows. For each window, create a median
# composite and accept the first one whose valid farm coverage meets the rule.
def choose_current_s2(farm_geometry, run_dt):
    """Return the first recent composite with adequate valid farm coverage."""
    last_count = 0
    best_coverage = 0.0
    # Try each configured lookback window from shortest to longest.
    for days in S2_FALLBACK_WINDOWS_DAYS:
        start_dt = run_dt - timedelta(days=days)
        collection = sentinel2_collection(farm_geometry, start_dt, run_dt)
        count = int(collection.size().getInfo() or 0)
        last_count = count
        if count == 0:
            continue
        composite = collection.median().clip(farm_geometry)
        coverage = sentinel2_valid_coverage(composite, farm_geometry)
        best_coverage = max(best_coverage, coverage)
        if coverage >= S2_MIN_VALID_COVERAGE:
            latest_date, latest_coverage = latest_usable_s2_observation(
                collection, farm_geometry
            )
            return (
                collection,
                composite,
                days,
                count,
                coverage,
                latest_date,
                latest_coverage,
            )
    return None, None, None, last_count, best_coverage, None, 0.0


# ============================================================
# STEPS 12-13: CLEAR-VIEW STATE
# load_s2_clear_view_state, save_s2_clear_view_state and
# resolve_persisted_clear_view are imported from clear_view_state.py.
# ============================================================


# ============================================================
# STEP 14: SHARED SPATIAL REDUCTION HELPERS
# farm_mean() clips an image and calculates mean band values inside the supplied
# geometry. latest_collection_date() returns the newest acquisition date.
# ============================================================
def farm_mean(image, bands, farm_geometry, scale):
    """
    Calculate means only through the current farm boundary.

    The image is clipped first and reduceRegion is also given the same geometry.
    This double restriction prevents neighbouring fields from entering the
    farm statistic for fine-resolution imagery.
    """
    clipped = image.clip(farm_geometry)
    result = (
        clipped.select(bands)
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=farm_geometry,
            scale=scale,
            bestEffort=True,
            maxPixels=1_000_000_000,
        )
        .getInfo()
    )
    return result or {}


def latest_collection_date(collection):
    """Return the most recent acquisition date in an Earth Engine collection."""
    if collection is None:
        return None
    latest_ms = collection.aggregate_max("system:time_start").getInfo()
    latest_dt = millis_to_datetime(latest_ms)
    return latest_dt.date().isoformat() if latest_dt else None


# ============================================================
# STEP 15: CURRENT SENTINEL-1 CHANGE ANALYSIS
# Compare the newest radar observation with an earlier baseline from the same
# relative orbit and pass direction so viewing geometry remains comparable.
# ============================================================
def sentinel1_change_summary(farm_geometry, run_dt):
    """Compare the newest radar acquisition with a same-orbit baseline.

    Sentinel-1 backscatter is sensitive to viewing geometry. The latest VV/VH
    values are therefore compared only with earlier observations that share
    the latest scene's relative orbit and ascending/descending pass. The
    resulting change is a structural radar signal; it is not an NDVI estimate
    and does not identify whether crop condition, water, tillage or another
    factor caused the shift.
    """
    # Start with a complete no-data result structure so the function can
    # safely return at any early-exit point without changing output columns.
    result = {
        "s1_scene_count": 0,
        "s1_latest_scene_count": 0,
        "latest_s1_date": None,
        "s1_relative_orbit": np.nan,
        "s1_orbit_pass": None,
        "current_vv_db": np.nan,
        "current_vh_db": np.nan,
        "s1_baseline_scene_count": 0,
        "s1_baseline_start_date": None,
        "s1_baseline_end_date": None,
        "baseline_vv_db": np.nan,
        "baseline_vh_db": np.nan,
        "vv_change_db": np.nan,
        "vh_change_db": np.nan,
        "current_rvi": np.nan,
        "baseline_rvi": np.nan,
        "rvi_change": np.nan,
    }

    # Search backwards for the newest Sentinel-1 acquisition available to the run.
    search_start = run_dt - timedelta(days=S1_SEARCH_DAYS)
    search = sentinel1_collection(farm_geometry, search_start, run_dt)
    search_count = int(search.size().getInfo() or 0)
    result["s1_scene_count"] = search_count
    if search_count == 0:
        return result

    # Use the newest scene to define the orbit and pass used for all comparisons.
    latest = ee.Image(search.sort("system:time_start", False).first())
    latest_ms = latest.get("system:time_start").getInfo()
    latest_dt = millis_to_datetime(latest_ms)
    if latest_dt is None:
        return result

    relative_orbit = latest.get("relativeOrbitNumber_start").getInfo()
    orbit_pass = latest.get("orbitProperties_pass").getInfo()
    result.update(
        {
            "latest_s1_date": latest_dt.date().isoformat(),
            "s1_relative_orbit": relative_orbit,
            "s1_orbit_pass": orbit_pass,
        }
    )

    # Build the current radar value from acquisitions on that same UTC day and
    # with the same orbit/pass, preventing incompatible viewing geometries.
    latest_day_start = latest_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    latest_day_end = min(latest_day_start + timedelta(days=1), run_dt)
    current = (
        sentinel1_collection(farm_geometry, latest_day_start, latest_day_end)
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )
    current_count = int(current.size().getInfo() or 0)
    result["s1_latest_scene_count"] = current_count
    if current_count == 0:
        return result

    current_stats = farm_mean(
        current.median(),
        ["VV", "VH", "RVI"],
        farm_geometry,
        20,
    )
    result["current_vv_db"] = current_stats.get("VV", np.nan)
    result["current_vh_db"] = current_stats.get("VH", np.nan)
    result["current_rvi"] = current_stats.get("RVI", np.nan)

    # Leave a gap before the current scene, then build the historical baseline
    # from the configured earlier period using the same orbit and pass.
    baseline_end = latest_day_start - timedelta(days=S1_BASELINE_GAP_DAYS)
    baseline_start = baseline_end - timedelta(days=S1_BASELINE_DAYS)
    baseline = (
        sentinel1_collection(farm_geometry, baseline_start, baseline_end)
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )
    baseline_count = int(baseline.size().getInfo() or 0)
    result.update(
        {
            "s1_baseline_scene_count": baseline_count,
            "s1_baseline_start_date": baseline_start.date().isoformat(),
            "s1_baseline_end_date": baseline_end.date().isoformat(),
        }
    )
    if baseline_count == 0:
        return result

    baseline_stats = farm_mean(
        baseline.median(),
        ["VV", "VH", "RVI"],
        farm_geometry,
        20,
    )
    baseline_vv = baseline_stats.get("VV", np.nan)
    baseline_vh = baseline_stats.get("VH", np.nan)
    baseline_rvi = baseline_stats.get("RVI", np.nan)
    result["baseline_vv_db"] = baseline_vv
    result["baseline_vh_db"] = baseline_vh
    result["baseline_rvi"] = baseline_rvi

    # Current minus baseline gives the VV/VH dB change; RVI change is a plain
    # linear-ratio difference rather than a dB difference.
    current_vv = result["current_vv_db"]
    current_vh = result["current_vh_db"]
    current_rvi = result["current_rvi"]
    if pd.notna(current_vv) and pd.notna(baseline_vv):
        result["vv_change_db"] = float(current_vv) - float(baseline_vv)
    if pd.notna(current_vh) and pd.notna(baseline_vh):
        result["vh_change_db"] = float(current_vh) - float(baseline_vh)
    if pd.notna(current_rvi) and pd.notna(baseline_rvi):
        result["rvi_change"] = float(current_rvi) - float(baseline_rvi)
    return result


# ============================================================
# STEP 16: RADAR CHECK SINCE THE LAST CLEAR OPTICAL DATE
# Anchor the radar comparison near the last clear S2 date and compare it with
# the latest same-orbit radar observation to flag unusually large VV/VH change.
# ============================================================
def sentinel1_gap_change_summary(farm_geometry, optical_date, run_dt):
    """Compare the last clear optical date with the latest same-orbit radar check.

    This does not convert radar into an optical crop index.  It only tests
    whether VV/VH backscatter changed materially after the optical assessment.
    The reference acquisition is the first same-orbit Sentinel-1 observation
    on or shortly after the optical date.  The latest acquisition uses that
    same relative orbit and pass direction so viewing geometry is comparable.
    """
    # Start with explicit no-data defaults; later steps overwrite these
    # fields only when suitable radar observations are available.
    result = {
        "s1_gap_status": "NO_DATA",
        "s1_gap_reference_date": None,
        "s1_gap_latest_date": None,
        "s1_gap_reference_relation": None,
        "s1_gap_reference_vv_db": np.nan,
        "s1_gap_reference_vh_db": np.nan,
        "s1_gap_latest_vv_db": np.nan,
        "s1_gap_latest_vh_db": np.nan,
        "s1_gap_vv_change_db": np.nan,
        "s1_gap_vh_change_db": np.nan,
    }

    # Stop early when there is no valid earlier optical date to anchor the check.
    parsed = pd.to_datetime(optical_date, errors="coerce", utc=True)
    if pd.isna(parsed):
        return result
    optical_dt = parsed.to_pydatetime()
    if optical_dt >= run_dt:
        return result

    # Search from the optical date to the run date and identify the latest radar scene.
    search = sentinel1_collection(farm_geometry, optical_dt, run_dt)
    if int(search.size().getInfo() or 0) == 0:
        return result

    # Use the newest scene to define the orbit and pass used for all comparisons.
    latest = ee.Image(search.sort("system:time_start", False).first())
    latest_dt = millis_to_datetime(latest.get("system:time_start").getInfo())
    if latest_dt is None:
        return result
    relative_orbit = latest.get("relativeOrbitNumber_start").getInfo()
    orbit_pass = latest.get("orbitProperties_pass").getInfo()

    # Build the current radar value from acquisitions on that same UTC day and
    # with the same orbit/pass, preventing incompatible viewing geometries.
    latest_day_start = latest_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    latest_day_end = min(latest_day_start + timedelta(days=1), run_dt)
    latest_same = (
        sentinel1_collection(farm_geometry, latest_day_start, latest_day_end)
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )
    if int(latest_same.size().getInfo() or 0) == 0:
        return result

    # If the newest radar pass is on the same UTC day as the last clear optical
    # image, no time has passed since the optical assessment, so there is no
    # change to test. Report that directly. Searching anyway would ask Earth
    # Engine for the empty date range [optical day, optical day), which it
    # rejects with "Empty date ranges not supported".
    optical_day_start = optical_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    if latest_day_start <= optical_day_start:
        result.update({
            "s1_gap_status": "INSUFFICIENT_INTERVAL",
            "s1_gap_latest_date": latest_dt.date().isoformat(),
        })
        return result

    # Prefer the first comparable same-orbit observation shortly AFTER the optical
    # date. If none exists, the code later falls back to a nearby BEFORE observation.
    ref_end = min(
        optical_dt + timedelta(days=S1_GAP_REFERENCE_MAX_DAYS),
        latest_day_start,
    )
    reference = (
        sentinel1_collection(farm_geometry, optical_dt, ref_end)
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .sort("system:time_start", True)
    )
    relation = "AFTER"
    if int(reference.size().getInfo() or 0) == 0:
        ref_start = optical_dt - timedelta(days=S1_GAP_REFERENCE_MAX_DAYS)
        reference = (
            sentinel1_collection(farm_geometry, ref_start, optical_dt)
            .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
            .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
            .sort("system:time_start", False)
        )
        relation = "BEFORE"
    if int(reference.size().getInfo() or 0) == 0:
        result["s1_gap_status"] = "INSUFFICIENT_REFERENCE"
        return result

    reference_image = ee.Image(reference.first())
    reference_dt = millis_to_datetime(reference_image.get("system:time_start").getInfo())
    if reference_dt is None or reference_dt.date() == latest_dt.date():
        result["s1_gap_status"] = "INSUFFICIENT_INTERVAL"
        return result

    ref_day_start = reference_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    ref_day_end = ref_day_start + timedelta(days=1)
    reference_same = (
        sentinel1_collection(farm_geometry, ref_day_start, ref_day_end)
        .filter(ee.Filter.eq("relativeOrbitNumber_start", relative_orbit))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
    )

    # Reduce the reference and latest radar observations over the exact farm.
    ref_stats = farm_mean(reference_same.median(), ["VV", "VH"], farm_geometry, 20)
    latest_stats = farm_mean(latest_same.median(), ["VV", "VH"], farm_geometry, 20)
    ref_vv, ref_vh = ref_stats.get("VV", np.nan), ref_stats.get("VH", np.nan)
    latest_vv, latest_vh = latest_stats.get("VV", np.nan), latest_stats.get("VH", np.nan)

    result.update({
        "s1_gap_reference_date": reference_dt.date().isoformat(),
        "s1_gap_latest_date": latest_dt.date().isoformat(),
        "s1_gap_reference_relation": relation,
        "s1_gap_reference_vv_db": ref_vv,
        "s1_gap_reference_vh_db": ref_vh,
        "s1_gap_latest_vv_db": latest_vv,
        "s1_gap_latest_vh_db": latest_vh,
    })
    if pd.isna(ref_vv) or pd.isna(ref_vh) or pd.isna(latest_vv) or pd.isna(latest_vh):
        result["s1_gap_status"] = "INSUFFICIENT_VALUES"
        return result

    # Compare absolute VV/VH changes with configured thresholds and return a
    # simple status: unusual change or no unusual change.
    vv_change = float(latest_vv) - float(ref_vv)
    vh_change = float(latest_vh) - float(ref_vh)
    unusual = (
        abs(vv_change) >= S1_VV_CHANGE_THRESHOLD_DB
        or abs(vh_change) >= S1_VH_CHANGE_THRESHOLD_DB
    )
    result.update({
        "s1_gap_vv_change_db": vv_change,
        "s1_gap_vh_change_db": vh_change,
        "s1_gap_status": "UNUSUAL_CHANGE" if unusual else "NO_UNUSUAL_CHANGE",
    })
    return result

# ============================================================
# STEP 17: EXTRACT GPM RAINFALL CONTEXT
# Because GPM is coarse, sample a buffered area around the farm rather than the
# exact polygon. Convert half-hour precipitation to millimetres and sum the period.
# ============================================================
def observed_rainfall_between(farm_geometry, start_dt, end_dt):
    """
    Sum GPM half-hour rainfall over a local context area around the farm.

    GPM is too coarse for exact field-scale reduction on small parcels. The
    sampling geometry is therefore a centroid buffer and the result must be
    interpreted as local/regional rainfall context for the farm location.
    """
    # Create the coarse rainfall context around the farm centroid because
    # GPM pixels are much larger than individual farm parcels.
    context_geometry = coarse_context_geometry(
        farm_geometry,
        GPM_CONTEXT_RADIUS_METERS,
    )
    collection = (
        ee.ImageCollection(GPM_COLLECTION)
        .filterBounds(context_geometry)
        .filterDate(dt_string(start_dt), dt_string(end_dt))
        .select("precipitation")
    )
    count = int(collection.size().getInfo() or 0)
    if count == 0:
        return np.nan, 0

    # GPM precipitation is a rate for each half-hour image; multiply by 0.5
    # hour, then sum all images to obtain rainfall depth for the period.
    rainfall = collection.map(
        lambda image: image.multiply(0.5).rename("rain_mm")
    ).sum()

    value = (
        rainfall.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=context_geometry,
            scale=GPM_SCALE_METERS,
            bestEffort=True,
            maxPixels=1_000_000_000,
        )
        .get("rain_mm")
        .getInfo()
    )
    return (float(value) if value is not None else np.nan), count


# STEP 18: BUILD DAILY OBSERVED RAINFALL
# Walk through the configured trend period one UTC day at a time and store each
# day's GPM rainfall plus the number of GPM images contributing to that value.
def observed_rainfall_daily_for_farm(farm_id, farm_geometry, run_dt, geom_key=None):
    """Return recent DAILY observed GPM rainfall context for one farm.

    Each row represents one completed/available UTC day. The same coarse local
    context rule used elsewhere is retained; these values are not presented as
    field-gauge measurements. Missing source data stay missing rather than being
    converted to zero. Settled days are reused from the history cache when
    ``geom_key`` is given.
    """
    rows = []
    day = (run_dt - timedelta(days=RAINFALL_TREND_DAYS)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_day = run_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    cache = (
        WindowCache("rain_daily", farm_id, geom_key, day - timedelta(days=7))
        if geom_key is not None else None
    )
    # Move through the trend period one day at a time and keep missing days as NaN.
    while day <= end_day:
        next_day = min(day + timedelta(days=1), run_dt)
        if next_day <= day:
            break
        cached = cache.get(day, next_day, days=1) if cache is not None else None
        if cached is not None:
            rain_mm, image_count = cached["rain_mm"], cached["gpm_image_count"]
        else:
            rain_mm, image_count = observed_rainfall_between(farm_geometry, day, next_day)
            if cache is not None:
                cache.put(day, next_day, {"rain_mm": rain_mm, "gpm_image_count": image_count})
        rows.append({
            "farm_id": farm_id,
            "date": day.date().isoformat(),
            "rain_mm": rain_mm,
            "gpm_image_count": image_count,
            "observation_type": "observed",
        })
        day = day + timedelta(days=1)
    if cache is not None:
        cache.save()
    return rows


# STEP 19: FIND THE LATEST AVAILABLE GPM OBSERVATION
# Search recent GPM data and return the newest timestamp available by the run date.
def latest_gpm_datetime(farm_geometry, run_dt):
    """Find the newest usable GPM observation available by the run date."""
    search_start = run_dt - timedelta(days=45)
    # Search GPM over the same coarse local context used for rainfall totals.
    context_geometry = coarse_context_geometry(
        farm_geometry,
        GPM_CONTEXT_RADIUS_METERS,
    )
    collection = (
        ee.ImageCollection(GPM_COLLECTION)
        .filterBounds(context_geometry)
        .filterDate(dt_string(search_start), dt_string(run_dt))
    )
    if int(collection.size().getInfo() or 0) == 0:
        return None
    return millis_to_datetime(collection.aggregate_max("system:time_start").getInfo())


# STEP 20: SUMMARIZE RECENT RAINFALL
# Using the latest available GPM timestamp, calculate rainfall totals for the
# previous 7 and 30 days.
def observed_rainfall_summary(farm_geometry, run_dt):
    """Calculate 7-day and 30-day observed rainfall context for one farm."""
    latest_dt = latest_gpm_datetime(farm_geometry, run_dt)
    if latest_dt is None:
        return {
            "rain_previous_7d_mm": np.nan,
            "rain_previous_30d_mm": np.nan,
            "rain_latest_date": None,
        }

    # Extend slightly beyond the latest half-hour timestamp so that observation
    # is included when calculating the 7-day and 30-day totals.
    end_dt = latest_dt + timedelta(minutes=31)
    rain_7d, _ = observed_rainfall_between(
        farm_geometry, end_dt - timedelta(days=7), end_dt
    )
    rain_30d, _ = observed_rainfall_between(
        farm_geometry, end_dt - timedelta(days=30), end_dt
    )
    return {
        "rain_previous_7d_mm": rain_7d,
        "rain_previous_30d_mm": rain_30d,
        "rain_latest_date": latest_dt.date().isoformat(),
    }


# ============================================================
# STEP 21: EXTRACT CURRENT SMAP SOIL-MOISTURE CONTEXT
# Build the configured coarse context area, average recent SMAP observations,
# and return the soil-moisture value together with the latest SMAP date.
# ============================================================
def current_soil_moisture(farm_geometry, run_dt):
    """Calculate recent SMAP local/regional context around the farm centroid."""
    # Create the coarse SMAP context because SMAP is not field-resolution data.
    context_geometry = coarse_context_geometry(
        farm_geometry,
        SMAP_CONTEXT_RADIUS_METERS,
    )
    smap = smap_collection(context_geometry, run_dt - timedelta(days=10), run_dt)
    count = int(smap.size().getInfo() or 0)
    if count == 0:
        return {"current_soil_moisture": np.nan, "latest_smap_date": None}

    values = farm_mean(
        smap.mean(),
        ["soil_moisture"],
        context_geometry,
        SMAP_SCALE_METERS,
    )
    return {
        "current_soil_moisture": values.get("soil_moisture", np.nan),
        "latest_smap_date": latest_collection_date(smap),
    }


# ============================================================
# STEP 22: EXPORT CURRENT SENTINEL-2 MAPS
# Determine the farm's UTM CRS, create a farm-specific output folder, and export
# NDVI, NDRE, NDMI and RGB GeoTIFFs clipped to the exact farm boundary.
# ============================================================
def export_current_rasters(farm_id, shapely_geometry, farm_geometry, composite):
    """
    Export field-resolution Sentinel-2 rasters for dashboard mapping.

    NDVI is exported at 10 m.
    NDRE and NDMI are exported at 20 m because they use Sentinel-2 bands whose
    native resolution is 20 m.

    Each farm receives its own folder, for example
    ``outputs/raw/FARM125/FARM125_ndvi.tif``. This prevents rasters from
    different farms being mixed together.
    """
    # Use the farm's local UTM CRS so exported rasters have an appropriate
    # projected coordinate system for measurement and mapping.
    crs = utm_crs_for_geometry(shapely_geometry)

    farm_raster_dir = RAW_DIR / str(farm_id)
    farm_raster_dir.mkdir(parents=True, exist_ok=True)

    # Define stable per-farm filenames before downloading each GeoTIFF.
    ndvi_path = farm_raster_dir / f"{farm_id}_ndvi.tif"
    ndre_path = farm_raster_dir / f"{farm_id}_ndre.tif"
    ndmi_path = farm_raster_dir / f"{farm_id}_ndmi.tif"
    rgb_path = farm_raster_dir / f"{farm_id}_rgb.tif"

    download_geotiff(
        composite.select("NDVI").clip(farm_geometry).unmask(RASTER_NODATA_VALUE),
        farm_geometry,
        ndvi_path,
        10,
        crs,
    )
    download_geotiff(
        composite.select("NDRE").clip(farm_geometry).unmask(RASTER_NODATA_VALUE),
        farm_geometry,
        ndre_path,
        20,
        crs,
    )
    download_geotiff(
        composite.select("NDMI").clip(farm_geometry).unmask(RASTER_NODATA_VALUE),
        farm_geometry,
        ndmi_path,
        20,
        crs,
    )
    download_geotiff(
        composite.select(["B4", "B3", "B2"]).clip(farm_geometry).unmask(
            RASTER_NODATA_VALUE
        ),
        farm_geometry,
        rgb_path,
        10,
        crs,
    )

    return ndvi_path, ndre_path, ndmi_path, rgb_path, crs


# ============================================================
# STEP 23: BUILD THE CURRENT MONITORING RECORD FOR ONE FARM
# Combine the accepted optical result, radar change metrics, rainfall and soil
# moisture into one output row while protecting the saved clear-view history.
# ============================================================
def gee_s2_fields(farm, farm_geometry, run_dt, s2_state):
    """Earth Engine Sentinel-2 part of the current record (S2_SOURCE "gee").

    Returns (fields, s2_state). ``fields`` holds exactly the S2_CURRENT_FIELDS
    keys that extract_s2_regional.py writes when S2_SOURCE is "stac".
    """
    farm_id = str(farm["farm_id"])
    (
        s2,
        composite,
        lookback,
        s2_count,
        valid_coverage,
        latest_s2_date,
        latest_s2_coverage,
    ) = choose_current_s2(farm_geometry, run_dt)

    # Store exactly what THIS run found before deciding whether it is allowed
    # to replace the best clear view already reported for the farm.
    candidate = {
        "date": latest_s2_date,
        "coverage": latest_s2_coverage,
        "lookback": lookback,
        "ndvi": np.nan,
        "ndmi": np.nan,
        "ndre": np.nan,
        "evi": np.nan,
        "ndvi_raster": "",
        "ndre_raster": "",
        "ndmi_raster": "",
        "rgb_raster": "",
        "raster_crs": "",
    }
    if s2 is not None and composite is not None:
        stats = farm_mean(
            composite,
            ["NDVI", "NDMI", "NDRE", "EVI"],
            farm_geometry,
            20,
        )
        candidate.update(
            {
                "ndvi": stats.get("NDVI", np.nan),
                "ndmi": stats.get("NDMI", np.nan),
                "ndre": stats.get("NDRE", np.nan),
                "evi": stats.get("EVI", np.nan),
            }
        )

    # Resolve the candidate against saved state. A regressed/older candidate
    # does not replace the previously accepted clear optical result.
    resolved, s2_state, regressed = resolve_persisted_clear_view(
        farm_id, candidate, s2_state
    )

    # Export new optical rasters only when this run's result is accepted.
    if not regressed and s2 is not None and composite is not None:
        ndvi_path, ndre_path, ndmi_path, rgb_path, crs = export_current_rasters(
            farm_id, farm.geometry, farm_geometry, composite
        )
        resolved["ndvi_raster"] = str(ndvi_path)
        resolved["ndre_raster"] = str(ndre_path)
        resolved["ndmi_raster"] = str(ndmi_path)
        resolved["rgb_raster"] = str(rgb_path)
        resolved["raster_crs"] = crs
        state_row = s2_state["farm_id"] == farm_id
        s2_state.loc[state_row, "best_ndvi_raster"] = resolved["ndvi_raster"]
        s2_state.loc[state_row, "best_ndre_raster"] = resolved["ndre_raster"]
        s2_state.loc[state_row, "best_ndmi_raster"] = resolved["ndmi_raster"]
        s2_state.loc[state_row, "best_rgb_raster"] = resolved["rgb_raster"]
        s2_state.loc[state_row, "best_raster_crs"] = resolved["raster_crs"]

    fields = {
        "s2_scene_count": s2_count,
        "s2_lookback_days": resolved.get("lookback"),
        "s2_valid_coverage": valid_coverage,
        "s2_latest_usable_coverage": resolved.get("coverage"),
        "latest_s2_date": resolved.get("date"),
        "current_ndvi": resolved.get("ndvi", np.nan),
        "current_ndmi": resolved.get("ndmi", np.nan),
        "current_ndre": resolved.get("ndre", np.nan),
        "current_evi": resolved.get("evi", np.nan),
        "ndvi_raster": resolved.get("ndvi_raster", ""),
        "ndre_raster": resolved.get("ndre_raster", ""),
        "ndmi_raster": resolved.get("ndmi_raster", ""),
        "rgb_raster": resolved.get("rgb_raster", ""),
        "raster_crs": resolved.get("raster_crs", ""),
        "s2_candidate_date_this_run": candidate.get("date"),
        "s2_candidate_coverage_this_run": candidate.get("coverage"),
        "s2_clear_view_regressed_this_run": bool(regressed),
        "s2_source": "gee:COPERNICUS/S2_SR_HARMONIZED",
        "s2_offset_rule": "harmonized_by_earth_engine",
    }
    return fields, s2_state


def current_metrics_for_farm(farm, farm_geometry, run_dt, s2_state, s2_override=None):
    """Build one monitoring record using only this farm's boundary.

    ``s2_state`` is the persisted best-clear-view table (see
    ``load_s2_clear_view_state``). It is mutated in place with this farm's
    resolved result so the caller can save it once after all farms run.

    ``s2_override`` is this farm's row from s2_current.csv when S2_SOURCE is
    "stac". Earth Engine is then not queried for Sentinel-2 at all and the
    state table is left to extract_s2_regional.py, which already updated it.
    """
    farm_id = str(farm["farm_id"])
    if s2_override is not None:
        s2_fields = {key: s2_override.get(key) for key in S2_CURRENT_FIELDS}
    else:
        s2_fields, s2_state = gee_s2_fields(farm, farm_geometry, run_dt, s2_state)

    # Assemble farm metadata, optical metrics and empty radar placeholders into
    # the single current-metrics record that downstream processing will use.
    row = {
        "farm_id": farm_id,
        "farm_name": farm.get("farm_name", farm_id),
        "crop_type": farm.get("crop_type", ""),
        "planting_date": farm.get("planting_date", ""),
        "polygon_file": farm.get("polygon_file", ""),
        "polygon_path": farm.get("polygon_path", ""),
        "run_datetime": run_dt.isoformat(),
        **s2_fields,
        "s1_scene_count": 0,
        "s1_latest_scene_count": 0,
        "latest_s1_date": None,
        "s1_relative_orbit": np.nan,
        "s1_orbit_pass": None,
        "current_vv_db": np.nan,
        "current_vh_db": np.nan,
        "s1_baseline_scene_count": 0,
        "s1_baseline_start_date": None,
        "s1_baseline_end_date": None,
        "baseline_vv_db": np.nan,
        "baseline_vh_db": np.nan,
        "vv_change_db": np.nan,
        "vh_change_db": np.nan,
        "s1_gap_status": "NO_DATA",
        "s1_gap_reference_date": None,
        "s1_gap_latest_date": None,
        "s1_gap_reference_relation": None,
        "s1_gap_reference_vv_db": np.nan,
        "s1_gap_reference_vh_db": np.nan,
        "s1_gap_latest_vv_db": np.nan,
        "s1_gap_latest_vh_db": np.nan,
        "s1_gap_vv_change_db": np.nan,
        "s1_gap_vh_change_db": np.nan,
        "current_rvi": np.nan,
        "baseline_rvi": np.nan,
        "rvi_change": np.nan,
    }

    # Add current radar-baseline change, radar-since-optical change, recent
    # rainfall and soil moisture to complete the farm's current record.
    row.update(sentinel1_change_summary(farm_geometry, run_dt))
    row.update(
        sentinel1_gap_change_summary(farm_geometry, row["latest_s2_date"], run_dt)
    )

    row.update(observed_rainfall_summary(farm_geometry, run_dt))
    row.update(current_soil_moisture(farm_geometry, run_dt))
    return row, s2_state


# ============================================================
# STEP 24: BUILD THE HISTORICAL TIME SERIES
# Windows follow the fixed 10-day grid in history_cache.py, so the same
# windows recur on every run. The Earth Engine part of each finished window
# (radar, rainfall, soil moisture) is cached per farm and reused, and only
# the newest windows are recomputed. Sentinel-2 comes either from
# extract_s2_regional.py (S2_SOURCE "stac") or, in the legacy "gee" mode,
# from Earth Engine every run.
# ============================================================
def _s2_window_external(external):
    """Sentinel-2 window values from s2_timeseries.csv (S2_SOURCE "stac")."""
    values = {
        "ndvi": np.nan, "ndmi": np.nan, "ndre": np.nan, "evi": np.nan,
        "s2_valid_coverage": np.nan, "s2_acquisition_date": None,
        "s2_acq_ndvi": np.nan, "s2_acq_ndmi": np.nan, "s2_acq_ndre": np.nan,
        "s2_acq_valid_coverage": np.nan, "s2_scene_count": 0,
    }
    if external is None:
        return values
    for key in values:
        if key in external:
            values[key] = external[key]
    if values["s2_acquisition_date"] is not None and pd.isna(values["s2_acquisition_date"]):
        values["s2_acquisition_date"] = None
    count = values["s2_scene_count"]
    values["s2_scene_count"] = int(count) if pd.notna(count) else 0
    return values


def _s2_window_gee(farm_geometry, window_start, window_end):
    """Sentinel-2 window values from Earth Engine (legacy S2_SOURCE "gee")."""
    s2 = sentinel2_collection(farm_geometry, window_start, window_end)
    s2_count = int(s2.size().getInfo() or 0)
    optical = {"NDVI": np.nan, "NDMI": np.nan, "NDRE": np.nan, "EVI": np.nan}
    values = {
        "s2_valid_coverage": np.nan, "s2_acquisition_date": None,
        "s2_acq_ndvi": np.nan, "s2_acq_ndmi": np.nan, "s2_acq_ndre": np.nan,
        "s2_acq_valid_coverage": np.nan, "s2_scene_count": s2_count,
    }
    if s2_count > 0:
        composite = s2.median().clip(farm_geometry)
        optical.update(
            farm_mean(composite, ["NDVI", "NDMI", "NDRE", "EVI"], farm_geometry, 20)
        )
        values["s2_valid_coverage"] = sentinel2_valid_coverage(composite, farm_geometry)

        scene_list = s2.sort("system:time_start", False).toList(s2_count)
        for scene_index in range(s2_count):
            scene = ee.Image(scene_list.get(scene_index))
            coverage = sentinel2_valid_coverage(scene, farm_geometry)
            if coverage < S2_MIN_VALID_COVERAGE:
                continue
            scene_dt = millis_to_datetime(scene.get("system:time_start").getInfo())
            if scene_dt is None:
                continue
            scene_stats = farm_mean(scene, ["NDVI", "NDMI", "NDRE"], farm_geometry, 20)
            values.update({
                "s2_acquisition_date": scene_dt.date().isoformat(),
                "s2_acq_ndvi": scene_stats.get("NDVI", np.nan),
                "s2_acq_ndmi": scene_stats.get("NDMI", np.nan),
                "s2_acq_ndre": scene_stats.get("NDRE", np.nan),
                "s2_acq_valid_coverage": coverage,
            })
            break
    values.update({
        "ndvi": optical.get("NDVI", np.nan), "ndmi": optical.get("NDMI", np.nan),
        "ndre": optical.get("NDRE", np.nan), "evi": optical.get("EVI", np.nan),
    })
    return values


def _context_window_gee(farm_geometry, window_start, window_end, run_dt):
    """Radar, GPM rainfall and SMAP soil moisture for one window (Earth Engine).

    This is the part that is cached once a window has settled.
    """
    # Radar block: retain the newest real Sentinel-1 acquisition in the window
    # and its orbit/pass metadata rather than mixing incompatible passes.
    s1 = sentinel1_collection(farm_geometry, window_start, window_end)
    s1_count = int(s1.size().getInfo() or 0)
    vv = vh = rvi = np.nan
    s1_acquisition_date = None
    s1_relative_orbit = np.nan
    s1_orbit_pass = None
    if s1_count > 0:
        latest_s1 = ee.Image(s1.sort("system:time_start", False).first())
        latest_s1_dt = millis_to_datetime(latest_s1.get("system:time_start").getInfo())
        if latest_s1_dt is not None:
            s1_acquisition_date = latest_s1_dt.date().isoformat()
            s1_relative_orbit = latest_s1.get("relativeOrbitNumber_start").getInfo()
            s1_orbit_pass = latest_s1.get("orbitProperties_pass").getInfo()
            day_start = latest_s1_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = min(day_start + timedelta(days=1), run_dt)
            same_day = (
                sentinel1_collection(farm_geometry, day_start, day_end)
                .filter(ee.Filter.eq("relativeOrbitNumber_start", s1_relative_orbit))
                .filter(ee.Filter.eq("orbitProperties_pass", s1_orbit_pass))
            )
            radar = farm_mean(
                same_day.median().clip(farm_geometry),
                ["VV", "VH", "RVI"],
                farm_geometry,
                20,
            )
            vv, vh = radar.get("VV", np.nan), radar.get("VH", np.nan)
            rvi = radar.get("RVI", np.nan)

    # Environmental-context block: GPM rainfall and SMAP soil moisture.
    rain_mm, gpm_count = observed_rainfall_between(farm_geometry, window_start, window_end)

    smap_context = coarse_context_geometry(farm_geometry, SMAP_CONTEXT_RADIUS_METERS)
    smap = smap_collection(smap_context, window_start, window_end)
    smap_count = int(smap.size().getInfo() or 0)
    soil = np.nan
    if smap_count > 0:
        soil = farm_mean(
            smap.mean(), ["soil_moisture"], smap_context, SMAP_SCALE_METERS
        ).get("soil_moisture", np.nan)

    return {
        "vv_db": vv, "vh_db": vh, "rvi": rvi,
        "s1_acquisition_date": s1_acquisition_date,
        "s1_relative_orbit": s1_relative_orbit,
        "s1_orbit_pass": s1_orbit_pass,
        "rain_mm": rain_mm, "soil_moisture": soil,
        "s1_scene_count": s1_count,
        "gpm_image_count": gpm_count,
        "smap_image_count": smap_count,
    }


def time_series_for_farm(farm_id, farm_geometry, run_dt, s2_windows=None, geom_key=None):
    """Build 10-day windows using field-scale and coarse-context inputs.

    ``s2_windows`` maps window_start (ISO date) to that window's Sentinel-2
    values from s2_timeseries.csv when S2_SOURCE is "stac".

    ``geom_key`` identifies the farm polygon for the history cache. When it
    is None the cache is not used and every window is computed, as before.
    """
    history_start = run_dt - timedelta(days=TIME_SERIES_DAYS)
    cache = (
        WindowCache("observed_context", farm_id, geom_key, history_start - timedelta(days=30))
        if geom_key is not None else None
    )

    rows = []
    for window_start, window_end in iter_windows(history_start, run_dt):
        if s2_windows is not None:
            optical = _s2_window_external(s2_windows.get(window_start.date().isoformat()))
        else:
            optical = _s2_window_gee(farm_geometry, window_start, window_end)

        context = cache.get(window_start, window_end) if cache is not None else None
        if context is None:
            context = _context_window_gee(farm_geometry, window_start, window_end, run_dt)
            if cache is not None:
                cache.put(window_start, window_end, context)

        rows.append({
            "farm_id": farm_id,
            "window_start": window_start.date().isoformat(),
            "window_end": window_end.date().isoformat(),
            "ndvi": optical["ndvi"],
            "ndmi": optical["ndmi"],
            "ndre": optical["ndre"],
            "evi": optical["evi"],
            "s2_valid_coverage": optical["s2_valid_coverage"],
            "s2_acquisition_date": optical["s2_acquisition_date"],
            "s2_acq_ndvi": optical["s2_acq_ndvi"],
            "s2_acq_ndmi": optical["s2_acq_ndmi"],
            "s2_acq_ndre": optical["s2_acq_ndre"],
            "s2_acq_valid_coverage": optical["s2_acq_valid_coverage"],
            "vv_db": context["vv_db"],
            "vh_db": context["vh_db"],
            "rvi": context["rvi"],
            "s1_acquisition_date": context["s1_acquisition_date"],
            "s1_relative_orbit": context["s1_relative_orbit"],
            "s1_orbit_pass": context["s1_orbit_pass"],
            "rain_mm": context["rain_mm"],
            "soil_moisture": context["soil_moisture"],
            "s2_scene_count": optical["s2_scene_count"],
            "s1_scene_count": context["s1_scene_count"],
            "gpm_image_count": context["gpm_image_count"],
            "smap_image_count": context["smap_image_count"],
        })

    if cache is not None:
        cache.save()
        print(
            f"[CropIntel] {farm_id}: history windows reused {cache.hits}, "
            f"computed {cache.misses}",
            flush=True,
        )
    return rows


# ============================================================
# STEP 25: RUN THE PIPELINE FOR ALL FARMS
# Initialise Earth Engine, load all farms, process each farm independently,
# then save current metrics, time series, daily rainfall and clear-view state.
# ============================================================
def main(run_datetime: str):
    """Extract current and historical observed data for every farm polygon."""
    # Parse the monitoring time and establish the Earth Engine session.
    run_dt = parse_utc_datetime(run_datetime)
    if not GEE_PROJECT_ID:
        raise RuntimeError(
            "GEE_PROJECT_ID is not set. Export your Earth Engine Google Cloud project ID before running."
        )
    initialise_earth_engine(GEE_PROJECT_ID)
    # Discover every configured farm polygon and load the saved S2 clear-view state.
    farms = load_farms()

    current_rows = []
    history_rows = []
    rain_daily_rows = []
    s2_state = load_s2_clear_view_state()

    # With S2_SOURCE "stac", Sentinel-2 comes from extract_s2_regional.py,
    # which must have run first with the same --datetime value.
    s2_current_by_farm = None
    s2_windows_by_farm = None
    if S2_SOURCE == "stac":
        s2_current_path = RAW_DIR / "s2_current.csv"
        s2_timeseries_path = RAW_DIR / "s2_timeseries.csv"
        if not s2_current_path.exists() or not s2_timeseries_path.exists():
            raise FileNotFoundError(
                "CROPINTEL_S2_SOURCE is 'stac' but s2_current.csv or "
                "s2_timeseries.csv is missing. Run extract_s2_regional.py "
                "first with the same --datetime."
            )
        s2_current = pd.read_csv(s2_current_path, dtype={"farm_id": str})
        s2_current_by_farm = {
            str(r["farm_id"]): r for r in s2_current.to_dict("records")
        }
        s2_timeseries = pd.read_csv(s2_timeseries_path, dtype={"farm_id": str})
        s2_windows_by_farm = {}
        for record in s2_timeseries.to_dict("records"):
            s2_windows_by_farm.setdefault(str(record["farm_id"]), {})[
                str(record["window_start"])
            ] = record

    # Process farms independently so every metric and raster remains tied to farm_id.
    for _, farm in farms.iterrows():
        farm_id = str(farm["farm_id"])
        farm_geometry = ee_geometry_from_shapely(farm.geometry)
        geom_key = geometry_key(farm.geometry)
        print(
            f"[CropIntel] {farm_id}: field data from {farm['polygon_file']}; "
            "GPM/SMAP sampled as coarse local context"
        )

        s2_override = None
        s2_windows = None
        if s2_current_by_farm is not None:
            s2_override = s2_current_by_farm.get(farm_id)
            if s2_override is None:
                raise ValueError(
                    f"{farm_id} is missing from s2_current.csv. Re-run "
                    "extract_s2_regional.py so every farm is included."
                )
            s2_windows = s2_windows_by_farm.get(farm_id, {})

        row, s2_state = current_metrics_for_farm(
            farm, farm_geometry, run_dt, s2_state, s2_override
        )
        if row.get("s2_clear_view_regressed_this_run"):
            print(
                f"[CropIntel] {farm_id}: today's search found "
                f"{row.get('s2_candidate_date_this_run')}, older than the "
                f"recorded {row.get('latest_s2_date')}; keeping the recorded date."
            )
        current_rows.append(row)
        history_rows.extend(
            time_series_for_farm(
                farm_id, farm_geometry, run_dt, s2_windows, geom_key
            )
        )
        rain_daily_rows.extend(
            observed_rainfall_daily_for_farm(farm_id, farm_geometry, run_dt, geom_key)
        )

    # Write the three analysis tables and persist the updated optical state for
    # comparison during the next scheduled run.
    pd.DataFrame(current_rows).to_csv(
        RAW_DIR / "current_metrics.csv", index=False
    )
    pd.DataFrame(history_rows).to_csv(
        RAW_DIR / "timeseries.csv", index=False
    )
    pd.DataFrame(rain_daily_rows).to_csv(
        RAW_DIR / "rain_observed_daily.csv", index=False
    )
    if S2_SOURCE == "gee":
        # In "stac" mode extract_s2_regional.py owns and saves this state.
        save_s2_clear_view_state(s2_state)


# ============================================================
# STEP 26: COMMAND-LINE ENTRY POINT
# Require a monitoring datetime, then pass it to main(). Airflow can call the
# same script by supplying this datetime argument.
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datetime", required=True)
    args = parser.parse_args()
    main(args.datetime)
