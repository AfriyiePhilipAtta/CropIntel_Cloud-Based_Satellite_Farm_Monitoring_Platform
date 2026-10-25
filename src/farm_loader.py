"""Load validated CropIntel farms from ``data/farms.csv`` and GeoJSON."""

from __future__ import annotations

from pathlib import Path
import re

import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from config import DATE_DAY_FIRST, FARM_REGISTRY_FILE, POLYGON_DIR


REQUIRED_COLUMNS = [
    "farm_id", "farm_name", "region", "district", "crop", "variety",
    "planting_date", "production_system", "boundary_file", "contact",
]


def _safe_id(value: str) -> str:
    """Return a filename-safe farm identifier without accepting blanks."""
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value).strip()).strip("_")
    if not value:
        raise ValueError("farm_id cannot be blank")
    return value


def _repair_polygon(geometry, source: Path):
    """Require a usable polygon and attempt a standard topology repair."""
    if geometry is None or geometry.is_empty:
        raise ValueError(f"Empty geometry in {source}")
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(
            f"CropIntel requires Polygon/MultiPolygon boundaries; "
            f"{source.name} contains {geometry.geom_type}."
        )
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError(f"Invalid polygon could not be repaired: {source}")
    return geometry


def _parse_date_value(value):
    """Parse one common CSV date value and return a pandas timestamp.

    Accepted examples include ``2026-06-01``, ``06/01/2026``,
    ``01-06-2026``, ``1 June 2026`` and Excel-style date-time strings.
    Ambiguous numeric dates follow ``CROPINTEL_DATE_DAY_FIRST``.
    """
    text = str(value).strip()
    if not text:
        return pd.NaT

    # Support five-digit Excel serial dates occasionally written to CSV.
    if re.fullmatch(r"\d{5}(?:\.0+)?", text):
        serial = float(text)
        return pd.Timestamp("1899-12-30") + pd.to_timedelta(serial, unit="D")

    return pd.to_datetime(
        text,
        errors="coerce",
        format="mixed",
        dayfirst=DATE_DAY_FIRST,
    )


def load_farm_registry(path: Path = FARM_REGISTRY_FILE) -> pd.DataFrame:
    """Load the farm registry and validate fields used by the pipeline."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Farm registry not found: {path}")
    registry = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in registry.columns]
    if missing:
        raise ValueError(f"farms.csv is missing columns: {missing}")
    if registry.empty:
        raise ValueError("farms.csv must contain at least one farm")

    for column in REQUIRED_COLUMNS:
        registry[column] = registry[column].astype(str).str.strip()
    required_values = [
        "farm_id", "farm_name", "region", "district", "crop",
        "planting_date", "production_system", "boundary_file",
    ]
    for column in required_values:
        blank_rows = registry.index[registry[column].eq("")].tolist()
        if blank_rows:
            raise ValueError(
                f"farms.csv column {column!r} is blank on rows "
                f"{[row + 2 for row in blank_rows]}"
            )

    registry["farm_id"] = registry["farm_id"].map(_safe_id)
    if registry["farm_id"].duplicated().any():
        duplicates = registry.loc[
            registry["farm_id"].duplicated(), "farm_id"
        ].tolist()
        raise ValueError(f"Duplicate farm_id values in farms.csv: {duplicates}")
    parsed_dates = registry["planting_date"].map(_parse_date_value)
    if parsed_dates.isna().any():
        rows = (parsed_dates.index[parsed_dates.isna()] + 2).tolist()
        raise ValueError(
            "Invalid planting_date on rows "
            f"{rows}. Use a recognizable calendar date."
        )
    # Downstream scripts always receive one consistent, unambiguous format.
    registry["planting_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    registry["production_system"] = registry["production_system"].str.lower()
    if not registry["production_system"].isin({"rainfed", "irrigated"}).all():
        raise ValueError("production_system must be rainfed or irrigated")
    return registry


def _read_boundary(path: Path) -> tuple[object, str]:
    """Read one boundary file and dissolve multiple polygon features."""
    if not path.is_file():
        raise FileNotFoundError(f"Farm boundary not found: {path}")
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Farm boundary contains no features: {path}")
    if gdf.crs is None:
        if path.suffix.lower() in {".geojson", ".json"}:
            gdf = gdf.set_crs("EPSG:4326")
        else:
            raise ValueError(f"Farm boundary has no CRS: {path}")
    source_crs = gdf.crs.to_string()
    gdf = gdf.to_crs("EPSG:4326")
    polygons = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if polygons.empty:
        raise ValueError(f"No Polygon/MultiPolygon found in {path}")
    geometry = polygons.geometry.union_all()
    return _repair_polygon(geometry, path), source_crs


def load_farms(
    polygon_dir: Path = POLYGON_DIR,
    registry_path: Path = FARM_REGISTRY_FILE,
) -> gpd.GeoDataFrame:
    """Return one validated WGS84 farm row for each registry entry."""
    polygon_dir = Path(polygon_dir).expanduser().resolve()
    registry = load_farm_registry(registry_path)
    rows = []
    for _, record in registry.iterrows():
        boundary_name = Path(record["boundary_file"])
        if boundary_name.name != record["boundary_file"]:
            raise ValueError(
                f"Farm {record['farm_id']}: boundary_file must be a filename only"
            )
        boundary_path = polygon_dir / boundary_name
        geometry, source_crs = _read_boundary(boundary_path)
        # Total farm area, for display alongside the other farm-record
        # fields. Computed by reprojecting to an appropriate local UTM zone
        # (geopandas' own estimate_utm_crs, not gee_helpers - this avoids
        # adding an Earth Engine dependency to a file that currently has
        # none) rather than measuring in WGS84 degrees, which does not give
        # a true area.
        utm_crs = gpd.GeoSeries([geometry], crs="EPSG:4326").estimate_utm_crs()
        total_area_ha = (
            gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(utm_crs).area.iloc[0]
            / 10_000
        )
        rows.append(
            {
                "farm_id": record["farm_id"],
                "farm_name": record["farm_name"],
                "region": record["region"],
                "district": record["district"],
                "crop_type": record["crop"],
                "variety": record["variety"],
                "planting_date": record["planting_date"],
                "production_system": record["production_system"],
                "polygon_file": boundary_name.name,
                "polygon_path": str(boundary_path),
                "source_crs": source_crs,
                "total_area_ha": total_area_ha,
                "geometry": geometry,
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
