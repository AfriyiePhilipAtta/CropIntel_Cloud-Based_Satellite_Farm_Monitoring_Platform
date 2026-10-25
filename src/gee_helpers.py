"""Shared Google Earth Engine and raster-download utilities for CropIntel.

The functions in this module initialize Earth Engine, convert Python and
Shapely objects to Earth Engine representations, select an appropriate UTM
coordinate reference system, and download GeoTIFF outputs.

Earth Engine credentials are not stored in this source file. Authentication
uses the file specified by the GOOGLE_APPLICATION_CREDENTIALS environment
variable. If that variable is not set, Earth Engine attempts to use previously
authenticated personal credentials.

Processing sequence
-------------------
1. Connect to the configured Google Earth Engine project.
2. Parse scheduler dates and standardize them as UTC.
3. Convert Shapely farm polygons to Earth Engine geometries.
4. Create local sampling areas for coarse environmental datasets.
5. Select the appropriate UTM coordinate reference system.
6. Request and save prototype-sized GeoTIFF outputs.
7. Convert Earth Engine timestamps to UTC datetimes.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ee
import requests


def initialise_earth_engine(project_id: str) -> None:
    """Initialize Google Earth Engine using configured credentials.

    When GOOGLE_APPLICATION_CREDENTIALS identifies a service-account JSON
    file, this function extracts the service-account email and authenticates
    explicitly with that file.

    If GOOGLE_APPLICATION_CREDENTIALS is empty, the function falls back to
    personal credentials previously created by ``earthengine authenticate``.

    Parameters
    ----------
    project_id:
        Google Cloud project identifier registered for Earth Engine access.

    Raises
    ------
    RuntimeError
        If the project ID is missing, the JSON file is invalid, the
        service-account email is missing, or Earth Engine rejects access.
    FileNotFoundError
        If the configured service-account JSON file does not exist.
    """
    # Earth Engine requires a Cloud project for quota and access management.
    project_id = str(project_id).strip()

    if not project_id:
        raise RuntimeError(
            "GEE_PROJECT_ID is not configured. Add it to the .env file."
        )

    # Docker supplies this path from the docker-compose environment section.
    credentials_path = os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "",
    ).strip()

    # Use explicit service-account authentication when a file is configured.
    if credentials_path:
        key_path = Path(credentials_path).expanduser().resolve()

        # Fail with the exact expected location if Docker did not mount it.
        if not key_path.is_file():
            raise FileNotFoundError(
                f"Earth Engine credential file not found: {key_path}"
            )

        # Read only the metadata needed to identify the service account.
        try:
            with key_path.open("r", encoding="utf-8") as credential_file:
                key_information = json.load(credential_file)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid service-account JSON file: {key_path}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(
                f"Could not read the service-account file: {key_path}"
            ) from exc

        # Google service-account keys contain the account email in this field.
        service_account_email = str(
            key_information.get("client_email", "")
        ).strip()

        if not service_account_email:
            raise RuntimeError(
                "The service-account JSON does not contain client_email."
            )

        # Build the Earth Engine credentials from the private-key file.
        credentials = ee.ServiceAccountCredentials(
            service_account_email,
            str(key_path),
        )

        # Pass both credentials and project explicitly for unattended runs.
        ee.Initialize(
            credentials=credentials,
            project=project_id,
        )

        return

    # This fallback works only when personal Earth Engine credentials are
    # already available inside the execution environment.
    ee.Initialize(project=project_id)


def parse_utc_datetime(value: str) -> datetime:
    """Parse a date or ISO-8601 timestamp and return a UTC datetime.

    Parameters
    ----------
    value:
        Either a date in ``YYYY-MM-DD`` format or an ISO-8601 timestamp.

    Returns
    -------
    datetime
        A timezone-aware datetime normalized to UTC.

    Raises
    ------
    ValueError
        If the supplied value cannot be parsed as a supported date or
        timestamp.
    """
    # Remove accidental spaces received from CLI or scheduler arguments.
    value = str(value).strip()

    if not value:
        raise ValueError("A run date or timestamp is required.")

    # A date without a time component is interpreted as midnight UTC.
    if len(value) == 10:
        parsed = datetime.strptime(value, "%Y-%m-%d")
        return parsed.replace(tzinfo=timezone.utc)

    # Python accepts +00:00 but not the Z suffix in some older environments.
    normalized_value = value.replace("Z", "+00:00")

    # Parse timestamps that include a time and optional timezone offset.
    parsed = datetime.fromisoformat(normalized_value)

    # Treat a timestamp without timezone information as UTC.
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    # Convert offset-aware timestamps to one consistent UTC representation.
    return parsed.astimezone(timezone.utc)


def dt_string(value: datetime) -> str:
    """Convert a datetime to the timestamp format expected by Earth Engine.

    Parameters
    ----------
    value:
        A timezone-aware or naive Python datetime.

    Returns
    -------
    str
        Timestamp formatted as ``YYYY-MM-DDTHH:MM:SS`` in UTC.
    """
    # Assign UTC to naive datetimes to prevent dependence on server local time.
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    # Normalize the value before constructing the Earth Engine date string.
    return value.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def ee_geometry_from_shapely(geometry: Any) -> ee.Geometry:
    """Convert a Shapely geometry into an Earth Engine geometry.

    Parameters
    ----------
    geometry:
        A Shapely geometry exposing the ``__geo_interface__`` property.

    Returns
    -------
    ee.Geometry
        Earth Engine geometry representing the same coordinates.

    Raises
    ------
    TypeError
        If the object does not provide a GeoJSON-compatible representation.
    """
    if not hasattr(geometry, "__geo_interface__"):
        raise TypeError(
            "The supplied geometry does not provide __geo_interface__."
        )

    return ee.Geometry(geometry.__geo_interface__)


def coarse_context_geometry(
    farm_geometry: ee.Geometry,
    radius_m: float,
) -> ee.Geometry:
    """Create a local sampling area for coarse environmental datasets.

    GPM, SMAP and GFS pixels are much larger than many farm parcels. Reducing
    these products over a very small polygon can return no value or create a
    false impression of field-level precision. The function therefore creates
    a documented buffer around the farm centroid.

    Parameters
    ----------
    farm_geometry:
        Earth Engine geometry representing the farm boundary.
    radius_m:
        Buffer radius in metres.

    Returns
    -------
    ee.Geometry
        Circular local-context geometry around the farm centroid.

    Notes
    -----
    Values sampled with this geometry are local or regional context. They must
    not be described as measurements made inside the exact farm boundary.
    """
    radius_m = float(radius_m)

    if radius_m <= 0:
        raise ValueError("The context radius must be greater than zero.")

    # The first error margin supports the centroid calculation.
    centroid = farm_geometry.centroid(maxError=100)

    # The second error margin supports construction of the buffer boundary.
    return centroid.buffer(
        distance=radius_m,
        maxError=100,
    )


def utm_crs_for_geometry(geometry: Any) -> str:
    """Select a UTM CRS from a WGS84 geometry centroid.

    Parameters
    ----------
    geometry:
        Shapely geometry whose coordinates are longitude and latitude.

    Returns
    -------
    str
        EPSG identifier for the relevant northern or southern UTM zone.

    Raises
    ------
    ValueError
        If the centroid coordinates fall outside valid longitude or latitude
        ranges.
    """
    # Shapely calculates the centroid in the geometry's current coordinate
    # system. Farm geometries must therefore be converted to WGS84 first.
    centroid = geometry.centroid

    longitude = float(centroid.x)
    latitude = float(centroid.y)

    if not -180 <= longitude <= 180:
        raise ValueError(
            f"Longitude outside the WGS84 range: {longitude}"
        )

    if not -90 <= latitude <= 90:
        raise ValueError(
            f"Latitude outside the WGS84 range: {latitude}"
        )

    # UTM divides longitude into 60 zones, each approximately six degrees wide.
    zone = int((longitude + 180) // 6) + 1

    # Longitude exactly equal to 180 would otherwise produce invalid zone 61.
    zone = max(1, min(zone, 60))

    # EPSG 326xx represents northern zones and 327xx southern zones.
    epsg = (32600 if latitude >= 0 else 32700) + zone

    return f"EPSG:{epsg}"


def download_geotiff(
    image: ee.Image,
    geometry: ee.Geometry,
    output_path: Path,
    scale: float,
    crs: str,
) -> Path:
    """Download a prototype-sized Earth Engine image as GeoTIFF.

    Parameters
    ----------
    image:
        Earth Engine image to export.
    geometry:
        Earth Engine region used to clip the export.
    output_path:
        Local path where the GeoTIFF will be written.
    scale:
        Requested pixel size in metres.
    crs:
        Output coordinate reference system, such as ``EPSG:32630``.

    Returns
    -------
    Path
        Resolved path of the downloaded GeoTIFF.

    Raises
    ------
    ValueError
        If the requested scale is not positive.
    requests.HTTPError
        If Earth Engine's generated download URL returns an HTTP error.
    requests.RequestException
        If the network request fails or times out.

    Notes
    -----
    ``getDownloadURL`` is appropriate for prototype-sized farm downloads.
    Larger production exports should use Earth Engine batch export tasks and
    managed object storage.
    """
    scale = float(scale)

    if scale <= 0:
        raise ValueError("GeoTIFF export scale must be greater than zero.")

    # Normalize the output path so callers receive a consistent Path object.
    output_path = Path(output_path).expanduser().resolve()

    # Create the destination directory before starting the network request.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Ask Earth Engine to clip, scale and reproject the selected image.
    download_url = image.getDownloadURL(
        {
            "region": geometry,
            "scale": scale,
            "crs": str(crs),
            "format": "GEO_TIFF",
        }
    )

    # Stream the response to avoid holding a complete GeoTIFF in memory.
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    try:
        with requests.get(
            download_url,
            timeout=240,
            stream=True,
        ) as response:
            # Convert unsuccessful HTTP responses into explicit exceptions.
            response.raise_for_status()

            # Write fixed-size chunks to the temporary file.
            with temporary_path.open("wb") as destination:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        destination.write(chunk)

        # Rename only after a complete download to avoid partial final files.
        temporary_path.replace(output_path)

    except Exception:
        # Remove an incomplete temporary file when a request fails.
        temporary_path.unlink(missing_ok=True)
        raise

    return output_path


def millis_to_datetime(value: Any) -> datetime | None:
    """Convert Earth Engine milliseconds since epoch to a UTC datetime.

    Parameters
    ----------
    value:
        Millisecond timestamp returned by Earth Engine, or ``None``.

    Returns
    -------
    datetime or None
        UTC datetime when a timestamp exists; otherwise ``None``.
    """
    if value is None:
        return None

    # Earth Engine timestamps are milliseconds; Python expects seconds.
    seconds_since_epoch = float(value) / 1000.0

    return datetime.fromtimestamp(
        seconds_since_epoch,
        tz=timezone.utc,
    )