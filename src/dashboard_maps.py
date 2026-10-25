"""
Reusable spatial-map helpers for CropIntel dashboards.

Spatial interpretation rules
----------------------------
- NDVI, NDRE and NDMI come from Sentinel-2 and can support within-field maps.
- NDRE maps are nutrient-screening maps, not nitrogen-deficiency diagnoses.
- NDMI maps are crop/canopy water-status screening maps, not soil-moisture maps.
- ERA5-Land, GPM, SMAP and GFS are too coarse for credible within-field maps.
  Water & Climate therefore uses the RGB farm image only as a field-location
  context panel and keeps the coarse climate indicators as numbers/charts.

Execution steps
---------------
1. Resolve each raster path, including fallback lookup in the raw-data folder.
2. Read either a single index band or a three-band RGB image.
3. Enhance RGB contrast and saturation without modifying source data.
4. Keep missing RGB pixels transparent instead of drawing them as black blocks.
5. Reproject and overlay the matching farm boundary.
6. Add light, shape-based condition symbols while keeping the RGB visible.
7. Draw the map and return its image object for an optional color bar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.plot import plotting_extent
from rasterio.transform import from_bounds
from scipy.ndimage import gaussian_filter

from config import RAW_DIR


def resolve_raster_path(value) -> Path | None:
    """Resolve a stored raster path, with a fallback to RAW_DIR by filename."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None

    path = Path(str(value))
    if path.exists():
        return path

    candidate = RAW_DIR / path.name
    if candidate.exists():
        return candidate

    return None


def select_farm_boundary(farms, farm_id):
    """Return the GeoDataFrame row for one farm, or None when unavailable."""
    if farms is None or "farm_id" not in farms.columns:
        return None
    selected = farms[farms["farm_id"].astype(str) == str(farm_id)]
    return selected if not selected.empty else None


def read_single_band(path):
    """Read one raster band and return its masked array, extent and CRS."""
    path = resolve_raster_path(path)
    if path is None:
        return None, None, None

    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float32")
        array = np.ma.masked_where(
            (~np.isfinite(array)) | (array < -1.0) | (array > 1.0),
            array,
        )
        extent = plotting_extent(src)
        crs = src.crs
    return array, extent, crs


def read_rgb(path):
    """Read an RGB raster and return a display-ready RGBA image.

    Context rasters may contain a fourth 0/255 visibility band.  When present,
    that band is used as alpha so missing/cloud-masked pixels stay transparent
    rather than appearing as a black farm shape.
    """
    path = resolve_raster_path(path)
    if path is None:
        return None, None, None

    with rasterio.open(path) as src:
        if src.count < 3:
            return None, None, None
        rgb_bands = src.read([1, 2, 3]).astype("float32")
        extent = plotting_extent(src)
        crs = src.crs
        if src.count >= 4:
            alpha = src.read(4).astype("float32")
            valid = alpha > 0
        else:
            valid = src.dataset_mask() > 0

    rgb = np.moveaxis(rgb_bands, 0, -1)

    # Display-ready context exports are uint8-like values in the 0..255 range.
    # Keep their natural colour balance instead of stretching each band
    # separately.  Older/current reflectance rasters still use percentile
    # stretching below.
    finite = np.isfinite(rgb)
    finite_values = rgb[finite]
    looks_display_ready = (
        finite_values.size > 0
        and np.nanmin(finite_values) >= 0
        and np.nanmax(finite_values) > 1.5
        and np.nanmax(finite_values) <= 255.0
    )

    if looks_display_ready:
        enhanced = np.clip(rgb / 255.0, 0, 1)
    else:
        stretched = np.zeros_like(rgb, dtype="float32")
        for i in range(3):
            band = rgb[:, :, i]
            good = np.isfinite(band) & valid
            if not good.any():
                continue
            lo, hi = np.nanpercentile(band[good], [1, 99])
            if hi <= lo:
                hi = lo + 1e-6
            stretched[:, :, i] = np.clip((band - lo) / (hi - lo), 0, 1)

        luminance = (
            0.2126 * stretched[:, :, 0]
            + 0.7152 * stretched[:, :, 1]
            + 0.0722 * stretched[:, :, 2]
        )
        enhanced = luminance[:, :, None] + 1.18 * (
            stretched - luminance[:, :, None]
        )
        enhanced = 0.5 + 1.10 * (enhanced - 0.5)
        enhanced = np.power(np.clip(enhanced, 0, 1), 0.96)

    # dataset_mask() (used above when there's no 4th alpha band) can only
    # detect no-data pixels if the GeoTIFF has an explicit nodata value set
    # in its metadata. Many real Earth Engine exports don't set one, and
    # fill masked/no-coverage pixels with near-zero values instead. Checked
    # against the *mean* of the three bands rather than requiring each band
    # individually below threshold: an earlier version required all three
    # channels separately below the cutoff, which broke as soon as sensor
    # noise pushed any single channel a little above it, even though the
    # pixel was still obviously near-black overall. A genuine, brightly lit
    # optical reading (vegetation, soil, water) essentially never averages
    # this close to zero, so this is a safe, deliberate threshold.
    NEAR_ZERO_THRESHOLD = 6.0
    near_zero_fill = rgb.mean(axis=2) <= NEAR_ZERO_THRESHOLD
    valid &= ~near_zero_fill

    valid &= np.isfinite(rgb).all(axis=2)
    rgba = np.dstack((enhanced, valid.astype("float32")))
    return rgba, extent, crs


def _plot_boundary(ax, farm_boundary, crs, color="white", linewidth=2.4):
    """Reproject and draw a farm boundary on an existing map axis.

    Drawn as a dark halo underneath a lighter line on top, so the boundary
    stays readable regardless of what colour the imagery underneath happens
    to be (a plain white line can disappear against pale soil or cloud, and
    a plain dark line can disappear against dense dark-green canopy).
    """
    if farm_boundary is None:
        return

    boundary = farm_boundary.copy()
    if boundary.crs is not None and crs is not None:
        boundary = boundary.to_crs(crs)

    boundary.boundary.plot(
        ax=ax,
        linewidth=linewidth + 1.6,
        edgecolor="#12261C",
        zorder=4.5,
    )
    boundary.boundary.plot(
        ax=ax,
        linewidth=linewidth,
        edgecolor=color,
        zorder=4.6,
    )


def _farm_pixel_mask(farm_boundary, crs, shape, extent):
    """Return True only for raster pixels inside the selected farm polygon.

    Earth Engine downloads are rectangular, even when the requested region is
    an irregular farm. This mask removes that surrounding rectangle from the
    dashboard while preserving the original pixels inside the field.
    """
    if farm_boundary is None or getattr(farm_boundary, "empty", True):
        return np.ones(shape, dtype=bool)

    boundary = farm_boundary.copy()
    if boundary.crs is not None and crs is not None:
        boundary = boundary.to_crs(crs)

    geometries = [
        geometry.__geo_interface__
        for geometry in boundary.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not geometries:
        return np.ones(shape, dtype=bool)

    left, right, bottom, top = extent
    transform = from_bounds(left, bottom, right, top, shape[1], shape[0])
    return geometry_mask(
        geometries,
        out_shape=shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )


def _clip_rgba_to_farm(rgba, extent, crs, farm_boundary):
    """Make RGB pixels outside the farm transparent."""
    clipped = rgba.copy()
    inside = _farm_pixel_mask(farm_boundary, crs, clipped.shape[:2], extent)
    clipped[:, :, 3] *= inside.astype("float32")
    return clipped


def _style_map_axis(ax):
    """Remove ticks and borders so the axis is presented as a clean map."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_index_map(
    ax,
    raster_path,
    farm_boundary=None,
    *,
    cmap="RdYlGn",
    vmin=None,
    vmax=None,
    rgb_path=None,
    alpha=0.82,
    relative_low_quantile=0.25,
):
    """
    Draw one field-resolution Sentinel-2 index.

    A dashed contour optionally marks the lowest relative quantile within the
    valid farm pixels. This is a within-field screening zone, not a universal
    agronomic threshold.
    """
    index, extent, crs = read_single_band(raster_path)
    if index is None:
        ax.text(
            0.5, 0.5, "MAP NOT AVAILABLE",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=11, fontweight="bold",
        )
        _style_map_axis(ax)
        return None, None

    if rgb_path:
        rgb, rgb_extent, _ = read_rgb(rgb_path)
        if rgb is not None:
            ax.imshow(rgb, extent=rgb_extent, origin="upper")

    data = np.asarray(index.filled(np.nan), dtype=float)
    valid = np.isfinite(data)

    if valid.any():
        if vmin is None:
            vmin = float(np.nanpercentile(data[valid], 2))
        if vmax is None:
            vmax = float(np.nanpercentile(data[valid], 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6

    image = ax.imshow(
        index,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
    )

    relative_threshold = None
    if valid.any() and relative_low_quantile is not None:
        relative_threshold = float(
            np.nanquantile(data[valid], relative_low_quantile)
        )
        low_binary = np.where(
            valid,
            (data <= relative_threshold).astype(float),
            np.nan,
        )

        finite_low = low_binary[np.isfinite(low_binary)]
        if finite_low.size and np.nanmax(finite_low) == 1 and np.nanmin(finite_low) == 0:
            ax.contour(
                low_binary,
                levels=[0.5],
                colors="white",
                linewidths=1.3,
                linestyles="--",
                extent=extent,
                origin="upper",
            )

    _plot_boundary(ax, farm_boundary, crs)
    _style_map_axis(ax)
    return image, relative_threshold


# NOTE: draw_rgb_context_map() used to be defined twice in this file. Python
# silently keeps only the LAST definition of a repeated function name, so the
# earlier copy (draw the RGB context without bilinear interpolation, boundary
# linewidth 1.8, no return value) was never actually reachable -- every call
# in this codebase was already running the surviving definition further
# below (bilinear interpolation, boundary linewidth 2.0, returns True/False).
# The dead copy was removed rather than fixed in place, since a second
# definition that happens to look plausible is a standing risk: reordering
# these two functions, or deleting only the second one during a future edit,
# would silently change dashboard rendering with no error raised anywhere.


def draw_symbolized_rgb_map(
    ax,
    rgb_path,
    index_path,
    farm_boundary=None,
    *,
    breaks=(0.35, 0.55),
    max_symbols=420,
):
    """Draw deepened RGB imagery with a light three-level symbol overlay.

    The overlay uses shapes rather than opaque colour classes:

    - an ``x`` marks pixels below the lower screening break;
    - a small dot marks pixels between the two breaks; and
    - an open circle marks pixels above the upper break.

    This keeps the true-colour field visible. The categories are screening
    aids only and must be checked against crop stage and field observations.
    """
    rgb, rgb_extent, rgb_crs = read_rgb(rgb_path)
    if rgb is None:
        ax.text(
            0.5, 0.5, "FIELD CONTEXT MAP NOT AVAILABLE",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
        )
        _style_map_axis(ax)
        return False

    rgb = _clip_rgba_to_farm(rgb, rgb_extent, rgb_crs, farm_boundary)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    # Bilinear display softens block edges in a static PNG. It does not create
    # new spatial detail: the underlying Sentinel-2 RGB pixels remain 10 m.
    ax.imshow(rgb, extent=rgb_extent, origin="upper", interpolation="bilinear")

    index, index_extent, index_crs = read_single_band(index_path)
    if index is not None:
        data = np.asarray(index.filled(np.nan), dtype=float)
        valid = np.isfinite(data)
        valid &= _farm_pixel_mask(
            farm_boundary,
            index_crs,
            data.shape,
            index_extent,
        )
        if rgb.shape[:2] == data.shape and np.allclose(rgb_extent, index_extent):
            valid &= rgb[:, :, 3] > 0
        lower, upper = sorted(float(value) for value in breaks)
        categories = np.full(data.shape, -1, dtype="int8")
        categories[valid & (data < lower)] = 0
        categories[valid & (data >= lower) & (data < upper)] = 1
        categories[valid & (data >= upper)] = 2

        # Limit symbol density so the satellite texture remains visible. The
        # stride is deterministic, which makes repeated reports comparable.
        valid_count = int(valid.sum())
        stride = max(1, int(np.ceil(np.sqrt(valid_count / max_symbols))))
        rows = np.arange(0, data.shape[0], stride)
        columns = np.arange(0, data.shape[1], stride)
        row_grid, column_grid = np.meshgrid(rows, columns, indexing="ij")

        left, right, bottom, top = index_extent
        x_step = (right - left) / data.shape[1]
        y_step = (top - bottom) / data.shape[0]
        x_grid = left + (column_grid + 0.5) * x_step
        y_grid = top - (row_grid + 0.5) * y_step
        sampled = categories[row_grid, column_grid]

        symbol_specs = (
            # marker, colour, size, linewidth, face style
            (0, "x", "#7A2E26", 30, 1.6, None),
            (1, ".", "#8C5A12", 26, 1.3, None),
            (2, "o", "#1B4F38", 24, 1.4, "none"),
        )
        for category, marker, colour, size, width, face in symbol_specs:
            selected = sampled == category
            if not np.any(selected):
                continue
            kwargs = {
                "marker": marker,
                "s": size,
                "linewidths": width,
                "alpha": 0.92,
                "zorder": 4,
            }
            if face == "none":
                kwargs.update(facecolors="none", edgecolors=colour)
            else:
                kwargs.update(c=colour)
            ax.scatter(x_grid[selected], y_grid[selected], **kwargs)

        boundary_crs = index_crs
    else:
        boundary_crs = rgb_crs
        ax.text(
            0.02, 0.03, "Condition symbols unavailable",
            transform=ax.transAxes, fontsize=7, color="#5B665F",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )

    _plot_boundary(ax, farm_boundary, boundary_crs, linewidth=3.0)
    _style_map_axis(ax)
    return index is not None


def draw_scale_bar(ax, index_path, *, y_frac=0.06, color="white"):
    """Draw a real-distance scale bar from a raster's own UTM extent.

    Only correct because export_current_rasters() (extract_data.py) always
    projects farm rasters to a UTM zone via utm_crs_for_geometry() before
    download -- extent is therefore already in metres, not degrees. Calling
    this against a raster in a geographic CRS (EPSG:4326) would silently
    produce a wrong scale, since one degree of longitude is not a fixed
    distance; this function does not attempt to detect or guard against
    that case, so it must only be used with rasters known to be projected.
    """
    path = resolve_raster_path(index_path)
    if path is None:
        return
    with rasterio.open(path) as src:
        if src.crs is not None and src.crs.is_geographic:
            return  # refuse to draw a distance scale in degree units
        extent = plotting_extent(src)
    left, right, _bottom, _top = extent
    width_m = right - left
    if width_m <= 0:
        return

    # Pick a "nice" round bar length (~35-45% of the map width).
    target = width_m * 0.4
    magnitude = 10 ** np.floor(np.log10(max(target, 1)))
    for step in (1, 2, 5, 10):
        candidate = step * magnitude
        if candidate >= target:
            bar_m = candidate
            break
    else:
        bar_m = 10 * magnitude

    bar_frac = bar_m / width_m
    x0 = 0.06
    y = y_frac
    ax.plot([x0, x0 + bar_frac], [y, y], transform=ax.transAxes, color=color, linewidth=2.2, solid_capstyle="butt")
    for fraction in (0.0, 0.5, 1.0):
        ax.plot([x0 + bar_frac * fraction] * 2, [y - 0.012, y + 0.012], transform=ax.transAxes, color=color, linewidth=1.6)
    label = f"{bar_m:.0f} m" if bar_m < 1000 else f"{bar_m / 1000:.1f} km"
    ax.text(x0 + bar_frac, y + 0.028, label, transform=ax.transAxes, fontsize=7.5,
            color=color, ha="right", family="DejaVu Sans")


def draw_north_arrow(ax, *, x=0.93, y=0.90, size=0.05, color="white"):
    """Draw a simple decorative north indicator.

    This is a conventional up-is-north marker, not a georeferenced compass
    computed from the raster's actual orientation -- acceptable for context
    display, but it should not be read as a precise bearing.
    """
    ax.annotate(
        "", xy=(x, y), xytext=(x, y - size),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 2.0, "mutation_scale": 14},
    )
    ax.text(x, y + 0.018, "N", transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=color, ha="center", family="DejaVu Sans")


def draw_index_colorbar(fig, image, cax_rect, *, label="NDVI"):
    """Draw a vertical colorbar for a draw_index_map() image.

    Separate from draw_index_map() because a colorbar needs its own axes
    (cax_rect), sized and positioned by the caller's layout.
    """
    if image is None:
        return None
    cax = fig.add_axes(cax_rect)
    bar = fig.colorbar(image, cax=cax)
    bar.set_label(label, fontsize=9.5, fontweight="bold", color="#173228")
    bar.ax.tick_params(labelsize=8, color="#173228", labelcolor="#173228")
    for spine in bar.ax.spines.values():
        spine.set_visible(False)
    return bar


def _hex_to_rgb01(hex_color: str):
    """Convert a #RRGGBB hex string to an (r, g, b) tuple in the 0-1 range."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def draw_true_color_with_bands(ax, rgb_path, index_path, farm_boundary=None,
                                *, breaks, band_colors=None, band_alpha=0.42,
                                edge_sigma=0.9):
    """Draw the plain true-colour field photo, with the low, moderate, AND
    good condition-band pixels tinted directly in colour (not just
    outlined).

    breaks has no default on purpose: the caller must pass the actual
    per-index breakpoints (config.py's NDVI_*_BREAK / NDMI_*_BREAK /
    NDRE_*_BREAK constants, picked by report mode in field_report.py's
    MAP_SETTINGS) rather than this function silently falling back to a
    stale guess. An earlier version defaulted to NDVI's old (0.35, 0.55)
    breakpoints here, duplicating -- and, after that pair was updated
    elsewhere, capable of silently drifting from -- the real single source
    of truth in config.py. The same NDVI_LOW_BREAK/NDVI_GOOD_BREAK-style
    constants also drive the indicator-badge text next to the map
    (field_report.py's _condition_word), so a legend and badge built from
    those same values can never disagree about what "low"/"moderate"/"good"
    means for a given index.

    All three bands are tinted, including "good", so a field that is
    genuinely growing well is visibly marked as such rather than only
    implied by the absence of a low/moderate tint. NDVI map conventions
    (e.g. AgroReports' NDVI colour-scale reference) consistently give
    healthy vegetation its own distinct colour rather than leaving it
    unclassified, and this pipeline's own status colour for "good"
    (GREEN, used elsewhere for the hero number and priority badge) was
    already defined for the map legend -- it just was not being painted.

    band_colors lets the caller supply the exact colours it will also use in
    its own legend, so the two can never drift out of sync the way an
    earlier version did (this function had its own hardcoded colours,
    separate from and only coincidentally close to the legend's).

    The tint is rendered with a soft edge (edge_sigma, in source pixels)
    instead of the raw pixel grid's hard staircase boundary, and band_alpha
    defaults lower than a fully opaque block (0.42) so the true-colour photo
    still reads through the tint -- together this is meant to read as a
    translucent status wash over a real photograph, not a flat colour-block
    chart. This is a rendering choice only: bands_found and every band
    membership decision are still computed from the untouched, un-blurred
    per-pixel classification below, so nothing about which pixels count as
    low/moderate/good, or any number this pipeline reports elsewhere, is
    affected by the blur. The blur is done on alpha-premultiplied colour so
    band colours don't fringe into fully-transparent pixels as the edge
    softens.

    Returns (drawn, bands_found), where bands_found is a set containing
    "low", "moderate", and/or "good" for whichever bands actually had
    pixels in this field -- the caller should only show legend entries for
    bands that are genuinely present, not the full set unconditionally.
    """
    if band_colors is None:
        band_colors = {"low": "#B3261E", "moderate": "#D9A404", "good": "#2F4F3D"}

    rgb, rgb_extent, rgb_crs = read_rgb(rgb_path)
    if rgb is None:
        ax.text(
            0.5, 0.5, "FIELD VIEW UNAVAILABLE",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
        )
        _style_map_axis(ax)
        return False, set()

    rgb = _clip_rgba_to_farm(rgb, rgb_extent, rgb_crs, farm_boundary)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.imshow(rgb, extent=rgb_extent, origin="upper", interpolation="bilinear")

    bands_found = set()
    index, index_extent, index_crs = read_single_band(index_path)
    if index is not None:
        data = np.asarray(index.filled(np.nan), dtype=float)
        valid = np.isfinite(data)
        valid &= _farm_pixel_mask(farm_boundary, index_crs, data.shape, index_extent)
        lower, upper = sorted(float(v) for v in breaks)

        overlay = np.zeros((*data.shape, 4), dtype="float32")
        for band_name, mask in (
            ("low", valid & (data < lower)),
            ("moderate", valid & (data >= lower) & (data < upper)),
            ("good", valid & (data >= upper)),
        ):
            if not mask.any():
                continue
            colour_rgb = _hex_to_rgb01(band_colors[band_name])
            overlay[mask, 0] = colour_rgb[0]
            overlay[mask, 1] = colour_rgb[1]
            overlay[mask, 2] = colour_rgb[2]
            overlay[mask, 3] = band_alpha
            bands_found.add(band_name)

        if bands_found:
            # Soften the hard per-pixel boundary into a natural-looking
            # transition. Naively blurring straight RGBA would bleed each
            # band's colour into the fully-transparent pixels around it
            # (a dark fringe, since unset colour channels are 0,0,0), so the
            # colour is premultiplied by alpha before blurring and divided
            # back out after -- the standard way to blur an RGBA image
            # without that fringing.
            alpha = overlay[..., 3]
            premultiplied = overlay[..., :3] * alpha[..., None]
            premultiplied = gaussian_filter(premultiplied, sigma=(edge_sigma, edge_sigma, 0))
            alpha = gaussian_filter(alpha, sigma=edge_sigma)
            safe_alpha = np.maximum(alpha, 1e-6)[..., None]
            overlay = np.dstack([premultiplied / safe_alpha, alpha]).astype("float32")
            ax.imshow(overlay, extent=index_extent, origin="upper", interpolation="bilinear")

    _plot_boundary(ax, farm_boundary, rgb_crs, linewidth=2.0)
    _style_map_axis(ax)
    return True, bands_found


def draw_rgb_context_map(ax, rgb_path, farm_boundary=None):
    """Draw a plain field view without condition symbols or technical labels."""
    rgb, rgb_extent, rgb_crs = read_rgb(rgb_path)
    if rgb is None:
        ax.text(
            0.5, 0.5, "FIELD VIEW UNAVAILABLE",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold",
        )
        _style_map_axis(ax)
        return False

    rgb = _clip_rgba_to_farm(rgb, rgb_extent, rgb_crs, farm_boundary)
    ax.set_facecolor("none")
    ax.patch.set_alpha(0)
    ax.imshow(rgb, extent=rgb_extent, origin="upper", interpolation="bilinear")
    _plot_boundary(ax, farm_boundary, rgb_crs, linewidth=2.0)
    _style_map_axis(ax)
    return True