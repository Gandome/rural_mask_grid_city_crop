#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-quality city/UHI + rural-reference overlay for MOD_Mask v2.

Version 2 offers two rural-area modes:

``exact`` (recommended)
    Reconstruct the actual union of rural reference cells used by the MOD_Mask
    search for the urban model cells inside each selected GHSL city polygon.
    This requires the PGD file and reads the thresholds/search settings from
    the UHI NetCDF global attributes whenever available.

``radius_box``
    Backward-compatible visual fallback reproducing the older city-centred box
    expanded by the maximum accepted ``nbg``. It must not be interpreted as
    the exact MOD_Mask rural reference footprint.

Raw UHI values are never percentile-normalized or rescaled. Curvilinear ALPX3
lon/lat coordinates are drawn with ``pcolormesh`` rather than ``imshow``.
"""
from __future__ import annotations

import argparse
import glob
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Point
from shapely.prepared import prep
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# Import the sibling grid_uhi_mask package directly from a repository clone.
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
GRID_ROOT = REPO_ROOT / "grid_uhi_mask"
if str(GRID_ROOT) not in sys.path:
    sys.path.insert(0, str(GRID_ROOT))

from spatial_UHI_mask import build_masks, find_rural_reference_once, to_yx  # noqa: E402


# ============================================================
# USER SETTINGS
# ============================================================
UHI_DIR = Path(
    "/archive/globc/quenum/Results/Data_UHI_Paper/UHI_MOD_MASK_V2_0_1/"
    "Min70_sea0p30_urb0p20_rur0p60"
)

PGD_FILE = Path(
    "/archive/globc/quenum/Results/Data_UHI_Paper/"
    "PGD_ALPX3_selected_alpine_cities_buffer_1p0deg.nc"
)

CITY_GPKG = REPO_ROOT / "clim_city_mask" / "data" / "GHS_UCDB_REGION_EUROPE_R2024A.gpkg"

OUT_DIR = Path("./output_city_uhi_figures_RAW_v2")
FIG_DIR = OUT_DIR / "figures"
NC_DIR = OUT_DIR / "netcdf"

TARGET_CITIES = [
    "Grenoble",
    "Chambery",
    "Geneva",
]

CITY_NAME_COLUMN = "GC_UCN_MAI_2025"
CITY_GPKG_LAYER = "GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A"
UHI_VAR = "UHI_px"
TIME_INDEX = 0

# "radius_box" reproduces the supplied standalone workflow without requiring PGD.
# Use "exact" when PGD_FILE is the PGD matching the UHI grid; this reconstructs
# the true MOD_Mask reference footprint and is recommended for method figures.
RURAL_MODE = "exact"
DEFAULT_R = 60

RAW_UHI_VMIN = -4.0
RAW_UHI_VMAX = 10.0
CITY_CMAP = "RdYlBu_r"

RURAL_ALPHA = 0.92
CITY_ALPHA = 1.0
# Nearest preserves model-grid values. Change to "gouraud" only for a purely
# visual rendering; no interpolated values are written to NetCDF.
PCOLORMESH_SHADING = "auto"

SAVE_DPI = 600
FIG_DPI = 180
FIGSIZE = (12.5, 8.0)


# ============================================================
# BASIC UTILITIES
# ============================================================
def sanitize_filename(name):
    name = str(name)
    for ch in [" ", "/", "\\", ":", ";", ",", "(", ")", "[", "]", "{", "}", "'", '"']:
        name = name.replace(ch, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("_")


def find_first_existing_var(ds, candidates):
    for name in candidates:
        if name in ds.variables:
            return name
    raise KeyError(f"None of {candidates} found. Available: {list(ds.variables)}")


def open_first_uhi_file(uhi_dir):
    files = sorted(glob.glob(str(uhi_dir / "*.nc")) + glob.glob(str(uhi_dir / "*.nc4")))
    if not files:
        raise FileNotFoundError(f"No NetCDF file found in {uhi_dir}")
    print(f"Opening UHI file:\n{files[0]}")
    return xr.open_dataset(files[0], decode_times=True), Path(files[0])


def get_lon_lat_names(ds):
    return (
        find_first_existing_var(ds, ["lon", "longitude", "LON", "XLONG"]),
        find_first_existing_var(ds, ["lat", "latitude", "LAT", "XLAT"]),
    )


def _attr(ds, name, default, cast=float):
    try:
        return cast(ds.attrs.get(name, default))
    except (TypeError, ValueError):
        return cast(default)


# ============================================================
# CITY POLYGONS
# ============================================================
def load_city_polygons(gpkg_file, target_names, name_col, layer=None):
    gdf = gpd.read_file(gpkg_file, layer=layer) if layer else gpd.read_file(gpkg_file)
    if name_col not in gdf.columns:
        possible = [c for c in gdf.columns if any(k in c.lower() for k in ["name", "city", "ucn", "urban"])]
        if not possible:
            raise KeyError(f"City-name column '{name_col}' not found. Columns={list(gdf.columns)}")
        name_col = possible[0]
        print(f"Using inferred city-name column: {name_col}")

    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

    rows = []
    for requested in target_names:
        names = gdf[name_col].astype(str)
        exact = gdf[names.str.casefold() == requested.casefold()]
        match = exact if not exact.empty else gdf[names.str.contains(requested, case=False, na=False, regex=False)]
        if not match.empty:
            row = match.iloc[[0]].copy()
            row["__requested_city__"] = requested
            rows.append(row)
        else:
            print(f"Warning: city not found in GeoPackage: {requested}")

    if not rows:
        raise ValueError("No target cities found in GHSL file")

    selected = gpd.GeoDataFrame(pd.concat(rows, ignore_index=True), crs=gdf.crs)
    selected["__city_name__"] = selected[name_col].astype(str).apply(sanitize_filename)
    print("Selected cities:")
    for city in selected["__city_name__"]:
        print(f"  - {city}")
    return selected


def grid_points_inside_polygon(lon2d, lat2d, polygon):
    """Vectorized point-in-polygon when Shapely >=2 is available."""
    try:
        from shapely import contains_xy

        inside = contains_xy(polygon, lon2d, lat2d)
        return np.asarray(inside, dtype=bool)
    except Exception:
        prepared = prep(polygon)
        inside = np.zeros(lon2d.size, dtype=bool)
        for i, (lo, la) in enumerate(zip(lon2d.ravel(), lat2d.ravel())):
            if np.isfinite(lo) and np.isfinite(la):
                # covers() includes grid points exactly on the polygon boundary.
                p = Point(float(lo), float(la))
                inside[i] = prepared.contains(p) or polygon.touches(p)
        return inside.reshape(lon2d.shape)


def crop_to_union_bbox(mask_a, mask_b=None, pad=2):
    mask = np.asarray(mask_a, bool).copy()
    if mask_b is not None:
        mask |= np.asarray(mask_b, bool)
    yy, xx = np.where(mask)
    if yy.size == 0:
        return None, None
    ny, nx = mask.shape
    return (
        slice(max(int(yy.min()) - pad, 0), min(int(yy.max()) + pad + 1, ny)),
        slice(max(int(xx.min()) - pad, 0), min(int(xx.max()) + pad + 1, nx)),
    )


# ============================================================
# UHI + RURAL REFERENCE MASKS
# ============================================================
def get_raw_uhi_2d(ds, uhi_var=UHI_VAR, time_index=TIME_INDEX):
    da = ds[uhi_var]
    if "time" in da.dims:
        return da.isel(time=time_index).values.astype(float)
    return da.values.astype(float)


def estimate_search_radius_from_nbg(ds, city_mask):
    if "nbg" not in ds:
        return int(DEFAULT_R)
    da = ds["nbg"]
    if "time" in da.dims:  # compatibility with v1 outputs
        da = da.isel(time=0)
    vals = np.asarray(da.values)[city_mask]
    vals = vals[np.isfinite(vals)]
    return max(1, int(np.nanmax(vals))) if vals.size else int(DEFAULT_R)


def create_radius_box_rural_mask(lon2d, lat2d, city_mask, R):
    yslice, xslice = crop_to_union_bbox(city_mask, pad=int(R))
    if yslice is None:
        raise ValueError("City mask is empty")
    local_domain = np.zeros(city_mask.shape, bool)
    local_domain[yslice, xslice] = True
    rural = local_domain & ~city_mask & np.isfinite(lon2d) & np.isfinite(lat2d)
    return rural, rural.astype(np.int32)


def reconstruct_exact_rural_reference(ds, city_mask, pgd_file):
    """Reconstruct the true rural-reference union for urban cells in a city."""
    with xr.open_dataset(pgd_file) as pgd:
        town = to_yx(pgd["SFX.FRAC_TOWN"]).values.astype(np.float32)
        nature = to_yx(pgd["SFX.FRAC_NATURE"]).values.astype(np.float32)
        sea = to_yx(pgd["SFX.FRAC_SEA"]).values.astype(np.float32)
        water = to_yx(pgd["SFX.FRAC_WATER"]).values.astype(np.float32)
        elevation = to_yx(pgd["SFX.ZS"]).values.astype(np.float32)

    if town.shape != city_mask.shape:
        raise ValueError(f"PGD shape {town.shape} does not match UHI grid {city_mask.shape}")

    urban_threshold = _attr(ds, "urban_threshold", 0.20)
    rural_threshold = _attr(ds, "rural_threshold", 0.60)
    sea_water_threshold = _attr(ds, "sea_water_threshold", 0.30)
    min_value = _attr(ds, "Min_Value_requested", 70.0)
    min_floor = _attr(ds, "Min_Value_floor", 50.0)
    ratio_step = _attr(ds, "Min_Value_step", 5.0)
    nO = _attr(ds, "nO", 2, int)
    initial_nbg = _attr(ds, "initial_nbg", 4, int)
    max_iterations = _attr(ds, "max_iterations", 26, int)

    sea_water_mask, _, urban_mask, _ = build_masks(
        xr.DataArray(town, dims=("y", "x")),
        xr.DataArray(nature, dims=("y", "x")),
        xr.DataArray(sea, dims=("y", "x")),
        xr.DataArray(water, dims=("y", "x")),
        urban_threshold,
        rural_threshold,
        sea_water_threshold,
    )
    sw = sea_water_mask.values.astype(bool)
    model_urban_in_city = city_mask & urban_mask.values.astype(bool)
    urban_points = np.argwhere(model_urban_in_city)

    rural_union = np.zeros(city_mask.shape, bool)
    reference_count = np.zeros(city_mask.shape, np.int32)
    accepted = 0
    failed = 0
    mismatches = 0

    nbg_out = None
    if "nbg" in ds:
        nbg_da = ds["nbg"].isel(time=0) if "time" in ds["nbg"].dims else ds["nbg"]
        nbg_out = np.asarray(nbg_da.values)

    for y, x in urban_points:
        ref = find_rural_reference_once(
            elevation=elevation,
            rural_frac=nature,
            urban_frac=town,
            sea_water_mask=sw,
            urban_point=(int(y), int(x)),
            rural_threshold=rural_threshold,
            urban_threshold=urban_threshold,
            min_value=min_value,
            nbg=initial_nbg,
            max_iterations=max_iterations,
            nO=nO,
            min_ratio_floor=min_floor,
            ratio_step=ratio_step,
        )
        if not ref.valid:
            failed += 1
            continue
        accepted += 1
        rural_union[ref.yy, ref.xx] = True
        np.add.at(reference_count, (ref.yy, ref.xx), 1)
        if nbg_out is not None and np.isfinite(nbg_out[y, x]) and not np.isclose(float(nbg_out[y, x]), ref.nbg):
            mismatches += 1

    metadata = {
        "rural_mode": "exact",
        "urban_cells_in_city": int(len(urban_points)),
        "accepted_urban_references": int(accepted),
        "failed_urban_references": int(failed),
        "nbg_consistency_mismatches": int(mismatches),
        "urban_threshold": urban_threshold,
        "rural_threshold": rural_threshold,
        "sea_water_threshold": sea_water_threshold,
        "Min_Value_requested": min_value,
        "Min_Value_floor": min_floor,
        "nO": nO,
        "initial_nbg": initial_nbg,
        "max_iterations": max_iterations,
    }
    return rural_union, reference_count, model_urban_in_city, metadata


def rural_green_cmap():
    cmap = LinearSegmentedColormap.from_list(
        "modmask_rural_green",
        ["#e8f5e9", "#a5d6a7", "#66bb6a", "#2e7d32", "#0b4f20"],
        N=256,
    )
    cmap.set_bad(alpha=0.0)
    return cmap


def rural_display_field(rural_mask, reference_count):
    """Green intensity proportional to reference reuse in exact mode."""
    out = np.full(rural_mask.shape, np.nan, float)
    vals = np.asarray(reference_count, float)
    if np.any(rural_mask):
        vmax = float(np.nanmax(vals[rural_mask]))
        if vmax > 1:
            scaled = np.log1p(vals) / np.log1p(vmax)
            out[rural_mask] = 0.25 + 0.75 * scaled[rural_mask]
        else:
            out[rural_mask] = 0.72
    return out


# ============================================================
# PLOTTING
# ============================================================
def add_scalebar(ax, length_km=20, location=(0.025, 0.035), linewidth=5):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    lon_left = x0 + location[0] * (x1 - x0)
    lat_bar = y0 + location[1] * (y1 - y0)
    lat_mean = 0.5 * (y0 + y1)
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat_mean))
    if km_per_deg_lon <= 0:
        return
    deg_len = length_km / km_per_deg_lon

    ax.plot([lon_left, lon_left + deg_len], [lat_bar, lat_bar], color="black", linewidth=linewidth,
            solid_capstyle="butt", zorder=20)
    ax.plot([lon_left, lon_left + deg_len / 2], [lat_bar, lat_bar], color="white",
            linewidth=linewidth * 0.55, solid_capstyle="butt", zorder=21)
    for x, label in [(lon_left, "0"), (lon_left + deg_len / 2, f"{int(length_km/2)}"),
                     (lon_left + deg_len, f"{int(length_km)} km")]:
        ax.text(x, lat_bar + 0.015 * (y1 - y0), label, ha="center", va="bottom", fontsize=10, weight="bold")


def plot_city_and_rural_overlay_raw(
    lon2d,
    lat2d,
    city_overlay,
    city_polygon_mask,
    rural_mask,
    reference_count,
    polygon,
    city_name,
    metadata,
    save_path,
):
    yslice, xslice = crop_to_union_bbox(city_polygon_mask, rural_mask, pad=2)
    if yslice is None:
        raise ValueError(f"Nothing to plot for {city_name}")

    lon = lon2d[yslice, xslice]
    lat = lat2d[yslice, xslice]
    city_local = city_overlay[yslice, xslice]
    rural_local = rural_mask[yslice, xslice]
    count_local = reference_count[yslice, xslice]
    rural_field = rural_display_field(rural_local, count_local)

    city_vals = city_local[np.isfinite(city_local)]
    if city_vals.size == 0:
        raise ValueError(f"No valid {UHI_VAR} values within {city_name}")

    vmin = float(np.nanmin(city_vals)) if RAW_UHI_VMIN is None else float(RAW_UHI_VMIN)
    vmax = float(np.nanmax(city_vals)) if RAW_UHI_VMAX is None else float(RAW_UHI_VMAX)
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5

    city_cmap = plt.get_cmap(CITY_CMAP).copy()
    city_cmap.set_bad(alpha=0.0)
    green_cmap = rural_green_cmap()

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=FIG_DPI)

    ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(rural_field),
        shading=PCOLORMESH_SHADING,
        cmap=green_cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=RURAL_ALPHA,
        zorder=1,
    )

    im = ax.pcolormesh(
        lon,
        lat,
        np.ma.masked_invalid(city_local),
        shading=PCOLORMESH_SHADING,
        cmap=city_cmap,
        norm=Normalize(vmin=vmin, vmax=vmax),
        alpha=CITY_ALPHA,
        zorder=5,
    )

    # Plot the true GHSL polygon boundary, not a rasterized proxy.
    gpd.GeoSeries([polygon], crs="EPSG:4326").boundary.plot(
        ax=ax, color="white", linewidth=2.0, zorder=8
    )
    gpd.GeoSeries([polygon], crs="EPSG:4326").boundary.plot(
        ax=ax, color="0.20", linewidth=0.55, zorder=9
    )

    ax.set_title(f"UHI over {city_name}", fontsize=24, weight="bold", pad=14)
    ax.set_xlabel("Longitude", fontsize=17)
    ax.set_ylabel("Latitude", fontsize=17)
    ax.tick_params(axis="both", labelsize=13, width=1.2, length=5)
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.20, color="black", zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.0%", pad=0.20)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Raw UHI (°C)", fontsize=17)
    cbar.ax.tick_params(labelsize=13)

    rep = polygon.representative_point()
    ax.text(
        rep.x, rep.y, "City", color="darkred", fontsize=16, weight="bold", ha="center", va="center", zorder=30,
        bbox=dict(facecolor="white", edgecolor="darkred", linewidth=1.5, boxstyle="round,pad=0.3", alpha=0.94),
    )

    ry, rx = np.where(rural_local)
    if ry.size:
        idx = max(0, ry.size // 6)
        ax.text(
            float(lon[ry[idx], rx[idx]]), float(lat[ry[idx], rx[idx]]), "rural reference",
            color="darkgreen", fontsize=13, weight="bold", ha="center", va="center", zorder=30,
            bbox=dict(facecolor="white", edgecolor="darkgreen", linewidth=1.3, boxstyle="round,pad=0.3", alpha=0.90),
        )

    legend_items = [
        Patch(facecolor="white", edgecolor="0.25", label="GHSL city boundary"),
        Patch(facecolor="#2e7d32", edgecolor="0.25", label="MOD_Mask rural reference"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=10, frameon=True, framealpha=0.88,
              facecolor="white", edgecolor="0.4")

    mode = metadata.get("rural_mode", RURAL_MODE)
    if mode == "exact":
        detail = (
            f"Rural mode: exact MOD_Mask\n"
            f"City urban cells: {metadata.get('urban_cells_in_city', 'NA')}\n"
            f"Accepted references: {metadata.get('accepted_urban_references', 'NA')}\n"
            f"Time index: {TIME_INDEX}\nNo UHI normalization"
        )
    else:
        detail = f"Rural mode: radius-box fallback\nR = {metadata.get('R', 'NA')} grid cells\nTime index: {TIME_INDEX}\nNo UHI normalization"

    ax.text(
        0.985, 0.025, detail, transform=ax.transAxes, fontsize=9.5, ha="right", va="bottom", zorder=30,
        bbox=dict(facecolor="white", edgecolor="0.25", linewidth=0.8, boxstyle="round,pad=0.35", alpha=0.90),
    )

    add_scalebar(ax, length_km=20)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


# ============================================================
# MAIN CITY PROCESSING
# ============================================================
def process_city(city_row, ds, lon2d, lat2d):
    city_name = city_row["__city_name__"]
    polygon = city_row.geometry
    print("\n" + "=" * 72)
    print(f"Processing city: {city_name}")
    print("=" * 72)

    city_polygon_mask = grid_points_inside_polygon(lon2d, lat2d, polygon)
    if not np.any(city_polygon_mask):
        print(f"Warning: {city_name} has no overlapping grid cells. Skipping.")
        return

    raw_uhi = get_raw_uhi_2d(ds)
    city_overlay = np.full(raw_uhi.shape, np.nan, float)
    # Preserve only actual raw model UHI values; missing model cells are not interpolated/fabricated.
    city_overlay[city_polygon_mask] = raw_uhi[city_polygon_mask]

    if RURAL_MODE.lower() == "exact":
        rural_mask, reference_count, model_urban_in_city, metadata = reconstruct_exact_rural_reference(
            ds, city_polygon_mask, PGD_FILE
        )
    elif RURAL_MODE.lower() == "radius_box":
        R = estimate_search_radius_from_nbg(ds, city_polygon_mask)
        rural_mask, reference_count = create_radius_box_rural_mask(lon2d, lat2d, city_polygon_mask, R)
        model_urban_in_city = city_polygon_mask & np.isfinite(raw_uhi)
        metadata = {"rural_mode": "radius_box", "R": int(R)}
    else:
        raise ValueError("RURAL_MODE must be 'exact' or 'radius_box'")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NC_DIR.mkdir(parents=True, exist_ok=True)

    out_ds = xr.Dataset(
        {
            "city_polygon_mask": (("y", "x"), city_polygon_mask.astype(np.int8)),
            "model_urban_in_city": (("y", "x"), model_urban_in_city.astype(np.int8)),
            f"{UHI_VAR}_raw_city_overlay": (("y", "x"), city_overlay.astype(np.float32)),
            "raw_uhi": (("y", "x"), raw_uhi.astype(np.float32)),
            "rural_reference_mask": (("y", "x"), rural_mask.astype(np.int8)),
            "rural_reference_count": (("y", "x"), reference_count.astype(np.int32)),
        },
        coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d)},
        attrs={
            "city": city_name,
            "method_version": "2.0.1",
            "uhi_variable": UHI_VAR,
            "time_index": int(TIME_INDEX),
            "rural_mode": metadata.get("rural_mode", RURAL_MODE),
            "note": "Raw UHI values are preserved; no percentile normalization or missing-value interpolation is written.",
            **{k: v for k, v in metadata.items() if k != "rural_mode"},
        },
    )

    nc_path = NC_DIR / f"raw_city_rural_overlay_v2_{city_name}.nc"
    fig_path = FIG_DIR / f"RAW_UHI_overlay_v2_{city_name}.png"
    out_ds.to_netcdf(nc_path)
    out_ds.close()

    plot_city_and_rural_overlay_raw(
        lon2d=lon2d,
        lat2d=lat2d,
        city_overlay=city_overlay,
        city_polygon_mask=city_polygon_mask,
        rural_mask=rural_mask,
        reference_count=reference_count,
        polygon=polygon,
        city_name=city_name,
        metadata=metadata,
        save_path=fig_path,
    )

    print(f"Saved figure: {fig_path}")
    print(f"Saved NetCDF: {nc_path}")


def _configure_from_cli(argv=None):
    global UHI_DIR, PGD_FILE, CITY_GPKG, TARGET_CITIES, CITY_NAME_COLUMN
    global CITY_GPKG_LAYER, UHI_VAR, TIME_INDEX, RURAL_MODE, OUT_DIR, FIG_DIR, NC_DIR
    parser = argparse.ArgumentParser(
        description="MOD_Mask v2 publication-quality GHSL city/rural/UHI overlay",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uhi-dir", type=Path, default=UHI_DIR)
    parser.add_argument("--pgd", type=Path, default=PGD_FILE)
    parser.add_argument("--city-gpkg", type=Path, default=CITY_GPKG)
    parser.add_argument("--cities", nargs="+", default=TARGET_CITIES)
    parser.add_argument("--city-name-column", default=CITY_NAME_COLUMN)
    parser.add_argument("--city-layer", default=CITY_GPKG_LAYER)
    parser.add_argument("--uhi-var", default=UHI_VAR)
    parser.add_argument("--time-index", type=int, default=TIME_INDEX)
    parser.add_argument("--rural-mode", choices=["exact", "radius_box"], default=RURAL_MODE)
    parser.add_argument("--output", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    UHI_DIR = args.uhi_dir
    PGD_FILE = args.pgd
    CITY_GPKG = args.city_gpkg
    TARGET_CITIES = list(args.cities)
    CITY_NAME_COLUMN = args.city_name_column
    CITY_GPKG_LAYER = args.city_layer or None
    UHI_VAR = args.uhi_var
    TIME_INDEX = args.time_index
    RURAL_MODE = args.rural_mode
    OUT_DIR = args.output
    FIG_DIR = OUT_DIR / "figures"
    NC_DIR = OUT_DIR / "netcdf"


def main(argv=None):
    _configure_from_cli(argv)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    NC_DIR.mkdir(parents=True, exist_ok=True)

    ds, opened_file = open_first_uhi_file(UHI_DIR)
    try:
        if UHI_VAR not in ds:
            raise KeyError(f"{UHI_VAR} not found. Data variables={list(ds.data_vars)}")

        lon_name, lat_name = get_lon_lat_names(ds)
        lon2d = ds[lon_name].values.astype(float)
        lat2d = ds[lat_name].values.astype(float)
        if lon2d.ndim != 2 or lat2d.ndim != 2:
            raise ValueError("Expected 2-D lon/lat arrays")

        cities_gdf = load_city_polygons(CITY_GPKG, TARGET_CITIES, CITY_NAME_COLUMN, CITY_GPKG_LAYER)
        for _, city_row in tqdm(cities_gdf.iterrows(), total=len(cities_gdf), desc="Processing cities"):
            try:
                process_city(city_row, ds, lon2d, lat2d)
            except Exception as exc:
                city_name = city_row.get("__city_name__", "unknown")
                print(f"Error processing {city_name}: {type(exc).__name__}: {exc}")
    finally:
        ds.close()

    print("\nDone.")
    print(f"Input file used: {opened_file}")
    print(f"Figures saved in: {FIG_DIR}")
    print(f"NetCDF files saved in: {NC_DIR}")


if __name__ == "__main__":
    main()
