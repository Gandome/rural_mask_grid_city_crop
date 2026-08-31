"""Horizontal-grid validation and optional contiguous-subdomain matching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr

from .io_utils import standardize_tas
from .urban_mask import to_yx

_COORD_LON = ("lon", "longitude", "LON", "XLONG")
_COORD_LAT = ("lat", "latitude", "LAT", "XLAT")


@dataclass(frozen=True)
class GridAlignment:
    """Description of how a tas grid maps onto the PGD grid."""

    yslice: slice
    xslice: slice
    source_shape: Tuple[int, int]
    target_shape: Tuple[int, int]
    method: str
    max_lon_error_deg: float = np.nan
    max_lat_error_deg: float = np.nan

    @property
    def is_full_grid(self) -> bool:
        return (
            self.yslice.start in (None, 0)
            and self.xslice.start in (None, 0)
            and self.yslice.stop == self.source_shape[0]
            and self.xslice.stop == self.source_shape[1]
        )


def _find_2d_coord(ds: xr.Dataset, candidates):
    for name in candidates:
        if name not in ds:
            continue
        da = ds[name].squeeze(drop=True)
        rename = {}
        if "Y" in da.dims:
            rename["Y"] = "y"
        if "X" in da.dims:
            rename["X"] = "x"
        if rename:
            da = da.rename(rename)
        if set(("y", "x")).issubset(da.dims) and da.ndim == 2:
            return da.transpose("y", "x")
    return None


def get_pgd_lon_lat(pgd: xr.Dataset):
    lon = _find_2d_coord(pgd, _COORD_LON)
    lat = _find_2d_coord(pgd, _COORD_LAT)
    if lon is None or lat is None:
        return None, None
    return np.asarray(lon.values, float), np.asarray(lat.values, float)


def get_tas_lon_lat(ds: xr.Dataset, tas: xr.DataArray):
    lon = next((tas.coords[n] for n in _COORD_LON if n in tas.coords), None)
    lat = next((tas.coords[n] for n in _COORD_LAT if n in tas.coords), None)

    if lon is None:
        lon = _find_2d_coord(ds, _COORD_LON)
    if lat is None:
        lat = _find_2d_coord(ds, _COORD_LAT)
    if lon is None or lat is None:
        return None, None

    lonv = np.asarray(lon.values, float)
    latv = np.asarray(lat.values, float)
    if lonv.ndim != 2 or latv.ndim != 2:
        return None, None
    return lonv, latv


def _nearest_index(lon2d, lat2d, lon0, lat0):
    valid = np.isfinite(lon2d) & np.isfinite(lat2d)
    if not np.any(valid):
        raise ValueError("tas longitude/latitude grid contains no finite points")
    # Longitude scaling by cos(latitude) gives a more isotropic local metric.
    scale = np.cos(np.deg2rad(float(lat0)))
    d2 = ((lon2d - lon0) * scale) ** 2 + (lat2d - lat0) ** 2
    d2 = np.where(valid, d2, np.inf)
    return np.unravel_index(int(np.argmin(d2)), d2.shape)


def determine_grid_alignment(
    pgd_file,
    tas_file,
    coord_tolerance_deg: float = 1.0e-4,
) -> GridAlignment:
    """Validate PGD/tas grids and find a matching contiguous tas crop if needed.

    If the horizontal shapes are identical, the full tas grid is used only after
    the 2-D lon/lat coordinates agree within ``coord_tolerance_deg``.  The v2.0.1
    default (1e-4 degree) is intentionally small relative to kilometre-scale
    regional-climate grids while allowing harmless coordinate rounding at about
    the 1e-5--5e-5 degree level.  It is a validation tolerance only: no spatial
    regridding, interpolation, or coordinate replacement is performed.

    When the PGD is a rectangular subset of the tas grid and both datasets expose
    2-D lon/lat coordinates, the corresponding contiguous tas slice is identified
    from the PGD corner coordinates and verified against every PGD coordinate.
    """
    pgd_file = Path(pgd_file)
    tas_file = Path(tas_file)

    with xr.open_dataset(pgd_file) as pgd:
        elev = to_yx(pgd["SFX.ZS"])
        target_shape = tuple(map(int, elev.shape))
        pgd_lon, pgd_lat = get_pgd_lon_lat(pgd)

    with xr.open_dataset(tas_file) as ds:
        if "tas" not in ds:
            raise KeyError(f"tas not found in {tas_file}")
        tas = standardize_tas(ds["tas"])
        source_shape = (int(tas.sizes["y"]), int(tas.sizes["x"]))
        tas_lon, tas_lat = get_tas_lon_lat(ds, tas)

    if source_shape == target_shape:
        max_lon = max_lat = np.nan
        method = "same_shape"
        if pgd_lon is not None and tas_lon is not None:
            max_lon = float(np.nanmax(np.abs(tas_lon - pgd_lon)))
            max_lat = float(np.nanmax(np.abs(tas_lat - pgd_lat)))
            if max(max_lon, max_lat) > coord_tolerance_deg:
                raise ValueError(
                    "PGD and tas have the same shape but their lon/lat grids do not match: "
                    f"max_lon_error={max_lon:.6g}°, max_lat_error={max_lat:.6g}° "
                    f"> tolerance={coord_tolerance_deg:.6g}°."
                )
            method = "same_shape_coordinates_verified"
        return GridAlignment(
            slice(0, source_shape[0]),
            slice(0, source_shape[1]),
            source_shape,
            target_shape,
            method,
            max_lon,
            max_lat,
        )

    if pgd_lon is None or pgd_lat is None or tas_lon is None or tas_lat is None:
        raise ValueError(
            f"PGD grid {target_shape} differs from tas grid {source_shape}, and 2-D lon/lat "
            "coordinates are unavailable to identify a safe contiguous subdomain."
        )

    ny, nx = target_shape
    corners = [
        _nearest_index(tas_lon, tas_lat, pgd_lon[0, 0], pgd_lat[0, 0]),
        _nearest_index(tas_lon, tas_lat, pgd_lon[0, -1], pgd_lat[0, -1]),
        _nearest_index(tas_lon, tas_lat, pgd_lon[-1, 0], pgd_lat[-1, 0]),
        _nearest_index(tas_lon, tas_lat, pgd_lon[-1, -1], pgd_lat[-1, -1]),
    ]
    ys = [p[0] for p in corners]
    xs = [p[1] for p in corners]
    y0, y1 = min(ys), max(ys) + 1
    x0, x1 = min(xs), max(xs) + 1

    if (y1 - y0, x1 - x0) != target_shape:
        # A second deterministic attempt starts from the closest PGD top-left
        # point and uses the known target shape. This handles nearly identical
        # corners whose nearest-cell search can differ by one cell.
        iy, ix = corners[0]
        candidates = []
        for sy in (1, -1):
            for sx in (1, -1):
                ya = iy if sy == 1 else iy - ny + 1
                xa = ix if sx == 1 else ix - nx + 1
                yb, xb = ya + ny, xa + nx
                if 0 <= ya < yb <= source_shape[0] and 0 <= xa < xb <= source_shape[1]:
                    candidates.append((slice(ya, yb), slice(xa, xb)))
    else:
        candidates = [(slice(y0, y1), slice(x0, x1))]

    best = None
    for ysli, xsli in candidates:
        lon_sub = tas_lon[ysli, xsli]
        lat_sub = tas_lat[ysli, xsli]
        if lon_sub.shape != target_shape:
            continue
        max_lon = float(np.nanmax(np.abs(lon_sub - pgd_lon)))
        max_lat = float(np.nanmax(np.abs(lat_sub - pgd_lat)))
        score = max(max_lon, max_lat)
        if best is None or score < best[0]:
            best = (score, ysli, xsli, max_lon, max_lat)

    if best is None or best[0] > coord_tolerance_deg:
        detail = "no candidate contiguous slice" if best is None else (
            f"best coordinate error={best[0]:.6g}°"
        )
        raise ValueError(
            f"Could not prove that PGD grid {target_shape} is a contiguous subset of tas grid "
            f"{source_shape}: {detail}; tolerance={coord_tolerance_deg:.6g}°."
        )

    _, ysli, xsli, max_lon, max_lat = best
    return GridAlignment(
        ysli,
        xsli,
        source_shape,
        target_shape,
        "contiguous_lonlat_match",
        max_lon,
        max_lat,
    )
