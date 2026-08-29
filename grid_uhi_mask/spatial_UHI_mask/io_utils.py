"""I/O helpers for the v2 UHI workflow."""
from __future__ import annotations

import numpy as np
import xarray as xr


def standardize_tas(da: xr.DataArray) -> xr.DataArray:
    """Return temperature with dimensions exactly ``(time, y, x)``."""
    # Remove only singleton auxiliary dimensions, never time.
    for dim in list(da.dims):
        if dim not in {"time", "y", "x", "Y", "X"}:
            if da.sizes[dim] != 1:
                raise ValueError(f"Unsupported non-singleton tas dimension {dim}={da.sizes[dim]}")
            da = da.isel({dim: 0}, drop=True)

    rename = {}
    if "Y" in da.dims:
        rename["Y"] = "y"
    if "X" in da.dims:
        rename["X"] = "x"
    if rename:
        da = da.rename(rename)

    if "time" not in da.dims:
        raise ValueError("tas variable has no time dimension")
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"Cannot identify y/x dimensions in tas: {da.dims}")
    return da.transpose("time", "y", "x")


def temperature_to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert Kelvin to Celsius while leaving Celsius inputs unchanged."""
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin", "degree_kelvin", "degrees_kelvin"}:
        out = da - 273.15
    elif units in {"c", "degc", "°c", "celsius", "degree_celsius", "degrees_celsius"}:
        out = da
    else:
        # Conservative numerical fallback for common atmospheric tas data.
        sample = np.asarray(da.isel(time=0).values)
        med = float(np.nanmedian(sample)) if np.isfinite(sample).any() else np.nan
        out = da - 273.15 if np.isfinite(med) and med > 150.0 else da
    out = out.copy()
    out.attrs.update(da.attrs)
    out.attrs["units"] = "degC"
    return out


def _standardize_2d_grid_da(da: xr.DataArray) -> xr.DataArray:
    da = da.squeeze(drop=True)
    rename = {}
    if "Y" in da.dims:
        rename["Y"] = "y"
    if "X" in da.dims:
        rename["X"] = "x"
    if rename:
        da = da.rename(rename)
    if da.ndim != 2 or "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"Expected a 2-D y/x grid, got dims={da.dims}")
    return da.transpose("y", "x")


def get_lon_lat_2d(ds: xr.Dataset, tas: xr.DataArray, spatial_slice=None):
    """Return 2-D lon/lat arrays aligned with the standardized ``tas`` grid.

    ``spatial_slice`` is only applied when lon/lat must be taken from dataset
    variables rather than from already-sliced ``tas`` coordinates.
    """
    candidates_lon = ["lon", "longitude", "LON", "XLONG"]
    candidates_lat = ["lat", "latitude", "LAT", "XLAT"]

    lon = next((tas.coords[n] for n in candidates_lon if n in tas.coords), None)
    lat = next((tas.coords[n] for n in candidates_lat if n in tas.coords), None)
    lon_from_tas = lon is not None
    lat_from_tas = lat is not None

    if lon is None:
        lon = next((ds[n] for n in candidates_lon if n in ds), None)
    if lat is None:
        lat = next((ds[n] for n in candidates_lat if n in ds), None)
    if lon is None or lat is None:
        raise KeyError("Could not find lon/lat variables")

    lon = _standardize_2d_grid_da(lon)
    lat = _standardize_2d_grid_da(lat)

    if spatial_slice is not None:
        ys, xs = spatial_slice
        if not lon_from_tas:
            lon = lon.isel(y=ys, x=xs)
        if not lat_from_tas:
            lat = lat.isel(y=ys, x=xs)

    lonv = np.asarray(lon.values)
    latv = np.asarray(lat.values)
    if lonv.ndim != 2 or latv.ndim != 2:
        raise ValueError("Version 2 currently expects 2-D curvilinear lon/lat arrays")
    if lonv.shape != tas.shape[-2:] or latv.shape != tas.shape[-2:]:
        raise ValueError(
            f"lon/lat shape mismatch after alignment: lon={lonv.shape}, lat={latv.shape}, "
            f"tas={tas.shape[-2:]}"
        )
    return lonv, latv
