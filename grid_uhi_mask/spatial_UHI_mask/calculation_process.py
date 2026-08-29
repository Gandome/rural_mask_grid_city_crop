"""Per-file UHI processing for MOD_Mask version 2."""
from __future__ import annotations

import datetime as _dt
import importlib.util
import os
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import xarray as xr

from .calculation import compute_uhi_timeseries
from .io_utils import get_lon_lat_2d, standardize_tas, temperature_to_celsius
from .rural_reference import RuralReference, diagnostics_from_references


def _log(msg: str):
    print(msg, flush=True)


def _mask_3d(arr: np.ndarray, sea_water_mask: np.ndarray) -> np.ndarray:
    return np.where(sea_water_mask[None, :, :], np.nan, arr).astype(np.float32)


def _copy_optional_grid_metadata(src: xr.Dataset, dst: xr.Dataset):
    """Copy useful horizontal-grid metadata when dimensions are compatible."""
    for name in ["lon_bnds", "lat_bnds", "rotated_pole", "Lambert_Conformal", "lambert_conformal"]:
        if name in src and name not in dst:
            try:
                dst[name] = src[name]
            except Exception:
                pass
    if "lon_bnds" in dst:
        dst["lon"].attrs["bounds"] = "lon_bnds"
    if "lat_bnds" in dst:
        dst["lat"].attrs["bounds"] = "lat_bnds"


def process_file(
    file_path,
    elevation,
    output_path,
    sea_water_mask,
    references: Mapping[Tuple[int, int], RuralReference],
    height_limits: Sequence[float] = (100, 200, 300, 500),
    lapse_rate: float = 0.0065,
    urban_threshold: float = 0.20,
    rural_threshold: float = 0.60,
    sea_water_threshold: float = 0.30,
    min_value_requested: float = 70.0,
    nO: int = 2,
    initial_nbg: int = 4,
    max_iterations: int = 26,
    min_ratio_floor: float = 50.0,
    ratio_step: float = 5.0,
    output_prefix: str = "Urban_Heat_Island_data_",
    spatial_slice=None,
):
    """Process one NetCDF file using static precomputed rural references."""
    path = Path(file_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = path.name

    _log("\n" + "-" * 90)
    _log(f"Processing file: {filename}")
    _log("-" * 90)

    try:
        with xr.open_dataset(path) as ds:
            if "tas" not in ds:
                raise KeyError(f"tas not found in {filename}; available={list(ds.data_vars)}")

            tas = standardize_tas(ds["tas"])
            if spatial_slice is not None:
                ys, xs = spatial_slice
                tas = tas.isel(y=ys, x=xs)
            tas = temperature_to_celsius(tas)
            tas_np = np.asarray(tas.values, dtype=np.float32)
            elev_np = np.asarray(elevation, dtype=np.float32)
            sw_np = np.asarray(sea_water_mask, dtype=bool)

            if tas_np.shape[1:] != elev_np.shape:
                raise ValueError(
                    f"tas horizontal shape {tas_np.shape[1:]} does not match PGD/elevation {elev_np.shape}"
                )
            if sw_np.shape != elev_np.shape:
                raise ValueError(f"sea/water mask shape {sw_np.shape} != elevation {elev_np.shape}")

            nt, ny, nx = tas_np.shape
            n_valid_refs = sum(int(ref.valid) for ref in references.values())
            _log(f"[{filename}] timesteps               : {nt}")
            _log(f"[{filename}] grid                    : {ny} x {nx}")
            _log(f"[{filename}] urban reference entries : {len(references)}")
            _log(f"[{filename}] accepted references      : {n_valid_refs}")

            results = compute_uhi_timeseries(
                tas_c=tas_np,
                elevation=elev_np,
                references=references,
                height_limits=height_limits,
                lapse_rate=lapse_rate,
            )

            lon, lat = get_lon_lat_2d(ds, tas, spatial_slice=spatial_slice)
            data_vars = {
                "UHI_px": (
                    ("time", "y", "x"),
                    _mask_3d(results["UHI_px"], sw_np),
                    {
                        "long_name": "Urban heat island without elevation filtering",
                        "units": "degC",
                    },
                ),
                "rural_temperature_mean": (
                    ("time", "y", "x"),
                    _mask_3d(results["rural_temperature_mean"], sw_np),
                    {
                        "long_name": "Mean rural reference temperature",
                        "units": "degC",
                    },
                ),
            }

            for lim in map(float, height_limits):
                tag = f"{int(lim) if float(lim).is_integer() else lim:g}"
                data_vars[f"UHI_LR{tag}"] = (
                    ("time", "y", "x"),
                    _mask_3d(results["UHI_LR"][lim], sw_np),
                    {
                        "long_name": f"Urban heat island with |urban-rural elevation difference| <= {tag} m",
                        "units": "degC",
                        "elevation_filter_m": float(lim),
                        "lapse_rate_K_m-1": float(lapse_rate),
                    },
                )
                data_vars[f"rural_temperature_LR{tag}_mean"] = (
                    ("time", "y", "x"),
                    _mask_3d(results["rural_temperature_LR_mean"][lim], sw_np),
                    {
                        "long_name": f"Elevation-filtered and lapse-rate-corrected rural reference temperature ({tag} m)",
                        "units": "degC",
                        "elevation_filter_m": float(lim),
                        "lapse_rate_K_m-1": float(lapse_rate),
                    },
                )

            diag = diagnostics_from_references(references, (ny, nx), sw_np)
            for name, da in diag.data_vars.items():
                data_vars[name] = (da.dims, da.values, dict(da.attrs))

            out_ds = xr.Dataset(
                data_vars=data_vars,
                coords={
                    "time": tas["time"],
                    "lon": (("y", "x"), lon),
                    "lat": (("y", "x"), lat),
                },
                attrs={
                    "title": "Urban Heat Island from MOD_Mask version 2",
                    "method_version": "2.0.0",
                    "creation_date": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "classification_order": (
                        "sea_water_mask=(sea_fraction+water_fraction)>sea_water_threshold is excluded first; "
                        "urban and rural masks are then defined over land only"
                    ),
                    "computation_strategy": (
                        "Static rural reference cells are identified once per experiment and reused for all input files; "
                        "all timesteps are then processed together for each urban grid cell"
                    ),
                    "search_window": (
                        "Square outer window (2*nbg+1)^2 with central square exclusion (2*nO+1)^2; "
                        "only non-sea/water candidate cells enter the denominator"
                    ),
                    "failed_search_rule": (
                        "Requested rural availability ratio is reduced in fixed steps down to the minimum floor; "
                        "if the floor is still unmet after the maximum search radius, the urban cell is invalid and UHI remains NaN"
                    ),
                    "elevation_correction": (
                        "For each LR threshold, rural cells are first filtered by |z_urban-z_rural| <= LR; "
                        "their mean temperature is then adjusted to urban elevation as T_r,adj=T_r-lapse_rate*mean(z_urban-z_rural)"
                    ),
                    "urban_threshold": float(urban_threshold),
                    "rural_threshold": float(rural_threshold),
                    "sea_water_threshold": float(sea_water_threshold),
                    "Min_Value_requested": float(min_value_requested),
                    "Min_Value_floor": float(min_ratio_floor),
                    "Min_Value_step": float(ratio_step),
                    "nO": int(nO),
                    "initial_nbg": int(initial_nbg),
                    "max_iterations": int(max_iterations),
                    "height_limits_m": ", ".join(f"{float(v):g}" for v in height_limits),
                    "lapse_rate_K_m-1": float(lapse_rate),
                    "input_file": filename,
                    "tas_spatial_slice": (
                        "full_grid" if spatial_slice is None else
                        f"y[{spatial_slice[0].start}:{spatial_slice[0].stop}],"
                        f"x[{spatial_slice[1].start}:{spatial_slice[1].stop}]"
                    ),
                },
            )
            _copy_optional_grid_metadata(ds, out_ds)

            # Preserve source time/global metadata that are safe and useful.
            for key in ["frequency", "institute_id", "model_id", "project_id", "domain", "nominal_resolution", "grid"]:
                if key in ds.attrs and key not in out_ds.attrs:
                    out_ds.attrs[key] = ds.attrs[key]

            # Compression is used when an HDF5-capable backend is available.
            # scipy.io.netcdf does not support zlib/complevel.
            if importlib.util.find_spec("netCDF4") is not None:
                engine = "netcdf4"
            elif importlib.util.find_spec("h5netcdf") is not None:
                engine = "h5netcdf"
            else:
                engine = "scipy"

            encoding = {}
            if engine != "scipy":
                float_comp = {"zlib": True, "complevel": 4, "dtype": "float32"}
                int_comp = {"zlib": True, "complevel": 4}
                for var in out_ds.data_vars:
                    if out_ds[var].ndim >= 1 and np.issubdtype(out_ds[var].dtype, np.floating):
                        encoding[var] = dict(float_comp)
                    elif out_ds[var].ndim >= 2 and np.issubdtype(out_ds[var].dtype, np.integer):
                        encoding[var] = dict(int_comp)

            suffix = filename[4:] if filename.startswith("tas_") else filename
            out_file = out_dir / f"{output_prefix}{suffix}"
            _log(f"[{filename}] writing output ({engine} backend): {out_file}")
            out_ds.to_netcdf(out_file, encoding=encoding, engine=engine)
            out_ds.close()

        _log(f"[{filename}] saved successfully.")
        return str(out_file)

    except Exception as exc:
        _log(f"ERROR processing {file_path}: {type(exc).__name__}: {exc}")
        raise
