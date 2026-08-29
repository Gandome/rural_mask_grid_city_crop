"""Compatibility utilities for older scripts.

Version 2 moved the core search/calculation into ``rural_reference.py`` and
``calculation.py``. The small helpers retained here avoid breaking historical
imports from the original repository.
"""
from __future__ import annotations

import numpy as np

from .calculation import compute_uhi_timeseries
from .rural_reference import find_rural_reference_once, precompute_rural_references


def uhi_formula(urban_temp, rural_temp):
    return urban_temp - rural_temp


def safe_nanmean(arr):
    arr = np.asarray(arr)
    return np.nan if arr.size == 0 or np.isnan(arr).all() else float(np.nanmean(arr))


def kelvin_humidity_convert(data, var_name=None):
    """Historical helper retained as a no-op except for obvious Kelvin temperature."""
    arr = np.asarray(data)
    if var_name and str(var_name).lower() in {"tas", "temperature", "t2m"}:
        return arr - 273.15 if np.nanmedian(arr) > 150 else arr
    return arr


def format_units(units):
    return units


def calculate_uhi(
    temperature,
    Elevation,
    rural_threshold,
    urban_threshold,
    rural_frac,
    urban_frac,
    urban_grid_points,
    Min_Value,
    nbg,
    max_iterations,
    nO,
    height_lim1,
    height_lim2,
    height_lim3,
    height_lim4,
    sea_water_mask=None,
):
    """Backward-compatible single-time-step wrapper around the v2 algorithm.

    New code should call ``precompute_rural_references`` once and
    ``compute_uhi_timeseries`` for all times instead.
    """
    temp = np.asarray(temperature, dtype=np.float32)
    if temp.ndim != 2:
        raise ValueError("Compatibility calculate_uhi expects a 2-D temperature field")
    sw = np.zeros_like(temp, dtype=bool) if sea_water_mask is None else np.asarray(sea_water_mask, bool)

    refs = precompute_rural_references(
        elevation=Elevation,
        rural_frac=rural_frac,
        urban_frac=urban_frac,
        sea_water_mask=sw,
        urban_grid_points=urban_grid_points,
        rural_threshold=rural_threshold,
        urban_threshold=urban_threshold,
        min_value=Min_Value,
        nbg=nbg,
        max_iterations=max_iterations,
        nO=nO,
    )
    out = compute_uhi_timeseries(
        temp[None, :, :],
        Elevation,
        refs,
        height_limits=(height_lim1, height_lim2, height_lim3, height_lim4),
    )

    def first(arr):
        return arr[0]

    ratio = np.full_like(temp, np.nan, dtype=np.float32)
    nbg_used = np.full_like(temp, np.nan, dtype=np.float32)
    for (y, x), ref in refs.items():
        ratio[y, x] = ref.ratio
        nbg_used[y, x] = ref.nbg

    return (
        first(out["UHI_px"]),
        first(out["UHI_LR"][float(height_lim1)]),
        first(out["UHI_LR"][float(height_lim2)]),
        first(out["UHI_LR"][float(height_lim3)]),
        first(out["UHI_LR"][float(height_lim4)]),
        first(out["rural_temperature_mean"]),
        first(out["rural_temperature_LR_mean"][float(height_lim1)]),
        first(out["rural_temperature_LR_mean"][float(height_lim2)]),
        first(out["rural_temperature_LR_mean"][float(height_lim3)]),
        first(out["rural_temperature_LR_mean"][float(height_lim4)]),
        ratio,
        nbg_used,
    )


__all__ = [
    "calculate_uhi",
    "compute_uhi_timeseries",
    "find_rural_reference_once",
    "format_units",
    "kelvin_humidity_convert",
    "precompute_rural_references",
    "safe_nanmean",
    "uhi_formula",
]
