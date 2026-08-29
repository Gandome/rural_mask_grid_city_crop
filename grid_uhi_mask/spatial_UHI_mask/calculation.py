"""Vectorized-in-time UHI calculation using precomputed static references."""
from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import numpy as np

from .rural_reference import RuralReference


def compute_uhi_timeseries(
    tas_c: np.ndarray,
    elevation: np.ndarray,
    references: Mapping[Tuple[int, int], RuralReference],
    height_limits: Iterable[float] = (100.0, 200.0, 300.0, 500.0),
    lapse_rate: float = 0.0065,
):
    """Compute UHI and rural-reference temperatures for all times.

    Parameters
    ----------
    tas_c
        Temperature in degC with shape ``(time, y, x)``.
    elevation
        Static terrain elevation in m with shape ``(y, x)``.
    references
        Static rural references returned by :func:`precompute_rural_references`.
    height_limits
        Absolute urban-rural elevation-difference thresholds (m).
    lapse_rate
        Positive environmental lapse rate in K m-1. Rural temperatures are
        adjusted to the urban-cell elevation as
        ``T_r,adj = mean(T_r) - lapse_rate * mean(z_u - z_r)``.
    """
    tas = np.asarray(tas_c, dtype=np.float32)
    elev = np.asarray(elevation, dtype=np.float32)
    if tas.ndim != 3:
        raise ValueError(f"tas_c must be 3-D (time,y,x), got {tas.shape}")
    if elev.shape != tas.shape[1:]:
        raise ValueError(f"Elevation shape {elev.shape} != tas horizontal shape {tas.shape[1:]}")

    limits = tuple(float(v) for v in height_limits)
    nt, ny, nx = tas.shape

    uhi_px = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    rural_px = np.full((nt, ny, nx), np.nan, dtype=np.float32)
    uhi_lr = {lim: np.full((nt, ny, nx), np.nan, dtype=np.float32) for lim in limits}
    rural_lr = {lim: np.full((nt, ny, nx), np.nan, dtype=np.float32) for lim in limits}

    for (y, x), ref in references.items():
        if not ref.valid:
            continue

        yy, xx = ref.yy, ref.xx
        urban_ts = tas[:, y, x]
        rural_all = tas[:, yy, xx]
        rural_mean = np.nanmean(rural_all, axis=1).astype(np.float32)
        uhi_px[:, y, x] = urban_ts - rural_mean
        rural_px[:, y, x] = rural_mean

        delta_elev = elev[y, x] - elev[yy, xx]
        for lim in limits:
            valid = np.isfinite(delta_elev) & (np.abs(delta_elev) <= lim)
            if not np.any(valid):
                continue

            rural_valid = tas[:, yy[valid], xx[valid]]
            rural_valid_mean = np.nanmean(rural_valid, axis=1).astype(np.float32)
            mean_delta = float(np.nanmean(delta_elev[valid]))
            corrected = rural_valid_mean - np.float32(lapse_rate * mean_delta)
            rural_lr[lim][:, y, x] = corrected
            uhi_lr[lim][:, y, x] = urban_ts - corrected

    return {
        "UHI_px": uhi_px,
        "rural_temperature_mean": rural_px,
        "UHI_LR": uhi_lr,
        "rural_temperature_LR_mean": rural_lr,
    }
