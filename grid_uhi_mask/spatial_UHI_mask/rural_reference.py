"""Adaptive static rural-reference search for MOD_Mask version 2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class RuralReference:
    """Accepted static rural reference for one urban grid cell."""

    y: int
    x: int
    yy: Optional[np.ndarray]
    xx: Optional[np.ndarray]
    ratio: float
    nbg: float
    min_value: float
    n_total: int
    n_rural: int

    @property
    def valid(self) -> bool:
        return self.yy is not None and self.xx is not None and self.n_rural > 0


def find_rural_reference_once(
    elevation: np.ndarray,
    rural_frac: np.ndarray,
    urban_frac: np.ndarray,
    sea_water_mask: np.ndarray,
    urban_point: Tuple[int, int],
    rural_threshold: float,
    urban_threshold: float,
    min_value: float,
    nbg: int,
    max_iterations: int,
    nO: int,
    min_ratio_floor: float = 50.0,
    ratio_step: float = 5.0,
) -> RuralReference:
    """Find the rural reference cells once for a static urban grid point.

    Search logic:
      1. Start at the requested ratio threshold (typically 70%).
      2. Expand the square outer radius from ``nbg`` for at most
         ``max_iterations`` tested radii.
      3. Exclude the central square of half-width ``nO``.
      4. Exclude sea/water before computing the rural/land-cell ratio.
      5. If no radius satisfies the ratio, lower the requested ratio by 5
         percentage points, but never below 50%.
      6. If the 50% criterion still cannot be achieved, return an invalid
         reference; the corresponding UHI cell remains NaN.
    """
    del elevation  # retained in signature for API/method symmetry and future QC

    y, x = map(int, urban_point)
    ny, nx = rural_frac.shape
    current_min_value = float(min_value)

    final_ratio = np.nan
    final_nbg = np.nan
    final_min_value = np.nan
    final_n_total = 0
    final_n_rural = 0

    while current_min_value >= min_ratio_floor - 1e-12:
        current_nbg = int(nbg)

        for _ in range(int(max_iterations)):
            y_min = max(0, y - current_nbg)
            y_max = min(ny, y + current_nbg + 1)
            x_min = max(0, x - current_nbg)
            x_max = min(nx, x + current_nbg + 1)

            sub_rural = rural_frac[y_min:y_max, x_min:x_max]
            sub_urban = urban_frac[y_min:y_max, x_min:x_max]
            sub_sea_water = sea_water_mask[y_min:y_max, x_min:x_max]

            total_mask = ~sub_sea_water.copy()

            cy = y - y_min
            cx = x - x_min
            iy0 = max(0, cy - nO)
            iy1 = min(total_mask.shape[0], cy + nO + 1)
            ix0 = max(0, cx - nO)
            ix1 = min(total_mask.shape[1], cx + nO + 1)
            total_mask[iy0:iy1, ix0:ix1] = False

            rural_reference_mask = (
                total_mask
                & np.isfinite(sub_rural)
                & np.isfinite(sub_urban)
                & (sub_rural >= rural_threshold)
                & (sub_urban <= urban_threshold)
            )

            n_total = int(np.count_nonzero(total_mask))
            n_rural = int(np.count_nonzero(rural_reference_mask))
            ratio = 100.0 * n_rural / n_total if n_total > 0 else 0.0

            final_ratio = float(ratio)
            final_nbg = float(current_nbg)
            final_min_value = float(current_min_value)
            final_n_total = n_total
            final_n_rural = n_rural

            if ratio >= current_min_value and n_rural > 0:
                yy, xx = np.where(rural_reference_mask)
                yy = (yy + y_min).astype(np.int32)
                xx = (xx + x_min).astype(np.int32)
                return RuralReference(
                    y=y,
                    x=x,
                    yy=yy,
                    xx=xx,
                    ratio=ratio,
                    nbg=float(current_nbg),
                    min_value=float(current_min_value),
                    n_total=n_total,
                    n_rural=n_rural,
                )

            current_nbg += 1

        current_min_value -= ratio_step

    return RuralReference(
        y=y,
        x=x,
        yy=None,
        xx=None,
        ratio=float(final_ratio),
        nbg=float(final_nbg),
        min_value=float(final_min_value),
        n_total=int(final_n_total),
        n_rural=int(final_n_rural),
    )


def precompute_rural_references(
    elevation,
    rural_frac,
    urban_frac,
    sea_water_mask,
    urban_grid_points: Iterable[Tuple[int, int]],
    rural_threshold: float,
    urban_threshold: float,
    min_value: float,
    nbg: int,
    max_iterations: int,
    nO: int,
    min_ratio_floor: float = 50.0,
    ratio_step: float = 5.0,
    progress_every: int = 0,
) -> Dict[Tuple[int, int], RuralReference]:
    """Precompute static rural references once per experiment.

    This is the principal v2 performance change: the reference geometry no
    longer needs to be rediscovered for every time step or every input file.
    """
    elev = np.asarray(elevation)
    rural = np.asarray(rural_frac)
    urban = np.asarray(urban_frac)
    sea_water = np.asarray(sea_water_mask, dtype=bool)
    points = [tuple(map(int, p)) for p in np.asarray(list(urban_grid_points))]

    refs: Dict[Tuple[int, int], RuralReference] = {}
    n = len(points)
    for i, point in enumerate(points, start=1):
        ref = find_rural_reference_once(
            elevation=elev,
            rural_frac=rural,
            urban_frac=urban,
            sea_water_mask=sea_water,
            urban_point=point,
            rural_threshold=rural_threshold,
            urban_threshold=urban_threshold,
            min_value=min_value,
            nbg=nbg,
            max_iterations=max_iterations,
            nO=nO,
            min_ratio_floor=min_ratio_floor,
            ratio_step=ratio_step,
        )
        refs[point] = ref
        if progress_every and (i % progress_every == 0 or i == n):
            print(f"Precomputed rural references: {i}/{n}", flush=True)
    return refs


def diagnostics_from_references(references, shape, sea_water_mask=None) -> xr.Dataset:
    """Convert reference metadata to static 2-D diagnostic fields."""
    ny, nx = shape
    ratio = np.full((ny, nx), np.nan, np.float32)
    nbg = np.full((ny, nx), np.nan, np.float32)
    min_used = np.full((ny, nx), np.nan, np.float32)
    n_total = np.full((ny, nx), np.nan, np.float32)
    n_rural = np.full((ny, nx), np.nan, np.float32)
    success = np.zeros((ny, nx), np.int8)
    footprint = np.zeros((ny, nx), np.int32)

    for (y, x), ref in references.items():
        ratio[y, x] = ref.ratio
        nbg[y, x] = ref.nbg
        min_used[y, x] = ref.min_value
        n_total[y, x] = ref.n_total
        n_rural[y, x] = ref.n_rural
        success[y, x] = int(ref.valid)
        if ref.valid:
            np.add.at(footprint, (ref.yy, ref.xx), 1)

    if sea_water_mask is not None:
        sw = np.asarray(sea_water_mask, dtype=bool)
        for arr in (ratio, nbg, min_used, n_total, n_rural):
            arr[sw] = np.nan
        success[sw] = 0
        footprint[sw] = 0

    return xr.Dataset(
        {
            "Ratio_used": (("y", "x"), ratio, {"units": "%", "long_name": "Accepted rural-to-candidate land-cell ratio"}),
            "Min_Value_used": (("y", "x"), min_used, {"units": "%", "long_name": "Minimum ratio threshold effectively accepted"}),
            "nbg": (("y", "x"), nbg, {"units": "grid cells", "long_name": "Accepted outer neighbourhood radius"}),
            "n_total_reference": (("y", "x"), n_total, {"units": "grid cells", "long_name": "Number of candidate non-sea/water cells in accepted/last search window"}),
            "n_rural_reference": (("y", "x"), n_rural, {"units": "grid cells", "long_name": "Number of accepted rural reference cells"}),
            "rural_search_success": (("y", "x"), success, {"units": "1", "flag_values": [0, 1], "flag_meanings": "failed accepted"}),
            "rural_reference_frequency": (("y", "x"), footprint, {"units": "urban cells", "long_name": "Number of urban cells using each grid cell as a rural reference"}),
        }
    )
