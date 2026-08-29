"""Static land-cover masks used by MOD_Mask/UHI version 2."""
from __future__ import annotations

import xarray as xr


def to_yx(da: xr.DataArray) -> xr.DataArray:
    """Rename common AROME PGD dimensions to ``(y, x)`` and transpose."""
    rename = {}
    if "Y" in da.dims:
        rename["Y"] = "y"
    if "X" in da.dims:
        rename["X"] = "x"
    if rename:
        da = da.rename(rename)
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"Expected horizontal dimensions y/x (or Y/X), got {da.dims}")
    return da.transpose("y", "x")


def build_masks(
    town_fract: xr.DataArray,
    nature_fract: xr.DataArray,
    sea_fract: xr.DataArray,
    water_fract: xr.DataArray,
    urban_threshold: float,
    rural_threshold: float,
    sea_water_threshold: float,
):
    """Build mutually consistent static masks.

    Version 2 applies the sea/water exclusion first. Urban and rural masks are
    then defined only over land. Sea and inland-water fractions are summed,
    matching the revised MOD_Mask formulation.
    """
    town = to_yx(town_fract)
    nature = to_yx(nature_fract)
    sea = to_yx(sea_fract)
    water = to_yx(water_fract)

    sea_water_mask = (sea + water) > sea_water_threshold
    land_mask = ~sea_water_mask
    urban_mask = (town > urban_threshold) & land_mask
    rural_mask = (nature > rural_threshold) & land_mask
    return sea_water_mask, land_mask, urban_mask, rural_mask


def classify_grid_points(args):
    """Backward-compatible categorical classifier.

    Parameters are passed as the historical tuple
    ``(town, nature, sea, water, urban_thr, rural_thr, sea_water_thr)``.
    Classification precedence in v2 is sea/water exclusion first, then urban,
    then rural. Land cells meeting neither threshold are labelled ``other``.
    """
    (
        town_fract,
        nature_fract,
        sea_fract,
        water_fract,
        urban_threshold,
        rural_threshold,
        sea_water_threshold,
    ) = args

    sea_water_mask, land_mask, urban_mask, rural_mask = build_masks(
        town_fract,
        nature_fract,
        sea_fract,
        water_fract,
        urban_threshold,
        rural_threshold,
        sea_water_threshold,
    )

    # Preserve the older distinction where possible for display purposes.
    sea = to_yx(sea_fract)
    water = to_yx(water_fract)
    water_dominant = sea_water_mask & (water >= sea)

    out = xr.full_like(to_yx(town_fract), fill_value="other", dtype=object)
    out = xr.where(sea_water_mask & ~water_dominant, "sea", out)
    out = xr.where(water_dominant, "water", out)
    out = xr.where(rural_mask & land_mask, "rural", out)
    out = xr.where(urban_mask & land_mask, "urban", out)
    out.name = "grid_classification"
    return out
