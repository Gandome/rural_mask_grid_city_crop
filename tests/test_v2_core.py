from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grid_uhi_mask"))

from spatial_UHI_mask.calculation import compute_uhi_timeseries
from spatial_UHI_mask.rural_reference import RuralReference, find_rural_reference_once


def test_adaptive_search_accepts_static_rural_ring():
    shape = (9, 9)
    elevation = np.zeros(shape, dtype=float)
    rural = np.ones(shape, dtype=float)
    urban = np.zeros(shape, dtype=float)
    urban[4, 4] = 1.0
    sea_water = np.zeros(shape, dtype=bool)

    ref = find_rural_reference_once(
        elevation=elevation,
        rural_frac=rural,
        urban_frac=urban,
        sea_water_mask=sea_water,
        urban_point=(4, 4),
        rural_threshold=0.60,
        urban_threshold=0.20,
        min_value=70,
        nbg=2,
        max_iterations=3,
        nO=1,
    )

    assert ref.valid
    assert ref.nbg == 2
    assert ref.min_value == 70
    assert ref.n_total == 16
    assert ref.n_rural == 16
    assert np.isclose(ref.ratio, 100.0)


def test_failed_search_stops_at_50_percent_floor():
    shape = (9, 9)
    ref = find_rural_reference_once(
        elevation=np.zeros(shape),
        rural_frac=np.zeros(shape),
        urban_frac=np.zeros(shape),
        sea_water_mask=np.zeros(shape, dtype=bool),
        urban_point=(4, 4),
        rural_threshold=0.60,
        urban_threshold=0.20,
        min_value=70,
        nbg=2,
        max_iterations=2,
        nO=1,
    )

    assert not ref.valid
    assert ref.min_value == 50
    assert ref.nbg == 3
    assert ref.n_rural == 0


def test_elevation_filter_uses_only_eligible_rural_cells_and_corrects_lapse_rate():
    tas = np.full((2, 3, 3), np.nan, dtype=np.float32)
    tas[:, 1, 1] = [30.0, 31.0]       # urban
    tas[:, 0, 0] = [20.0, 22.0]       # rural, dz = 100 m
    tas[:, 0, 1] = [22.0, 24.0]       # rural, dz = 50 m

    elevation = np.zeros((3, 3), dtype=np.float32)
    elevation[1, 1] = 100.0
    elevation[0, 0] = 0.0
    elevation[0, 1] = 50.0

    ref = RuralReference(
        y=1,
        x=1,
        yy=np.array([0, 0], dtype=np.int32),
        xx=np.array([0, 1], dtype=np.int32),
        ratio=100.0,
        nbg=2.0,
        min_value=70.0,
        n_total=2,
        n_rural=2,
    )

    out = compute_uhi_timeseries(
        tas,
        elevation,
        {(1, 1): ref},
        height_limits=(40, 60, 120),
        lapse_rate=0.0065,
    )

    # Unfiltered rural means are [21, 23].
    assert np.allclose(out["UHI_px"][:, 1, 1], [9.0, 8.0])

    # 40 m: no rural cell survives -> NaN.
    assert np.isnan(out["UHI_LR"][40.0][:, 1, 1]).all()

    # 60 m: only the dz=50 m rural cell survives.
    expected_rural_60 = np.array([22.0, 24.0]) - 0.0065 * 50.0
    assert np.allclose(out["rural_temperature_LR_mean"][60.0][:, 1, 1], expected_rural_60)

    # 120 m: both cells survive; mean dz = 75 m.
    expected_rural_120 = np.array([21.0, 23.0]) - 0.0065 * 75.0
    assert np.allclose(out["rural_temperature_LR_mean"][120.0][:, 1, 1], expected_rural_120)


def _write_synthetic_pgd(path, lon, lat):
    import xarray as xr

    ny, nx = lon.shape
    zeros = np.zeros((ny, nx), dtype=np.float32)
    ones = np.ones((ny, nx), dtype=np.float32)
    ds = xr.Dataset(
        {
            "SFX.FRAC_TOWN": (("y", "x"), zeros),
            "SFX.FRAC_NATURE": (("y", "x"), ones),
            "SFX.FRAC_SEA": (("y", "x"), zeros),
            "SFX.FRAC_WATER": (("y", "x"), zeros),
            "SFX.ZS": (("y", "x"), zeros),
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        }
    )
    ds.to_netcdf(path, engine="scipy")


def _write_synthetic_tas(path, lon, lat):
    import xarray as xr

    ny, nx = lon.shape
    ds = xr.Dataset(
        {
            "tas": (
                ("time", "y", "x"),
                np.full((2, ny, nx), 290.0, dtype=np.float32),
                {"units": "K"},
            )
        },
        coords={
            "time": np.arange(2),
            "lon": (("y", "x"), lon),
            "lat": (("y", "x"), lat),
        },
    )
    ds.to_netcdf(path, engine="scipy")


def test_grid_alignment_same_shape_coordinates_verified(tmp_path):
    from spatial_UHI_mask.grid_alignment import determine_grid_alignment

    y, x = np.meshgrid(np.arange(4), np.arange(5), indexing="ij")
    lon = 5.0 + 0.02 * x + 0.001 * y
    lat = 44.0 + 0.02 * y
    pgd = tmp_path / "pgd.nc"
    tas = tmp_path / "tas.nc"
    _write_synthetic_pgd(pgd, lon, lat)
    _write_synthetic_tas(tas, lon, lat)

    a = determine_grid_alignment(pgd, tas)
    assert a.source_shape == (4, 5)
    assert a.target_shape == (4, 5)
    assert a.method == "same_shape_coordinates_verified"
    assert a.is_full_grid
    assert np.isclose(a.max_lon_error_deg, 0.0)
    assert np.isclose(a.max_lat_error_deg, 0.0)


def test_grid_alignment_finds_contiguous_pgd_subset(tmp_path):
    from spatial_UHI_mask.grid_alignment import determine_grid_alignment

    y, x = np.meshgrid(np.arange(7), np.arange(8), indexing="ij")
    lon_full = 4.0 + 0.03 * x + 0.001 * y
    lat_full = 43.0 + 0.025 * y
    ys, xs = slice(1, 6), slice(2, 7)
    pgd = tmp_path / "pgd_crop.nc"
    tas = tmp_path / "tas_full.nc"
    _write_synthetic_pgd(pgd, lon_full[ys, xs], lat_full[ys, xs])
    _write_synthetic_tas(tas, lon_full, lat_full)

    a = determine_grid_alignment(pgd, tas)
    assert a.method == "contiguous_lonlat_match"
    assert (a.yslice.start, a.yslice.stop) == (1, 6)
    assert (a.xslice.start, a.xslice.stop) == (2, 7)
    assert a.target_shape == (5, 5)
