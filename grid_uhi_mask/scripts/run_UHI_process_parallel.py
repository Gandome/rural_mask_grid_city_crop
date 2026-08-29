#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public command-line runner for MOD_Mask / UHI version 2.0.0.

Examples
--------
Single ALPX3 yearly file::

    python grid_uhi_mask/scripts/run_UHI_process_parallel.py \
        --pgd /path/to/PGD.nc \
        --tas /path/to/tas_2000.nc \
        --output /path/to/UHI_MOD_MASK_V2 \
        --nproc 1

Multiple files, directories, and shell-style glob patterns are accepted after
``--tas``. The static MOD_Mask reference geometry is computed once per
sensitivity experiment and reused for all supplied temperature files.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import sys
from pathlib import Path

import numpy as np
import xarray as xr

THIS_DIR = Path(__file__).resolve().parent
GRID_PACKAGE_ROOT = THIS_DIR.parent
if str(GRID_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(GRID_PACKAGE_ROOT))

from spatial_UHI_mask import (  # noqa: E402
    build_masks,
    determine_grid_alignment,
    precompute_rural_references,
    process_files_parallel,
    to_yx,
)

VERSION = "2.0.0"


def log(msg):
    print(msg, flush=True)


def fmt_thr(v):
    return f"{v:.2f}".replace(".", "p")


def parse_number_list(text, cast=float):
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(cast(token))
    if not vals:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return vals


def resolve_tas_inputs(items):
    """Resolve explicit files, directories, or glob patterns."""
    resolved = []
    for raw in items:
        p = Path(raw).expanduser()
        if p.is_file():
            resolved.append(p.resolve())
        elif p.is_dir():
            resolved.extend(sorted(q.resolve() for q in p.glob("*.nc")))
            resolved.extend(sorted(q.resolve() for q in p.glob("*.nc4")))
        else:
            matches = [Path(q).resolve() for q in sorted(glob.glob(str(p)))]
            resolved.extend(q for q in matches if q.is_file())

    # Stable de-duplication.
    out, seen = [], set()
    for p in resolved:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    if not out:
        raise FileNotFoundError(f"No tas NetCDF files resolved from: {items}")
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description="MOD_Mask v2 rural-reference selection and UHI computation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pgd", required=True, type=Path, help="PGD NetCDF containing SFX.FRAC_* and SFX.ZS")
    p.add_argument(
        "--tas",
        required=True,
        nargs="+",
        help="tas NetCDF file(s), directories, or glob pattern(s)",
    )
    p.add_argument("--output", required=True, type=Path, help="base output directory")

    p.add_argument("--min-values", default="70", help="requested rural availability percentages, comma-separated")
    p.add_argument("--sea-water-thresholds", default="0.30", help="sea+water thresholds, comma-separated")
    p.add_argument("--urban-thresholds", default="0.20", help="urban-fraction thresholds, comma-separated")
    p.add_argument("--rural-thresholds", default="0.60", help="nature-fraction thresholds, comma-separated")

    p.add_argument("--nO", type=int, default=2, help="half-width of central exclusion square")
    p.add_argument("--initial-nbg", type=int, default=4, help="initial outer search half-width in grid cells")
    p.add_argument("--max-iterations", type=int, default=26, help="number of radii tested at each availability threshold")
    p.add_argument("--min-ratio-floor", type=float, default=50.0, help="hard lower bound for rural availability percentage")
    p.add_argument("--ratio-step", type=float, default=5.0, help="availability decrement in percentage points")
    p.add_argument("--height-limits", default="100,200,300,500", help="elevation filters in metres, comma-separated")
    p.add_argument("--lapse-rate", type=float, default=0.0065, help="temperature lapse rate in K m-1")

    p.add_argument("--nproc", type=int, default=1, help="file-level worker processes")
    p.add_argument("--progress-every", type=int, default=100, help="print reference-search progress every N urban cells; 0 disables")
    p.add_argument("--coord-tolerance", type=float, default=1.0e-5, help="maximum lon/lat mismatch in degrees when verifying grids")
    p.add_argument("--validate-only", action="store_true", help="validate files/grid alignment and masks, but do not compute UHI")
    p.add_argument("--version", action="version", version=f"MOD_Mask {VERSION}")
    return p


def _load_pgd(path):
    required = [
        "SFX.FRAC_TOWN",
        "SFX.FRAC_NATURE",
        "SFX.FRAC_SEA",
        "SFX.FRAC_WATER",
        "SFX.ZS",
    ]
    with xr.open_dataset(path) as pgd:
        missing = [v for v in required if v not in pgd]
        if missing:
            raise KeyError(f"PGD missing required variable(s): {missing}")
        return {
            "town": to_yx(pgd["SFX.FRAC_TOWN"]).load(),
            "nature": to_yx(pgd["SFX.FRAC_NATURE"]).load(),
            "sea": to_yx(pgd["SFX.FRAC_SEA"]).load(),
            "water": to_yx(pgd["SFX.FRAC_WATER"]).load(),
            "elevation": to_yx(pgd["SFX.ZS"]).load(),
        }


def _same_slice(a, b):
    return (
        a.yslice.start == b.yslice.start
        and a.yslice.stop == b.yslice.stop
        and a.xslice.start == b.xslice.start
        and a.xslice.stop == b.xslice.stop
        and a.source_shape == b.source_shape
        and a.target_shape == b.target_shape
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    pgd_file = args.pgd.expanduser().resolve()
    if not pgd_file.is_file():
        raise FileNotFoundError(f"PGD file not found: {pgd_file}")
    tas_files = resolve_tas_inputs(args.tas)

    min_values = parse_number_list(args.min_values, float)
    sea_water_thresholds = parse_number_list(args.sea_water_thresholds, float)
    urban_thresholds = parse_number_list(args.urban_thresholds, float)
    rural_thresholds = parse_number_list(args.rural_thresholds, float)
    height_limits = tuple(parse_number_list(args.height_limits, float))

    log("=" * 100)
    log(f"MOD_Mask / UHI version {VERSION}")
    log("=" * 100)
    log(f"PGD       : {pgd_file}")
    log(f"tas files : {len(tas_files)}")
    for f in tas_files[:5]:
        log(f"  - {f}")
    if len(tas_files) > 5:
        log(f"  ... {len(tas_files) - 5} more")
    log(f"Output    : {args.output}")

    # Validate alignment before any expensive rural-reference search.
    alignment = determine_grid_alignment(pgd_file, tas_files[0], args.coord_tolerance)
    for other in tas_files[1:]:
        other_alignment = determine_grid_alignment(pgd_file, other, args.coord_tolerance)
        if not _same_slice(alignment, other_alignment):
            raise ValueError(
                "All tas files must use the same horizontal grid/alignment. "
                f"First={alignment}; {other.name}={other_alignment}"
            )

    log("\nGrid compatibility: PASSED")
    log(f"  source tas shape : {alignment.source_shape}")
    log(f"  PGD target shape : {alignment.target_shape}")
    log(f"  alignment method : {alignment.method}")
    log(
        f"  tas slice        : y[{alignment.yslice.start}:{alignment.yslice.stop}], "
        f"x[{alignment.xslice.start}:{alignment.xslice.stop}]"
    )
    if np.isfinite(alignment.max_lon_error_deg):
        log(f"  max lon error    : {alignment.max_lon_error_deg:.3e} deg")
        log(f"  max lat error    : {alignment.max_lat_error_deg:.3e} deg")

    log("\nLoading PGD fields...")
    fld = _load_pgd(pgd_file)
    town, nature = fld["town"], fld["nature"]
    sea, water, elevation = fld["sea"], fld["water"], fld["elevation"]
    log(f"PGD shape: {town.shape}")

    for min_value, sw_thr, urb_thr, rur_thr in itertools.product(
        min_values, sea_water_thresholds, urban_thresholds, rural_thresholds
    ):
        exp_label = (
            f"Min{min_value:g}_sea{fmt_thr(sw_thr)}_urb{fmt_thr(urb_thr)}_rur{fmt_thr(rur_thr)}"
        )
        log("\n" + "=" * 100)
        log(f"Experiment: {exp_label}")
        log("=" * 100)

        sea_water_mask, land_mask, urban_mask, rural_mask = build_masks(
            town_fract=town,
            nature_fract=nature,
            sea_fract=sea,
            water_fract=water,
            urban_threshold=urb_thr,
            rural_threshold=rur_thr,
            sea_water_threshold=sw_thr,
        )
        urban_grid_points = np.argwhere(urban_mask.values)

        log(f"Total cells              : {int(np.prod(urban_mask.shape))}")
        log(f"Sea/water excluded cells : {int(sea_water_mask.sum().values)}")
        log(f"Land cells               : {int(land_mask.sum().values)}")
        log(f"Urban cells              : {int(urban_mask.sum().values)}")
        log(f"Rural cells              : {int(rural_mask.sum().values)}")

        if len(urban_grid_points) == 0:
            raise RuntimeError(f"No urban cells found for experiment {exp_label}")
        if int(rural_mask.sum().values) == 0:
            raise RuntimeError(f"No rural cells found for experiment {exp_label}")

        if args.validate_only:
            log("validate-only: mask construction PASSED; skipping reference search and UHI calculation")
            continue

        log("Precomputing static rural references...")
        references = precompute_rural_references(
            elevation=elevation.values,
            rural_frac=nature.values,
            urban_frac=town.values,
            sea_water_mask=sea_water_mask.values,
            urban_grid_points=urban_grid_points,
            rural_threshold=rur_thr,
            urban_threshold=urb_thr,
            min_value=min_value,
            nbg=args.initial_nbg,
            max_iterations=args.max_iterations,
            nO=args.nO,
            min_ratio_floor=args.min_ratio_floor,
            ratio_step=args.ratio_step,
            progress_every=args.progress_every,
        )
        n_ok = sum(int(ref.valid) for ref in references.values())
        log(f"Accepted rural references: {n_ok}/{len(references)} urban cells")

        output_dir = args.output.expanduser() / exp_label
        process_files_parallel(
            files=tas_files,
            elevation=elevation.values,
            output_path=output_dir,
            sea_water_mask=sea_water_mask.values,
            references=references,
            height_limits=height_limits,
            lapse_rate=args.lapse_rate,
            urban_threshold=urb_thr,
            rural_threshold=rur_thr,
            sea_water_threshold=sw_thr,
            min_value_requested=min_value,
            nO=args.nO,
            initial_nbg=args.initial_nbg,
            max_iterations=args.max_iterations,
            min_ratio_floor=args.min_ratio_floor,
            ratio_step=args.ratio_step,
            nproc=args.nproc,
            spatial_slice=(alignment.yslice, alignment.xslice),
        )

    log("\nAll requested MOD_Mask v2 computations completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
